#!/bin/bash
# Multi-node GRPO training for Qwen3-30B-A3B: 2 nodes x 8 GPUs = 16 GPUs, colocate.
# Run this script ON THE MASTER node. It starts the Ray head here, sshes into the
# worker (port 8081, host from /etc/mpi/hostfile) to join the Ray cluster, then
# submits the training job.
#
# Env already has slime + Megatron-LM; no pip install needed.
# Paths below are on shared storage reachable from both nodes.

set -ex

# ---------------- cluster ----------------
SLIME_DIR=/root/slime
MEGATRON_PATH=/root/Megatron-LM
SSH_PORT=8081
# Provide your W&B API key via env: `export WANDB_KEY=...` before running.
WANDB_KEY=${WANDB_KEY:?set WANDB_KEY env var (do not hardcode secrets)}

# Master node IP (reachable from worker; verified 10.107.229.32). Override via env.
MASTER_ADDR=${MASTER_ADDR:-10.107.229.32}

# Worker ssh host: the non-master line in /etc/mpi/hostfile.
WORKER_HOST=$(grep -v 'master' /etc/mpi/hostfile | awk 'NF{print $1; exit}')
echo "MASTER_ADDR=${MASTER_ADDR}  WORKER_HOST=${WORKER_HOST}  SSH_PORT=${SSH_PORT}"

SSH="ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no -o BatchMode=yes"

# ---------------- cleanup (master + worker) ----------------
# Same teardown as run-qwen3-30B-A3B.sh: pkill sglang, sleep 3, stop ray, sleep 3, retry.
CLEANUP_CMDS='pkill -9 sglang ; sleep 3 ; ray stop --force ; pkill -9 ray ; pkill -9 python ; sleep 3 ; pkill -9 ray ; pkill -9 python ; pkill -9 redis'
# worker first (in background), then master, so both teardowns overlap their sleeps.
${SSH} "${WORKER_HOST}" "${CLEANUP_CMDS}" || true &
eval "${CLEANUP_CMDS}" || true
wait

export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${WORKER_HOST}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ---------------- model config ----------------
source "${SLIME_DIR}/scripts/models/qwen3-30B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint /mnt/models/models/public/Qwen/Qwen3/Qwen3-30B-A3B
   --ref-load /mnt/data/data/home/czh/RL/Qwen3-30B-A3B_torch_dist/
   --load /mnt/data/data/home/czh/RL/Qwen3-30B-A3B_slime/
   --save /mnt/data/data/home/czh/RL/Qwen3-30B-A3B_slime/
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/zj-gpfs/home/czh/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 100
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /mnt/zj-gpfs/home/czh/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 32768
   --eval-top-p 1
)

# 16 GPUs: tp4 + ep8 (same proven shape as single-node, more DP across the 2 nodes)
#   non-expert dp = 16/(tp4*pp1*cp1) = 4 ; expert dp = 16/(etp1*ep8*pp1) = 2
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# Multi-node: drop CPU Adam offload (distributed optimizer shards opt states across DP).
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-30B-A3B-multinode-tp4ep4dp4-32k
   --wandb-key ${WANDB_KEY}
)

# One sglang engine per node (8 GPUs each), ep8 -> no cross-node inference comm.
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size 4
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   # optional: enable dp attention for higher rollout throughput
   --sglang-enable-dp-attention
   --sglang-dp-size 4
   --sglang-enable-dp-lm-head
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ---------------- start Ray cluster ----------------
# Head on master.
ray start --head --node-ip-address "${MASTER_ADDR}" --port 6379 --num-gpus 8 \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Worker joins; advertise its own node IP (first addr from `hostname -i`).
${SSH} "${WORKER_HOST}" "bash -lc '
  export no_proxy=127.0.0.1,localhost,${MASTER_ADDR};
  WIP=\$(hostname -i | awk \"{print \\\$1}\");
  echo joining ray from \$WIP;
  ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address \$WIP \
     --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
'"

# Wait until both nodes (16 GPUs) are registered.
python3 - <<'PY'
import ray, time
ray.init(address="auto")
for _ in range(60):
    g = ray.cluster_resources().get("GPU", 0)
    print("cluster GPUs:", g, flush=True)
    if g >= 16:
        break
    time.sleep(2)
assert ray.cluster_resources().get("GPU", 0) >= 16, "cluster did not reach 16 GPUs"
PY

# ---------------- submit training ----------------
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"TORCH_COMPILE_DISABLE\": \"1\",
    \"TORCHDYNAMO_DISABLE\": \"1\"
  }
}"
# If cross-node NCCL hangs at init, add the right fabric here, e.g.:
#   \"NCCL_SOCKET_IFNAME\": \"bond0\", \"NCCL_IB_HCA\": \"mlx5_cx6_0,mlx5_cx6_1\"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 2 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
