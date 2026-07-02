#!/bin/bash
# SMOKE TEST for DeepSeek-V2 ("021-32b") multinode GRPO: 2 nodes x 8 GPUs = 16 GPUs, colocate.
# Derived from run-deepseek-v2-multinode.sh with fast-path tweaks to validate the pipeline
# end to end (bridge HF load, YaRN factor-2/65536 on both Megatron & sglang, generation,
# weight sync, one train step) WITHOUT a full run:
#   - wandb disabled (no WANDB_KEY needed)
#   - num-rollout 1, small batch (rollout-batch-size 8 x n-samples 4 -> GBS 32) = one quick step
#   - eval disabled (EVAL_ARGS empty -> train.py skips the pre-train eval)
# Everything model/rope/length-related matches the real script so the smoke exercises the
# real config (rollout 32768, YaRN 65536, max-tokens-per-gpu 36864).
#
# PREREQ: the HF checkpoint dir must NOT contain `latest_checkpointed_iteration.txt`.

set -ex

# ---------------- cluster ----------------
SLIME_DIR=/root/slime
MEGATRON_PATH=/root/Megatron-LM
SSH_PORT=8081
# SMOKE TEST: wandb disabled, so no WANDB_KEY needed.

# Master node IP (reachable from worker). MUST be an IP, not the hostname: Ray uses
# this as --node-ip-address, and the sglang engines register with the router under it.
# This cluster INJECTS MASTER_ADDR as a *hostname* (env VC_MASTER_HOSTS); force the master
# IP unconditionally (overrides the injected hostname). EDIT THIS if the master changes.
MASTER_ADDR=10.200.100.205

# Worker ssh host: the non-master line in /etc/mpi/hostfile.
WORKER_HOST=$(grep -v 'master' /etc/mpi/hostfile | awk 'NF{print $1; exit}')
echo "MASTER_ADDR=${MASTER_ADDR}  WORKER_HOST=${WORKER_HOST}  SSH_PORT=${SSH_PORT}"

SSH="ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no -o BatchMode=yes"

# ---------------- apply megatron.patch on both nodes ----------------
# On a fresh machine the local Megatron-LM may not carry slime's patch (esp. the MLA
# v-pad fix this model needs). Apply it here on master AND worker before training.
# `patch --forward` is idempotent: it applies missing hunks and cleanly skips ones that
# are already applied, so this is safe whether the image pre-applied the patch or not.
APPLY_MEGATRON_PATCH="cd ${MEGATRON_PATH} && patch -p1 --forward < ${SLIME_DIR}/docker/patch/latest/megatron.patch >/dev/null 2>&1 ; grep -q _prepare_mla_core_attention_value megatron/core/transformer/multi_latent_attention.py && echo \"megatron.patch OK on \$(hostname)\" || echo \"WARN: megatron.patch NOT applied on \$(hostname)\""
eval "${APPLY_MEGATRON_PATCH}"
${SSH} "${WORKER_HOST}" "${APPLY_MEGATRON_PATCH}"

# ---------------- cleanup (master + worker) ----------------
CLEANUP_CMDS='pkill -9 sglang ; sleep 3 ; ray stop --force ; pkill -9 ray ; pkill -9 python ; sleep 3 ; pkill -9 ray ; pkill -9 python ; pkill -9 redis'
${SSH} "${WORKER_HOST}" "${CLEANUP_CMDS}" || true &
eval "${CLEANUP_CMDS}" || true
wait

export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${WORKER_HOST}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ---------------- model config ----------------
source "${SLIME_DIR}/scripts/models/deepseek-v2.sh"

# HF checkpoint (DeepseekV2ForCausalLM) — YaRN sibling dir (rope factor 2 / max_pos 65536).
HF_CKPT="/mnt/data/data/home/czh/RL/dsv2-021-yarn2-65536"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${HF_CKPT}"
   --load "${HF_CKPT}"
   --save /mnt/data/data/home/czh/RL/DeepSeek-V2-021_slime/
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/zj-gpfs/home/czh/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 1
   # SMOKE: small batch for a fast single step (32 samples).
   --rollout-batch-size 4
   --n-samples-per-prompt 2
   --rollout-max-response-len 32768
   --rollout-max-context-len 65536
   --rollout-temperature 1

   --global-batch-size 8
   --balance-data
)

# SMOKE: no eval at all (eval_interval unset -> train.py skips the pre-train eval).
# Eval correctness follows from rollout: same generation path; 49152+prompt < 65536 won't 400.
EVAL_ARGS=()

# 16 GPUs: tp4 + ep8 (80 experts / ep8 = 10 experts/rank), pp1, cp1.
# 16 GPUs: tp8 x cp2 (non-expert dp=1), ep8 x etp1 (expert dp=2). tp8 halves per-GPU
# attention/logits activations (sequence-parallel), cp2 splits the 32k sequence across
# 2 ranks -> another ~2x activation cut, letting rollout 32768 train on 80GB.
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 4
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   # >= longest single (prompt + response): ~1468 + 32768 = 34236. 36864 with headroom.
   --max-tokens-per-gpu 16384
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

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   # Offload optimizer states (fp32 master + Adam m/v, ~23GB/GPU) to CPU. Frees GPU for
   # the long-sequence training activations. --use-precision-aware-optimizer is required
   # by Megatron; --overlap-... hides the CPU step + D2H/H2D behind compute.
   --optimizer-cpu-offload
   --use-precision-aware-optimizer
   --optimizer-offload-fraction 1.0
   --overlap-cpu-optimizer-d2h-h2d
)

# SMOKE TEST: wandb disabled.
WANDB_ARGS=()

# One sglang engine per node (4 GPUs each), ep4 + dp-attention (best for MLA models).
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size 4
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-enable-dp-attention
   --sglang-dp-size 4
   --sglang-enable-dp-lm-head
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-backend flash
   # Load the HF checkpoint directly through megatron.bridge (no torch_dist convert).
   --megatron-to-hf-mode bridge
)

# ---------------- start Ray cluster ----------------
ray start --head --node-ip-address "${MASTER_ADDR}" --port 6379 --num-gpus 8 \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

${SSH} "${WORKER_HOST}" "bash -lc '
  export no_proxy=127.0.0.1,localhost,${MASTER_ADDR};
  WIP=\$(hostname -i | awk \"{print \\\$1}\");
  echo joining ray from \$WIP;
  ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address \$WIP \
     --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
'"

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
    \"TORCHDYNAMO_DISABLE\": \"1\",
    \"SLIME_OOM_SNAPSHOT\": \"1\"
  }
}"

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
