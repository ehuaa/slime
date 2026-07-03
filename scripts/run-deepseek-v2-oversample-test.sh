#!/bin/bash
# OVERSAMPLING TEST: model A + dynamic-sampling filter (check_reward_nonzero_std).
# (pre-train eval on aime + full rollout of 512 samples + one GBS-512 train step), to compare
# rollout/eval truncation ratios and pick the better base model for RL.
#
# Usage:  bash scripts/run-deepseek-v2-cmp.sh A   # cot-geo-46w SFT  (checkpoint-1472)
#         bash scripts/run-deepseek-v2-cmp.sh B   # 021-32B-A4B lcpt-1208
#
# Differences vs run-deepseek-v2-multinode.sh (besides the checkpoint):
#   - num-rollout 1 (single step), wandb off (read metrics from the log)
#   - --sglang-mem-fraction-static 0.85  (0.7 hit KV 0.94-0.96 peaks and 7 retracts at 512-way)
#   - --sglang-router-policy random      (cache_aware pinned whole 8-sample groups per engine ->
#     severe token imbalance; prompts are ~130 tok so prefix caching was worthless anyway)
#   - --sglang-dp-size 4 (16 KV pools, max aggregate capacity for the 2048-concurrent wave)

set -ex

MODEL_TAG=${1:?usage: run-deepseek-v2-cmp.sh A|B}
case "${MODEL_TAG}" in
  A) HF_CKPT="/mnt/data/data/home/czh/RL/dsv2-021A-cotgeo-yarn2-65536" ;;
  B) HF_CKPT="/mnt/data/data/home/czh/RL/dsv2-021B-a4b-yarn2-65536" ;;
  *) echo "unknown model tag ${MODEL_TAG}"; exit 1 ;;
esac
echo "=== comparison run for model ${MODEL_TAG}: ${HF_CKPT} ==="

# ---------------- cluster ----------------
SLIME_DIR=/root/slime
MEGATRON_PATH=/root/Megatron-LM
SSH_PORT=8081
MASTER_ADDR=10.200.100.205

WORKER_HOST=$(grep -v 'master' /etc/mpi/hostfile | awk 'NF{print $1; exit}')
echo "MASTER_ADDR=${MASTER_ADDR}  WORKER_HOST=${WORKER_HOST}  SSH_PORT=${SSH_PORT}"

SSH="ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no -o BatchMode=yes"

# ---------------- apply megatron.patch on both nodes ----------------
APPLY_MEGATRON_PATCH="if ! grep -q _prepare_mla_core_attention_value ${MEGATRON_PATH}/megatron/core/transformer/multi_latent_attention.py 2>/dev/null ; then ( cd ${MEGATRON_PATH} && patch -p1 --forward --batch -i ${SLIME_DIR}/docker/patch/latest/megatron.patch >/dev/null 2>&1 </dev/null ) ; fi ; grep -q _prepare_mla_core_attention_value ${MEGATRON_PATH}/megatron/core/transformer/multi_latent_attention.py && echo \"megatron.patch OK on \$(hostname)\" || echo \"WARN: megatron.patch NOT applied on \$(hostname)\""
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

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${HF_CKPT}"
   --load "${HF_CKPT}"
   --save "/mnt/data/data/home/czh/RL/DeepSeek-V2-021-cmp${MODEL_TAG}_slime/"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/zj-gpfs/home/czh/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   # ONE complete step for the comparison.
   --num-rollout 1
   --over-sampling-batch-size 128
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 64000
   --rollout-max-context-len 65536
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=()   # oversample test: no eval (baseline already measured)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 4
   --expert-model-parallel-size 16
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
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

   --optimizer-cpu-offload
   --use-precision-aware-optimizer
   --optimizer-offload-fraction 1.0
   --overlap-cpu-optimizer-d2h-h2d
)

# comparison run: wandb off, metrics read from the log.
WANDB_ARGS=()

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.85
   --sglang-ep-size 4
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-enable-dp-attention
   --sglang-dp-size 4
   --sglang-enable-dp-lm-head
   --sglang-router-policy random
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-backend flash
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
    \"TORCHDYNAMO_DISABLE\": \"1\"
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
