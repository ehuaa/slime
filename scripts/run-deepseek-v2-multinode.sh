#!/bin/bash
# Multi-node GRPO training for DeepSeek-V2 ("021-32b" SFT checkpoint): 2 nodes x 8 GPUs = 16 GPUs, colocate.
# Run this script ON THE MASTER node. It starts the Ray head here, sshes into the
# worker (port 8081, host from /etc/mpi/hostfile) to join the Ray cluster, then
# submits the training job.
#
# Model is a DeepseekV2ForCausalLM HF checkpoint loaded DIRECTLY via megatron.bridge
# (--megatron-to-hf-mode bridge). No HF->torch_dist conversion is needed: --load /
# --ref-load point straight at the HF folder and the bridge builds the Megatron model.
#
# PREREQ: the HF checkpoint dir must NOT contain `latest_checkpointed_iteration.txt`
# (slime would mis-route it to the Megatron loader). Rename it away before training.
#
# Env already has slime + Megatron-LM + megatron.bridge; no pip install needed.

set -ex

# ---------------- cluster ----------------
SLIME_DIR=/root/slime
MEGATRON_PATH=/root/Megatron-LM
SSH_PORT=8081
# Provide your W&B API key via env: `export WANDB_KEY=...` before running.
WANDB_KEY=${WANDB_KEY:?set WANDB_KEY env var (do not hardcode secrets)}

# Master node IP (reachable from worker). MUST be an IP, not the hostname: Ray uses
# this as --node-ip-address, and the sglang engines register with the router under it.
# If the master advertises a hostname while the worker advertises an IP (its `hostname -i`),
# the router ends up with mixed hostname+IP worker URLs and the worker pool breaks.
# This cluster INJECTS MASTER_ADDR as a *hostname* (env VC_MASTER_HOSTS); Ray and the
# sglang engines must register under a consistent IP. Force the master IP unconditionally
# (overrides the injected hostname). EDIT THIS if the master node changes.
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

# HF checkpoint (DeepseekV2ForCausalLM). Loaded directly via the bridge.
# This is a YaRN sibling dir: symlinks to the original SFT weights + a config.json with
# rope_scaling.factor=2.0 and max_position_embeddings=65536. Both the megatron.bridge
# (Megatron MLA, via MLA_ROPE_SCALING_MAPPING) and sglang read rope from this config, so
# train / rollout / eval all run consistently at YaRN factor-2 / 65536 context.
# The model was SFT'd at 32k but needs YaRN factor-2 (65536) for good long-CoT math (AIME).
# Original SFT weights (.../checkpoint-1521) are untouched.
HF_CKPT="/mnt/data/data/home/czh/RL/dsv2-021-yarn2-65536"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   # First run: init both actor (--load) and reference (--ref-load) from the HF checkpoint.
   # To RESUME after a save, change --load to the slime save dir below (which then holds
   # iter_XXXXXXX/), and keep --ref-load on the HF checkpoint.
   --ref-load "${HF_CKPT}"
   --load "${HF_CKPT}"
   --save /mnt/data/data/home/czh/RL/DeepSeek-V2-021_slime/
   # Each checkpoint is ~413GB (bf16 weights + fp32 optimizer distcp). WARNING: no auto-cleanup
   # -- checkpoints accumulate. The save FS has only ~1.7TB free (shared, 99% full), so ~4
   # checkpoints fill it; delete old iter_* manually or raise --save-interval if it fills up.
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/zj-gpfs/home/czh/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 100
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   # 64000 fills the 65536 YaRN window: longest dapo prompt is 1468 tok, so 1468+64000=65468
   # <= 65536 (no sample truncated by context). cp4 splits the longest single sequence to
   # 65468/4 = 16367 tok/rank <= --max-tokens-per-gpu 16384, so peak activation is unchanged.
   --rollout-max-response-len 64000
   --rollout-max-context-len 65536
   --rollout-temperature 1

   --global-batch-size 512
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /mnt/zj-gpfs/home/czh/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   # longest aime prompt is 371 tok, so 371+64000=64371 <= 65536 context.
   --eval-max-response-len 64000
   --eval-max-context-len 65536
   --eval-top-p 1
)

# 16 GPUs: tp4 x cp4 (non-expert dp=1), ep16 x etp1 (expert dp=1). cp4 splits the long
# sequence across 4 ranks so each GPU's per-microbatch tokens (and activation) drop ~4x,
# which — together with optimizer CPU offload below — lets rollout 64000 train on 80GB.
#   non-expert: tp4 x cp4 x dp1 = 16 ; expert: etp1 x ep16 x edp1 = 16
# ep16 (edp=1) is REQUIRED: with ep8/edp2 the expert param + fp32-grad buffers (~21GB) were
# REPLICATED across the 2 nodes and full GBS OOM'd (an OOM snapshot showed PyTorch pinned at
# ~62/80GB, static 37GB). ep16 removes the replication -> per-rank expert state ~10.5GB,
# static drops to ~26.5GB, and the validated smoke ran a full GBS-32 step at ~51GB with
# ~13GB headroom. Tradeoff: the MoE alltoall now spans both nodes (unavoidable without
# per-node expert replication). Peak activation is bounded by --max-tokens-per-gpu, so it
# does NOT grow with --global-batch-size or --rollout-max-response-len (only more/longer
# microbatches, i.e. more time). If you ever OOM: lower --max-tokens-per-gpu, or add
# --pipeline-model-parallel-size 2, or lower --sglang-mem-fraction-static.
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
   # With cp4 the longest sequence (1468+64000=65468) splits to ~16367 tokens/rank, which
   # fills one microbatch at ~16384. Raising this raises per-GPU activation -> OOM risk.
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

   # Offload optimizer states (fp32 master + Adam m/v, ~23GB/GPU) to CPU. Required to fit
   # long-sequence training on 80GB; ~1 optimizer step per rollout so the CPU step is negligible
   # vs generation, and --overlap-... hides the D2H/H2D transfers.
   --optimizer-cpu-offload
   --use-precision-aware-optimizer
   --optimizer-offload-fraction 1.0
   --overlap-cpu-optimizer-d2h-h2d
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group deepseek-v2-021-multinode-tp4ep16-bridge
   --wandb-key ${WANDB_KEY}
)

# One sglang engine per node (4 GPUs each), ep4 + dp-attention (best for MLA models).
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.85
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
   # MLA head_dim_qk=192 (128 nope + 64 rope) != head_dim_v=128. Raw TE on A100 (sm80) can't
   # serve qk!=v (FA2 needs qk==v; FA3 needs Hopper; cuDNN fused rejects 192/128 on sm80) ->
   # would fall to unfused and OOM at long seq. The MLA v-pad hunk in docker/patch/latest/
   # megatron.patch pads V to 192 so qk==v==192 and flash works on A100 (applied at the top
   # of this script on both nodes). See multi_latent_attention.py _prepare_mla_core_attention_value.
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
