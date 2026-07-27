#!/bin/bash
# Multi-node GRPO training for DeepSeek-V2 ("021-32b" SFT checkpoint): 8 nodes x 8 GPUs = 64 GPUs, colocate.
#
# SAME-SCRIPT MODE: launch this EXACT script on BOTH nodes simultaneously (the cluster
# job launcher runs it on every pod). PREREQ: /root/slime and /root/Megatron-LM must be
# identical on both nodes (copied ahead of time -- they are node-local disks), and the
# MASTER_ADDR env var must be set on every node (the launcher injects it; hostname or IP
# both work). Each node detects its role by comparing MASTER_ADDR against its own IPs:
#   master: cleans up, applies megatron.patch locally, starts the Ray head, waits for
#           all GPUs to join, submits the training job.
#   worker: cleans up, applies megatron.patch locally, waits for the Ray head, joins,
#           and stays alive until the head goes away (so the launcher pod doesn't exit).
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

# ---------------- cluster / role detection ----------------
# SLIME_DIR from env (defaults to /root/slime if unset).
SLIME_DIR=${SLIME_DIR:-/root/slime}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

# Run everything from SLIME_DIR: the ray job agent inherits the cwd of `ray start`,
# so the train.py entrypoint resolves relative to WHERE RAY WAS STARTED, not where
# `ray job submit` runs (2026-07-12: launching from scripts/ broke train.py lookup).
cd "${SLIME_DIR}"

# The ONLY cluster input is the MASTER_ADDR env var (this cluster injects it as the
# master HOSTNAME via VC_MASTER_HOSTS; setting it manually to a hostname or an IP also
# works). No /etc/mpi/hostfile dependency.
[ -n "${MASTER_ADDR:-}" ] || { echo "FATAL: MASTER_ADDR env var not set"; exit 1; }

# Number of nodes from env (NNODES=4 for the 4-node cluster, 8 for 64 GPUs, ...).
NNODES=${NNODES:-8}
TOTAL_GPUS=$((NNODES * 8))
echo "NNODES=${NNODES}  TOTAL_GPUS=${TOTAL_GPUS}"
MASTER_HOST=${MASTER_ADDR%%,*}   # first entry if comma-separated

# Resolve to an IP. Ray uses it as --node-ip-address and the sglang engines register
# with the router under it; mixed hostname+IP worker URLs break the router pool, so a
# bare hostname must be resolved before use.
if echo "${MASTER_HOST}" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'; then
   MASTER_ADDR=${MASTER_HOST}
else
   MASTER_ADDR=$(getent hosts "${MASTER_HOST}" | awk '{print $1; exit}')
fi
[ -n "${MASTER_ADDR}" ] || { echo "FATAL: cannot resolve master IP from ${MASTER_HOST}"; exit 1; }

# Role: this node is the master iff one of its own IPs equals MASTER_ADDR.
# NOTE: must be `hostname -I` (ALL addresses) -- these nodes have many NICs and
# `hostname -i` returns only one of them (not necessarily the master-network one).
if hostname -I | tr ' ' '\n' | grep -qx "${MASTER_ADDR}"; then ROLE=master; else ROLE=worker; fi
echo "ROLE=${ROLE}  MASTER_ADDR=${MASTER_ADDR}"

export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${MASTER_HOST}"

# ---------------- cleanup (each node cleans itself) ----------------
CLEANUP_CMDS='pkill -9 sglang ; sleep 3 ; ray stop --force ; pkill -9 ray ; pkill -9 python ; sleep 3 ; pkill -9 ray ; pkill -9 python ; pkill -9 redis'
eval "${CLEANUP_CMDS}" || true

# ---------------- host free-memory guard (both roles) ----------------
# Colocate offloads ~110GB to host per node within seconds at every step boundary. On
# long-uptime nodes the page cache eats MemFree (available stays high but free drops to
# tens of GB), and the kernel cannot reclaim fast enough for that allocation burst:
# torch_memory_saver pause then fails with "cudaError 1 (invalid argument)" and kills all
# ranks on the node (2026-07-07 03:58 crash: master free 44GB -> 8/8 ranks died; worker
# free 450GB -> 8/8 fine). Drop caches now and keep MemFree > 250GB with a watchdog.
sync && { echo 3 > /proc/sys/vm/drop_caches; } 2>/dev/null || true
cat > /root/free_watchdog.sh <<'WDEOF'
while true; do
  f=$(awk '/MemFree/{print int($2/1048576)}' /proc/meminfo)
  if [ "$f" -lt 250 ]; then
    sync
    echo 3 > /proc/sys/vm/drop_caches
    echo "$(date +%F_%T) dropped caches (free was ${f}GB)"
  fi
  sleep 120
done
WDEOF
pkill -f 'bash /root/free_watchdog.sh' 2>/dev/null || true
setsid nohup bash /root/free_watchdog.sh >> /root/free_watchdog.log 2>&1 < /dev/null &

# ---------------- apply megatron.patch locally (both roles) ----------------
# Idempotent grep-guard; /root/Megatron-LM is node-local so each node patches itself.
if ! grep -q _prepare_mla_core_attention_value "${MEGATRON_PATH}/megatron/core/transformer/multi_latent_attention.py" 2>/dev/null; then
   # NOTE: patch --forward exits 1 when some hunks are already applied (fresh images ship
   # Megatron with part of the patch baked in) -- that is fine, the grep below is the real
   # gate, so don't let set -e kill the script here.
   ( cd "${MEGATRON_PATH}" && patch -p1 --forward --batch -i "${SLIME_DIR}/docker/patch/latest/megatron.patch" >/dev/null 2>&1 </dev/null ) || true
fi
grep -q _prepare_mla_core_attention_value "${MEGATRON_PATH}/megatron/core/transformer/multi_latent_attention.py" \
   && echo "megatron.patch OK on $(hostname)" || { echo "FATAL: megatron.patch NOT applied on $(hostname)"; exit 1; }

# ---------------- worker: join the Ray cluster and stay alive ----------------
if [ "${ROLE}" = "worker" ]; then
   until (exec 3<>"/dev/tcp/${MASTER_ADDR}/6379") 2>/dev/null; do echo "waiting for ray head..."; sleep 5; done
   WIP=$(hostname -i | awk '{print $1}')
   echo "joining ray from ${WIP}"
   ray start --address="${MASTER_ADDR}:6379" --num-gpus 8 --node-ip-address "${WIP}" \
      --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
   # Keep the pod alive while the head is up; exit when the job/head is gone.
   sleep 30
   while (exec 3<>"/dev/tcp/${MASTER_ADDR}/6379") 2>/dev/null; do sleep 30; done
   echo "ray head gone, worker exiting"
   exit 0
fi

# ---------------- master only from here on ----------------
# Provide your W&B API key via env: `export WANDB_KEY=...` before running.
WANDB_KEY=${WANDB_KEY:?set WANDB_KEY env var (do not hardcode secrets)}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ---------------- model config ----------------
source "${SLIME_DIR}/scripts/models/deepseek-v2.sh"

# HF checkpoint (DeepseekV2ForCausalLM). Loaded directly via the bridge.
# Model A "cot-geo" (checkpoint-1472) WON the base-model comparison (2026-07-03):
#   A: eval/aime 0.700, trunc 0.20, rollout median 14.6K  |  old cot-glm51: 0.454 / 0.41
#   B (a4b-lcpt base): never emits EOS on chat prompts -> ~100% truncation, unusable for RL.
# This is a YaRN sibling dir: symlinks to the original SFT weights + a config.json with
# rope_scaling.factor=2.0 and max_position_embeddings=65536. Both the megatron.bridge
# (Megatron MLA, via MLA_ROPE_SCALING_MAPPING) and sglang read rope from this config, so
# train / rollout / eval all run consistently at YaRN factor-2 / 65536 context.
HF_CKPT="/mnt/data/data/home/czh/RL/dsv2-021A-cotgeo-yarn2-65536"
SAVE_DIR=${SAVE_DIR:-/mnt/data/data/home/czh/RL/DeepSeek-V2-021A_slime_dapo_overlong}
PROMPT_DATA=${PROMPT_DATA:-/mnt/zj-gpfs/home/czh/dapo-math-17k.jsonl}
EVAL_DATA_AIME=${EVAL_DATA_AIME:-/mnt/zj-gpfs/home/czh/aime-2024.jsonl}

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   # RESUME mode: --load points at the slime save dir (holds latest_checkpointed_iteration.txt
   # = 99 + iter_0000099/). In bridge mode slime detects the Megatron checkpoint there and
   # loads weights + optimizer + RNG, resuming at rollout_id 100 (loaded_id 99 + 1); the
   # rollout/ subdir restores the dataloader position too. --ref-load STAYS on the HF
   # checkpoint so the KL anchor remains the initial policy. To start a FRESH run instead,
   # point --load back at "${HF_CKPT}" (start_rollout_id resets to 0).
   --ref-load "${HF_CKPT}"
   --load "${SAVE_DIR}/"
   --save "${SAVE_DIR}/"
   # Each checkpoint is ~413GB (bf16 weights + fp32 optimizer distcp). WARNING: no auto-cleanup
   # -- checkpoints accumulate. The save FS has only ~1.7TB free (shared, 99% full), so ~4
   # checkpoints fill it; delete old iter_* manually or raise --save-interval if it fills up.
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   # Standard DAPO Soft Overlong Punishment: deepscaler 0/1 accuracy is stored as the raw
   # filtering/metric signal, mapped to a -1/+1 training reward, then given a linear length
   # penalty over the last 16000 tokens before the cap (0 -> -1). Dynamic sampling filters
   # on raw accuracy, so shaping cannot make an all-correct/all-wrong group pass. Eval rewards
   # remain pure 0/1 accuracy via the is_eval metadata guard. Buffer: env DAPO_OVERLONG_BUFFER.
   --custom-rm-path slime.rollout.rm_hub.dapo_overlong.custom_rm
   # ABSOLUTE endpoint, not a delta: train.py loops range(start_rollout_id, num_rollout).
   # Resuming at rollout_id 100, so 200 = another 100 steps (saves land at 119/139/159/179/199
   # per --save-interval 20). Raise this if you want to train further.
   --num-rollout 200
   # Standard DAPO dynamic sampling: keep 64 groups whose raw 0/1 accuracy is non-constant.
   # Generate each 256-group wave to completion, then select valid groups in original prompt
   # order. This avoids the short-response bias caused by accepting FIRST_COMPLETED groups.
   # Soft-overlong shaping affects the training reward but not this correctness filter.
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --over-sampling-batch-size 256
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_raw_reward_nonzero_std
   --dynamic-sampling-wait-all
   # 64000 fills the 65536 YaRN window: longest dapo prompt is 1468 tok, so 1468+64000=65468
   # <= 65536 (no sample truncated by context). cp4 splits the longest single sequence to
   # 65468/4 = 16367 tok/rank <= --max-tokens-per-gpu 16384, so peak activation is unchanged.
   # (Measured on model A: only ~8% of advantage samples exceed 32768; if rollout wall clock
   # ever needs a ~30-40% cut, dropping this to 32768 costs only ~4-5% of positive signal.)
   --rollout-max-response-len 64000
   --rollout-max-context-len 65536
   --rollout-temperature 1

   # 64 groups x 8 samples = 512 trajectories, split into two GBS-256 optimizer steps.
   --num-steps-per-rollout 2
   --balance-data
)

EVAL_CONFIG_RENDERED=/tmp/eval-config-deepseek-v2.yaml
export EVAL_DATA_AIME
envsubst < "${SLIME_DIR}/scripts/eval-config-deepseek-v2.yaml" > "${EVAL_CONFIG_RENDERED}"

EVAL_ARGS=(
   --eval-interval 10
   # Dataset config moved to YAML: it also tags eval samples with metadata is_eval=true so
   # the dapo_overlong custom RM skips reward shaping during eval (scores = pure accuracy).
   --eval-config "${EVAL_CONFIG_RENDERED}"
   --eval-max-context-len 65536
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
   # 1e-3 KL anchor to the ref (initial) policy. Run zkoev8r5 (coef 0.00) drifted: entropy
   # rose monotonically 0.36 -> 0.53 over 96 steps while eval/aime fell 0.69 -> 0.556 and
   # eval median length grew 26.7K -> 36.8K. clip-higher (0.28) keeps pushing low-prob
   # tokens up and nothing pulls back; this anchor is the restoring force. Escalation
   # ladder if entropy still climbs after 20-40 steps: 5e-3 / 1e-2 (GRPO paper default
   # 0.04 = strong anchor), or cut --eps-clip-high to 0.22-0.24 instead. Watch rollout/kl:
   # healthy is a slow climb staying < ~0.05/token; pinned at 0 means anchor too tight.
   --kl-loss-coef 1e-3
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --eps-clip-c 10.0
   # DAPO Token-Level PG Loss: default sample-level normalization averages the loss inside
   # each sample first, so a 64K-token negative sample contributes ~1/64000 gradient per
   # token while short positives hit at full weight -- long generations were rewarded at
   # full strength but punished at a discount, feeding the length runaway above. Per-token
   # normalization makes every token weigh the same regardless of sample length.
   --calculate-per-token-loss
   # R3, Rollout Routing Replay (arXiv 2510.11370): sglang returns the per-token/per-layer
   # top-7 routed-expert ids (enable_return_routed_experts is set automatically) and the
   # training forward replays them instead of re-doing argmax, so gradients are computed on
   # the exact expert path that generated the tokens. MoE router flips are the dominant
   # source of train/train_rollout_logprob_abs_diff (0.027 -> 0.034 and growing on
   # zkoev8r5); expect it to drop sharply on the first step after enabling -- if it
   # doesn't, R3 is not taking effect. Dense layer 0 is skipped via moe_layer_freq; cp4 +
   # sequence-parallel slicing handled in fill_routing_replay (megatron_utils/actor.py).
   # Cost: 40 layers x 7 x int32 ~= 1.1KB/token, ~23GB host RAM per rollout step (free
   # watchdog above already guards MemFree).
   --use-rollout-routing-replay
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --override-opt-param-scheduler
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
    # The CPU optimizer buffers are pinned by default (~23GB/GPU x8 ranks = ~184GB/node of
   # pinned host memory). On 1TB-RAM nodes this starves torch_memory_saver.pause(), whose
   # cudaMallocHost then returns nullptr -> "cudaError error: 1 (invalid argument)" and the
   # train actor dies (THUDM/slime#1786). Unpinned makes the once-per-rollout optimizer
   # D2H/H2D copies slower, which is negligible here.
   --no-pin-cpu-grads
   --no-pin-cpu-params
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group deepseek-v2-021A-dapo-tp4cp4ep16
   --wandb-key ${WANDB_KEY}
)

# 4 GPUs/engine, ep4, dp-attention dp4 — the measured optimum on this A100 16-GPU cluster
# (full-step A/B tests 2026-07-03, all vs this baseline's step_time 4464s):
#   - dp2 (attention tp2): single-stream tail +24%, BUT halved KV pools -> 14 retracts +
#     queueing at the 1024-concurrent wave and ~45% slower valid-group collection. LOST.
#   - 8-GPU engine + ep8: single-stream +1.5% only (decode at bs=1 is kernel-latency-bound,
#     not bandwidth-bound; fewer experts/GPU doesn't cut kernel count). No win.
#   - --sglang-enable-torch-compile: no gain, CUDA graphs already cover launch overhead.
# router policy random (not the sglang default cache_aware): prompts are ~130 tok so prefix
# caching is worthless, while cache_aware pinned each prompt's 8 samples to one engine and
# clumped long groups (KV 0.94-0.99 + retracts on one engine while others idled).
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
   # MLA head_dim_qk=192 (128 nope + 64 rope) != head_dim_v=128. Raw TE on A100 (sm80) can't
   # serve qk!=v (FA2 needs qk==v; FA3 needs Hopper; cuDNN fused rejects 192/128 on sm80) ->
   # would fall to unfused and OOM at long seq. The MLA v-pad hunk in docker/patch/latest/
   # megatron.patch pads V to 192 so qk==v==192 and flash works on A100 (applied at the top
   # of this script on both nodes). See multi_latent_attention.py _prepare_mla_core_attention_value.
   --attention-backend flash
   # Load the HF checkpoint directly through megatron.bridge (no torch_dist convert).
   --megatron-to-hf-mode bridge
)

# ---------------- start Ray cluster (worker joins by itself, see the worker branch) ----------------
ray start --head --node-ip-address "${MASTER_ADDR}" --port 6379 --num-gpus 8 \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Wait for the workers (running this same script) to join. Cold starts copy the code
# tree from shared storage first (minutes when several nodes copy in parallel), so
# allow up to ~30 min of skew.
export TOTAL_GPUS
python3 - <<'PY'
import ray, time
ray.init(address="auto")
import os
target = int(os.environ["TOTAL_GPUS"])
for _ in range(900):
    g = ray.cluster_resources().get("GPU", 0)
    print("cluster GPUs:", g, flush=True)
    if g >= target:
        break
    time.sleep(2)
assert ray.cluster_resources().get("GPU", 0) >= target, f"cluster did not reach {target} GPUs"
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

# train.py is resolved relative to the job cwd; run from SLIME_DIR so the script works
# no matter where it was launched from.
cd "${SLIME_DIR}"
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes ${NNODES} \
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
