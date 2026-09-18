#!/bin/bash
# Qwen3.8-27B GRPO on 4x8 A100. See the knobs block below for what is tunable.

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex
export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(dirname "${SCRIPT_DIR}")"
# ray job's driver cwd is inherited from the cwd of `ray start`, not of `ray job submit`
cd "${SLIME_DIR}"

# ---------------------------------------------------------------- paths
MODEL_FOLDER=${MODEL_FOLDER:-/mnt/models/models/public/Qwen}
SAVE_FOLDER=${SAVE_FOLDER:-/mnt/data/data/home/zhh/ckpt}
DAPO_DATA=${DAPO_DATA:-/mnt/data/data/home/zhh/data/data4RL/math/processed/dapo_math_17k.jsonl}
AIME_DATA=${AIME_DATA:-/mnt/data/data/datasets/geo-rl/eval/aime-2024.jsonl}

# ---------------------------------------------------------------- cluster
# k8s names in /etc/mpi/hostfile do not resolve; MASTER_ADDR/HOSTFILE carry bond0 IPs.
# (Temporarily restoring /tmp/resolv.conf.bak makes them resolvable long enough to look up.)
MASTER_ADDR=${MASTER_ADDR:-}
if [ -z "${MASTER_ADDR}" ]; then echo "MASTER_ADDR is not set (bond0 10.200.x IP)."; exit 1; fi
HOSTFILE=${HOSTFILE:-${SLIME_DIR}/hostfile.ip}
SSH_PORT=${SSH_PORT:-8081}          # 22 is the host sshd and rejects our key

ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-4}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
SOCKET_IFNAME=${SOCKET_IFNAME:-bond0}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

# ---------------------------------------------------------------- knobs
# Megatron parallelism. TP*PP*CP must divide 32, and MAX_TOK_PER_GPU*CP_SIZE must be
# >= MAX_RESP or a full-window sample cannot be packed into any bin at all.
# Defaults below are the winner of a 7-config sweep on 4x8 A100 (50-microbatch steady
# state, identical synthetic rollout). TP4/PP4/CP2 ran 7.70 s/microbatch vs 14.03 for
# the previous TP4/PP2/CP4 default -- 1.83x -- and used less memory (16-18GB vs 33.6GB
# after wake_up). What the sweep established:
#   * TP=4 is a hard floor. The logits buffer (max_tokens_per_gpu x vocab 248320) is
#     split by TP alone: TP2 and TP1 both died needing 7.58 GiB, and TP1 additionally
#     loses sequence-parallel (arguments.py:906 forces it off) -- TP1/PP8 needed 30.55 GiB.
#   * PP is what buys the speedup, not CP. F (TP4/PP2/CP2, same logits as B) OOMed in
#     custom_backward: PP2 keeps 32 layers/rank, so activation (tokens x layers) doubles
#     to 1048576. PP4 halves layers/rank to 17 and that is where the 1.83x comes from.
#   * max_tokens_per_gpu x CP is pinned to >= 65536 by the packer, so lowering CP forces
#     max_tokens_per_gpu up, which pushes activation back up. CP2+32768 only works at PP4.
#   * recompute must stay `full`. `selective` keeps MLP activations and recomputes
#     attention, but 48 of 64 layers here are GDN linear attention, so it preserves the
#     dominant term instead: it OOMed inside fla/ops/gated_delta_rule/wy_fast.py
#     (recompute_w_u_fwd). `none` OOMed too.
TP_SIZE=${TP_SIZE:-4}               # hard floor: vocab-parallel for the 248320 vocab
PP_SIZE=${PP_SIZE:-4}               # fewer layers/rank -> lower activation peak
CP_SIZE=${CP_SIZE:-2}
MAX_TOK_PER_GPU=${MAX_TOK_PER_GPU:-32768}   # x CP_SIZE must stay >= MAX_RESP
RECOMPUTE=${RECOMPUTE:-full}        # full | selective | none -- selective/none OOM on GDN
# sglang
ROLLOUT_MEM_UTILIZATION=${ROLLOUT_MEM_UTILIZATION:-0.75}
MAX_RUNNING_REQ=${MAX_RUNNING_REQ:-64}
CHUNKED_PREFILL=${CHUNKED_PREFILL:-4096}
# rollout shape
ROLLOUT_BS=${ROLLOUT_BS:-256}
N_SAMPLES=${N_SAMPLES:-8}
MAX_RESP=${MAX_RESP:-65536}
GLOBAL_BS=${GLOBAL_BS:-2048}
NUM_ROLLOUT=${NUM_ROLLOUT:-3000}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ---------------------------------------------------------------- per-node prep
cat > /tmp/slime_node_prep.sh <<PREP
set -e
MEGATRON_PATH="${MEGATRON_PATH}"
PREP
cat >> /tmp/slime_node_prep.sh <<'PREP'

ipcs -m | awk '$4 == 666 {print $2}' | while read -r shmid; do ipcrm -m "$shmid" || true; done

# Colocate offload bursts ~110GB to host per step boundary and torch_memory_saver's
# pause() needs physically contiguous pages. drop_caches frees pages but does NOT
# defragment, so compact too: on long-uptime nodes the high-order blocks run out even
# with ~900GB MemFree and pause() then dies with "cudaError 1 (invalid argument)".
sync && { echo 3 > /proc/sys/vm/drop_caches; } 2>/dev/null || true
{ echo 1 > /proc/sys/vm/compact_memory; } 2>/dev/null || true
echo "node $(hostname): uptime=$(awk '{printf "%d", $1/86400}' /proc/uptime)d order9/10=$(awk '/Normal/{print $(NF-1)"/"$NF}' /proc/buddyinfo | paste -sd' ' -)"
cat > /root/free_watchdog.sh <<'WDEOF'
while true; do
  f=$(awk '/MemFree/{print int($2/1048576)}' /proc/meminfo)
  if [ "$f" -lt 250 ]; then sync; echo 3 > /proc/sys/vm/drop_caches; echo "$(date +%F_%T) dropped caches (free was ${f}GB)"; fi
  sleep 120
done
WDEOF
pkill -f 'bash /root/free_watchdog.sh' 2>/dev/null || true
setsid nohup bash /root/free_watchdog.sh >> /root/free_watchdog.log 2>&1 < /dev/null &

# The image bakes slime's old megatron.patch edit into /root/Megatron-LM, dropping the
# `group` arg so PP p2p runs on the default world communicator -> deadlock when PP>1 and
# CP>1. Removing the hunk from the patch cannot undo an edit already on disk; repair here.
P2P_FILE="${MEGATRON_PATH}/megatron/core/pipeline_parallel/p2p_communication.py"
if [ "$(grep -c 'pipeline_rank, group,$' "${P2P_FILE}")" -ne 4 ]; then
   sed -i -E 's/^( +torch\.distributed\.(isend|irecv), tensor_(send|recv)_(prev|next), (prev|next)_pipeline_rank),$/\1, group,/' "${P2P_FILE}"
   echo "repaired _batched_p2p_ops group arg on $(hostname)"
fi
[ "$(grep -c 'pipeline_rank, group,$' "${P2P_FILE}")" -eq 4 ] \
   || { echo "FATAL: _batched_p2p_ops still drops the process group on $(hostname)"; exit 1; }
echo "_batched_p2p_ops passes group OK on $(hostname)"
PREP

bash /tmp/slime_node_prep.sh
if [ -f "${HOSTFILE}" ]; then
  for WORKER_IP in $(awk 'NF{print $1}' "${HOSTFILE}"); do
    [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]] && continue
    scp -P "${SSH_PORT}" -o StrictHostKeyChecking=no /tmp/slime_node_prep.sh root@"${WORKER_IP}":/tmp/slime_node_prep.sh
    ssh -n -p "${SSH_PORT}" -o StrictHostKeyChecking=no root@"${WORKER_IP}" "bash /tmp/slime_node_prep.sh"
  done
fi

# ---------------------------------------------------------------- args
source "${SCRIPT_DIR}/models/qwen3.5-27B.sh"   # Qwen3.8-27B shares the qwen3_5 architecture

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_FOLDER}/Qwen3.8-27B"
   --ref-load "${MODEL_FOLDER}/Qwen3.8-27B_torch_dist/"
)
# NO_SAVE=1: train.py force_syncs a save on the last rollout, and that checkpoint's
# OptimizerParamScheduler encodes ITS global-batch-size -- a later run with a different
# --global-batch-size then dies in load_state_dict with "class input value N and
# checkpoint value M ... do not match". Throwaway runs should not write one.
if [ -z "${NO_SAVE:-}" ]; then
   CKPT_ARGS+=( --load "${SAVE_FOLDER}/Qwen3.8-27B_slime/" --save "${SAVE_FOLDER}/Qwen3.8-27B_slime/" --save-interval ${SAVE_INTERVAL:-20} )
fi
# DEBUG_ROLLOUT_DATA replays a dumped/synthetic rollout; it implies debug_train_only
# (arguments.py:1797) so no sglang engine starts at all. Used to benchmark Megatron
# parallelism on byte-identical input without paying ~51 min per rollout.
if [ -n "${DEBUG_ROLLOUT_DATA:-}" ]; then
   CKPT_ARGS+=( --load-debug-rollout-data "${DEBUG_ROLLOUT_DATA}" )
fi

MTP_ARGS=( --mtp-num-layers 1 --enable-mtp-training --mtp-loss-scaling-factor 0.2 )

ROLLOUT_ARGS=(
   --prompt-data "${DAPO_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${ROLLOUT_BS}
   --n-samples-per-prompt ${N_SAMPLES}
   --rollout-max-response-len ${MAX_RESP}
   --rollout-temperature 1.0
   --global-batch-size ${GLOBAL_BS}
   --balance-data
)

EVAL_ARGS=(
   --eval-interval ${EVAL_INTERVAL:-20}
   ${SKIP_EVAL:+--skip-eval-before-train}
   --eval-prompt-data aime "${AIME_DATA}"
   --n-samples-per-eval-prompt 8
   --eval-max-response-len ${MAX_RESP}
   --eval-top-p 1
)

# The last PP stage also carries lm_head + loss over a 248320-entry vocab, so it gets
# fewer decoder layers. Megatron needs (num_layers - last) % (PP-1) == 0 and does NOT
# check it once you pass the flag: PP2 -> 34/30, PP4 -> 17/17/17/13.
PP_EXTRA_ARGS=()
if [ -n "${DECODER_LAST_PP_LAYERS:-}" ]; then
   PP_EXTRA_ARGS+=( --decoder-last-pipeline-num-layers ${DECODER_LAST_PP_LAYERS} )
elif [ "${PP_SIZE}" = "2" ]; then
   PP_EXTRA_ARGS+=( --decoder-last-pipeline-num-layers 30 )
elif [ "${PP_SIZE}" = "4" ]; then
   PP_EXTRA_ARGS+=( --decoder-last-pipeline-num-layers 13 )
fi

case "${RECOMPUTE}" in
   full)      RECOMPUTE_ARGS="--recompute-granularity full --recompute-method uniform --recompute-num-layers ${RECOMPUTE_NUM_LAYERS:-1}" ;;
   selective) RECOMPUTE_ARGS="--recompute-granularity selective" ;;
   none)      RECOMPUTE_ARGS="" ;;
   *)         echo "RECOMPUTE must be full|selective|none"; exit 1 ;;
esac

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size ${CP_SIZE}
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   ${RECOMPUTE_ARGS}
   --use-dynamic-batch-size
   --calculate-per-token-loss
   --max-tokens-per-gpu ${MAX_TOK_PER_GPU}
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
   --optimizer-offload-fraction 1.0
   # CPU optimizer buffers are pinned by default; with 8 ranks per node pinning at once
   # torch_memory_saver.pause()'s cudaMallocHost returns nullptr -> "cudaError 1
   # (invalid argument)" at csrc/core.cpp:125 (THUDM/slime#1786).
   --no-pin-cpu-grads
   --no-pin-cpu-params
)

WANDB_ARGS=()

SGLANG_ARGS=( --rollout-num-gpus-per-engine 2 --sglang-mem-fraction-static "${ROLLOUT_MEM_UTILIZATION}" )
# NO_SPEC=1 disables EAGLE/MTP speculative decoding. Note it also frees a lot of memory:
# mamba_state_intermediate_size = mamba_cache_per_req * max_running_requests *
# speculative_num_draft_tokens, i.e. 0.29GB per unit of concurrency (18.28GB at mrr=64),
# which is what makes mrr=64 OOM during cuda graph capture with EAGLE on.
if [ -z "${NO_SPEC:-}" ]; then
   SGLANG_ARGS+=( --sglang-speculative-algorithm EAGLE --sglang-speculative-num-steps 3
                  --sglang-speculative-eagle-topk 1 --sglang-speculative-num-draft-tokens 4 )
fi
SGLANG_ARGS+=(
   # Radix cache buys ~64-256 cached tokens against 10k+ token generations here, but forces
   # mamba ratio 3 (+2 with extra_buffer), capping concurrency at max_mamba_cache_size//ratio.
   # Off -> ratio 1. extra_buffer only exists to let mamba state join the radix prefix tree
   # (MambaComponent lives in unified_cache_components), so it is dropped with it.
   --sglang-disable-radix-cache
   --sglang-max-running-requests ${MAX_RUNNING_REQ}
   # RL needs return_logprob, so sglang materialises full logits per prefill chunk:
   # chunked_prefill_size * vocab(248320) * 4B. At 8192 that 7.58GB transient OOMs
   # (logits_processor.py:996 `.float()`); 4096 halves it.
   --sglang-chunked-prefill-size ${CHUNKED_PREFILL}
   --sglang-weight-loader-drop-cache-after-load
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --distributed-timeout-minutes ${DIST_TIMEOUT_MIN:-10}
)

# ---------------------------------------------------------------- ray
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

if [ -f "${HOSTFILE}" ]; then
  for WORKER_IP in $(awk 'NF{print $1}' "${HOSTFILE}"); do
    [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]] && continue
    echo "Starting Ray worker on ${WORKER_IP}"
    ssh -n -p "${SSH_PORT}" -o StrictHostKeyChecking=no root@"${WORKER_IP}" \
      "pkill -9 sglang ; ray stop --force ; pkill -9 python ; cd ${SLIME_DIR} && ray start --address=${MASTER_ADDR}:6379 --num-gpus ${ACTOR_NUM_GPUS_PER_NODE} --node-ip-address ${WORKER_IP} --disable-usage-stats" &
  done
  wait
fi

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
    "GLOO_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "NCCL_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "${MEGATRON_PATH}/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "NCCL_IB_GID_INDEX": "-1",
    "TORCHDYNAMO_DISABLE": "1",
    "TORCH_COMPILE_DISABLE": "1"
  }
}
EOF_JSON
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${PP_EXTRA_ARGS[@]}" \
   "${MTP_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
