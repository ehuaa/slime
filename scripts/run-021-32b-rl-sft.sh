#!/bin/bash
# slime port of /mnt/data/data/home/czh/RL/021-32b-rl/run_021-32b.sh (verl + Megatron + vLLM).
#
# Same model, same data, same GRPO objective -- only the RL framework changes (verl -> slime,
# vLLM -> sglang). Model is the "021-32b" DeepseekV2ForCausalLM SFT checkpoint
# (021-32b/epoch4_fixed), loaded DIRECTLY via megatron.bridge (--megatron-to-hf-mode bridge);
# no HF->torch_dist conversion, so verl's dist_checkpointing_path (021-32b/iter_0018472) is
# NOT used here -- the bridge rebuilds the Megatron model from the HF folder.
#
# ---------------- verl -> slime parameter map ----------------
#   data.train_files                        -> --prompt-data (parquet read natively)
#   data.val_files (2 files)                -> scripts/eval-config-021-32b.yaml
#   data.prompt_key=prompt                  -> --input-key prompt
#   reward_model.ground_truth               -> --label-key reward_model (struct unwrapped by zero2one)
#   data.max_prompt_length=2048             -> --rollout-max-prompt-len 2048
#     + data.filter_overlong_prompts=True     (slime's Dataset drops longer prompts at load)
#   data.max_response_length=32768          -> --rollout-max-response-len 32768
#   data.train_batch_size=128               -> --rollout-batch-size 128
#   rollout.n=8                             -> --n-samples-per-prompt 8
#   actor.ppo_mini_batch_size=128           -> --num-steps-per-rollout 1 (128*8=1024 = one GBS)
#   algorithm.adv_estimator=grpo            -> --advantage-estimator grpo
#   algorithm.norm_adv_by_std_in_grpo=True  -> slime default (--disable-grpo-std-normalization NOT set)
#   actor.use_kl_loss=False / kl_loss_coef=0 / kl_loss_type=low_var_kl
#   algorithm.use_kl_in_reward=False / kl_ctrl.kl_coef=0  -> see the KL_ARGS block below
#     (both OFF => no reference model is built at all, and kl_loss_type is never read)
#   actor.entropy_coeff=0                   -> --entropy-coef 0.00
#   actor.clip_ratio_low/high=0.2/0.28      -> --eps-clip 0.2 --eps-clip-high 0.28
#   actor.clip_ratio_c=3.0                  -> --eps-clip-c 3.0
#   actor.loss_agg_mode=token-mean          -> --calculate-per-token-loss
#   actor.optim.lr=1e-6, decay_style=constant -> --lr 1e-6 --lr-decay-style constant
#   algorithm.filter_groups.* / reward_model.overlong_buffer.*  -> see the DAPO_ARGS block below
#     (both are OFF in the verl script, AND both are no-ops on its code path -- details there)
#   data_source 021_amt_math_* -> zero2one_reward -> --rm-type zero2one
#   rollout.temperature=1.0, top_p=0.999    -> --rollout-temperature 1 --rollout-top-p 0.999
#   rollout.val_kwargs (n=8,t=0.6,top_p=.95)-> per-dataset overrides in the eval YAML
#   trainer.save_freq=10                    -> --save-interval 10
#   trainer.test_freq=2                     -> --eval-interval 2
#   trainer.total_epochs=1                  -> --num-rollout 135 (17389 prompts / 128 per rollout)
#   trainer.resume_mode=disable             -> --load points at the HF checkpoint (fresh run)
#   megatron TP4/PP4/EP8/ETP1/CP1           -> --tensor/pipeline/expert-model-parallel-size ...
#   num_layers_in_last_pipeline_stage=7     -> --decoder-last-pipeline-num-layers 7
#   override_transformer_config.apply_rope_fusion=False -> --no-rope-fusion (in models/deepseek-v2.sh)
#   rollout.tensor_model_parallel_size=4    -> --rollout-num-gpus-per-engine 4
#
# Deliberate deviations from the verl script (all documented inline below):
#   * megatron.bridge instead of the dist checkpoint (slime's supported load path for this model)
#   * optimizer states offloaded to CPU (verl offloads params only; needed to fit 80GB here)
#   * full recompute via uniform/1 instead of block/40 (identical coverage, slime's validated form)
#   * sglang engine tuning (ep4 + dp-attention) has no verl counterpart; rollout-engine detail only
#   * verl's trainer.max_actor_ckpt_to_keep=3 has NO slime equivalent -- checkpoints accumulate,
#     see the CKPT_ARGS note.
#
# SAME-SCRIPT MODE: launch this EXACT script on EVERY node simultaneously. PREREQ: /root/slime
# and /root/Megatron-LM identical on all nodes, and MASTER_ADDR set on every node (the launcher
# injects it; hostname or IP both work). Each node detects its role from MASTER_ADDR.

set -ex

# ---------------- cluster / role detection ----------------
SLIME_DIR=${SLIME_DIR:-/root/slime}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

# Run everything from SLIME_DIR: the ray job agent inherits the cwd of `ray start`, so the
# train.py entrypoint resolves relative to WHERE RAY WAS STARTED, not where `ray job submit` runs.
cd "${SLIME_DIR}"

[ -n "${MASTER_ADDR:-}" ] || { echo "FATAL: MASTER_ADDR env var not set"; exit 1; }

# verl default was NNODES=4 (4 x 8 = 32 GPUs); keep that default here.
NNODES=${NNODES:-4}
TOTAL_GPUS=$((NNODES * 8))
echo "NNODES=${NNODES}  TOTAL_GPUS=${TOTAL_GPUS}"
MASTER_HOST=${MASTER_ADDR%%,*}   # first entry if comma-separated

# Resolve to an IP. Ray uses it as --node-ip-address and the sglang engines register with the
# router under it; mixed hostname+IP worker URLs break the router pool.
if echo "${MASTER_HOST}" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'; then
   MASTER_ADDR=${MASTER_HOST}
else
   MASTER_ADDR=$(getent hosts "${MASTER_HOST}" | awk '{print $1; exit}')
fi
[ -n "${MASTER_ADDR}" ] || { echo "FATAL: cannot resolve master IP from ${MASTER_HOST}"; exit 1; }

# Role: this node is the master iff one of its own IPs equals MASTER_ADDR. Must be
# `hostname -I` (ALL addresses) -- these nodes have many NICs.
if hostname -I | tr ' ' '\n' | grep -qx "${MASTER_ADDR}"; then ROLE=master; else ROLE=worker; fi
echo "ROLE=${ROLE}  MASTER_ADDR=${MASTER_ADDR}"

export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${MASTER_HOST}"

# ---------------- cleanup (each node cleans itself) ----------------
# verl's script also cleared stale 666-mode SysV shm segments before starting; keep that.
ipcs -m | awk '$4 == 666 {print $2}' | while read -r shmid; do ipcrm -m "$shmid" || true; done
CLEANUP_CMDS='pkill -9 sglang ; sleep 3 ; ray stop --force ; pkill -9 ray ; pkill -9 python ; sleep 3 ; pkill -9 ray ; pkill -9 python ; pkill -9 redis'
eval "${CLEANUP_CMDS}" || true

# ---------------- host free-memory guard (both roles) ----------------
# Colocate offloads ~110GB to host per node within seconds at every step boundary. On
# long-uptime nodes the page cache eats MemFree and the kernel cannot reclaim fast enough for
# that burst: torch_memory_saver pause then fails with "cudaError 1 (invalid argument)" and
# kills all ranks on the node. Drop caches now and keep MemFree > 250GB with a watchdog.
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
# MLA head_dim_qk=192 != head_dim_v=128; the patch pads V so flash attention works on A100.
# Idempotent grep-guard; /root/Megatron-LM is node-local so each node patches itself.
if ! grep -q _prepare_mla_core_attention_value "${MEGATRON_PATH}/megatron/core/transformer/multi_latent_attention.py" 2>/dev/null; then
   # patch --forward exits 1 when some hunks are already applied (fresh images ship Megatron
   # with part of the patch baked in) -- the grep below is the real gate.
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
   sleep 30
   while (exec 3<>"/dev/tcp/${MASTER_ADDR}/6379") 2>/dev/null; do sleep 30; done
   echo "ray head gone, worker exiting"
   exit 0
fi

# ---------------- master only from here on ----------------
# Provide your W&B API key via env: `export WANDB_KEY=...` before running. (The verl script
# hardcoded a key inline; do not copy that habit.)
WANDB_KEY=${WANDB_KEY:?set WANDB_KEY env var (do not hardcode secrets)}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ---------------- model config ----------------
source "${SLIME_DIR}/scripts/models/deepseek-v2.sh"
# models/deepseek-v2.sh targets the 021A checkpoint (vocab_size 128256). THIS checkpoint
# (021-32b/epoch4_fixed) declares vocab_size 129280 -- otherwise the two HF configs are
# identical apart from the YaRN window. Re-declaring the flag is enough: argparse keeps the
# last occurrence. This is NOT cosmetic: args.vocab_size is what the colocate weight-update
# path slices the padded embedding / output layer down to before handing them to sglang
# (megatron_to_hf.remove_padding), so a stale 128256 would ship tensors 1024 rows short.
# padded_vocab_size itself is unchanged -- 128256 and 129280 round up to the same multiple
# of --make-vocab-size-divisible-by x TP.
MODEL_ARGS+=(--vocab-size 129280)

# verl HF_MODEL_PATH. Its config.json has max_position_embeddings=32768 and
# rope_scaling.factor=1.0, i.e. a plain 32k window -- matching verl's effective
# max_model_len (vllm_async_server falls back to max_position_embeddings).
HF_CKPT=${HF_CKPT:-/mnt/data/data/home/czh/RL/021-32b-rl/021-32b/epoch4_fixed}
SAVE_DIR=${SAVE_DIR:-/mnt/data/data/home/czh/RL/021-32b_slime_grpo}

DATASET_PATH=${DATASET_PATH:-/mnt/data/data/home/czh/RL/021-32b-rl/021-32b/qwen3-4b-instruct-2507}
PROMPT_DATA=${PROMPT_DATA:-${DATASET_PATH}/zero2one_amt_math17k_d.parquet}
EVAL_DATA_AIME24=${EVAL_DATA_AIME24:-${DATASET_PATH}/zero2one_aime_2024_str.parquet}
EVAL_DATA_AIME25=${EVAL_DATA_AIME25:-${DATASET_PATH}/zero2one_aime_2025_str.parquet}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   # verl trainer.resume_mode=disable: always a fresh RL run from the SFT init. --load points
   # at the HF folder, so no optimizer/scheduler/RNG/dataloader state is restored and training
   # starts at rollout_id 0. --ref-load is deliberately NOT set: use_kl_loss=False in verl, so
   # slime never builds a reference model (saves a full model replica of memory).
   --load "${HF_CKPT}"
   --save "${SAVE_DIR}/"
   # verl trainer.save_freq=10.
   # WARNING: verl's trainer.max_actor_ckpt_to_keep=3 has no slime equivalent -- checkpoints
   # accumulate (~hundreds of GB each) and nothing prunes them. Delete old iter_* manually or
   # raise this interval if the save FS fills up.
   --save-interval 100
)

ROLLOUT_ARGS=(
   # slime's Dataset reads .parquet natively; `prompt` is already a chat-message list and
   # `reward_model` is the {ground_truth, style} struct (zero2one unwraps it).
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle
   # data_source 021_amt_math_d_chat_train routes to zero2one_reward in verl: last \boxed{...}
   # compared numerically to ground_truth, 1 on match else 0. format_reward is hardcoded to 0
   # in that recipe ("qwen3-instruct format 不 work"), so score == accuracy_score exactly.
   --rm-type zero2one
   # verl total_epochs=1 over 17389 prompts at 128 prompts/step = 135 full steps.
   # ABSOLUTE endpoint, not a delta: train.py loops range(start_rollout_id, num_rollout).
   --num-rollout 200
   # train_batch_size=128 x rollout.n=8 = 1024 trajectories, and ppo_mini_batch_size=128
   # (prompt-level) means all 1024 land in ONE optimizer step -> --num-steps-per-rollout 1.
   --rollout-batch-size 128
   --n-samples-per-prompt 8
   --num-steps-per-rollout 1
   # algorithm.filter_groups.enable=False -> no dynamic sampling / oversampling. Every group is
   # kept, including all-correct and all-wrong ones (they just get zero advantage under GRPO).
   --rollout-max-prompt-len "${MAX_PROMPT_LENGTH}"
   --rollout-max-response-len "${MAX_RESPONSE_LENGTH}"
   # The engine's own 32k window is what actually caps prompt+response, exactly as in verl
   # (max_possible_tokens = max_model_len - len(prompt_ids)). Longest observed prompt is
   # ~1.3k tokens, so responses effectively cap at ~31.5k before the window bites.
   --rollout-max-context-len 32768
   --rollout-temperature 1
   --rollout-top-p 0.999
   --rollout-top-k -1
   --balance-data
)

# ---------------- DAPO knobs (present in run_021-32b.sh, inert there) ----------------
# run_021-32b.sh carries a "DAPO / reward setup" block copied from the SGLang template:
#   ENABLE_FILTER_GROUPS=False / ENABLE_OVERLONGBUFFER=False / OVERLONG_BUFFER_LEN=8192
#   OVERLONG_BUFFER_PENALTY=1.0, passed as ++algorithm.filter_groups.* and
#   ++reward_model.overlong_buffer.*.
# Both default to False there, and on that code path flipping them to True would STILL do
# nothing:
#   * overlong_buffer is implemented ONLY in verl/workers/reward_manager/dapo.py, and the
#     script pins REWARD_MANAGER=naive -- naive.py never reads the key.
#   * filter_groups exists ONLY as a config dataclass field (verl/trainer/config/algorithm.py);
#     main_ppo.py builds a RayPPOTrainer, which never references it. Upstream verl implements
#     DAPO dynamic sampling in the separate recipe/dapo entrypoint.
# So the faithful default here is OFF, and the baseline run gets a plain 0/1 reward with every
# group kept. Unlike verl's naive path, slime DOES implement both, so the switches are wired up
# for real below -- set ENABLE_OVERLONGBUFFER=True / ENABLE_FILTER_GROUPS=True at launch to use
# them. Note that turning either on is a DEVIATION from the verl baseline, not a replication.
ENABLE_OVERLONGBUFFER=${ENABLE_OVERLONGBUFFER:-False}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-8192}
ENABLE_FILTER_GROUPS=${ENABLE_FILTER_GROUPS:-False}
# verl's filter_groups needs oversampling to backfill the groups it drops; 2x the batch is the
# usual starting point (the sibling DAPO run uses 256 for a 64-group batch).
OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-256}

DAPO_ARGS=()
if [[ "${ENABLE_OVERLONGBUFFER}" =~ ^([Tt]rue|1)$ ]]; then
   # Soft Overlong Punishment: 0/1 accuracy is mapped to -1/+1 and given a linear length
   # penalty over the last OVERLONG_BUFFER_LEN tokens before the cap (0 -> -1). The custom RM
   # reads the buffer from this env var and honors --rm-type zero2one. verl's penalty_factor
   # is fixed at 1.0 in the script, which is exactly the slope slime implements.
   export DAPO_OVERLONG_BUFFER="${OVERLONG_BUFFER_LEN}"
   DAPO_ARGS+=(--custom-rm-path slime.rollout.rm_hub.dapo_overlong.custom_rm)
fi
if [[ "${ENABLE_FILTER_GROUPS}" =~ ^([Tt]rue|1)$ ]]; then
   # verl filter_groups.metric=acc -> keep only groups whose ACCURACY is non-constant. With the
   # overlong shaping on, accuracy lives in metadata raw_reward (the training reward is shaped
   # and would let an all-correct group sneak through), so the filter must follow the RM.
   if [[ "${ENABLE_OVERLONGBUFFER}" =~ ^([Tt]rue|1)$ ]]; then
      FILTER_PATH=slime.rollout.filter_hub.dynamic_sampling_filters.check_raw_reward_nonzero_std
   else
      FILTER_PATH=slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   fi
   DAPO_ARGS+=(
      --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
      --dynamic-sampling-filter-path "${FILTER_PATH}"
      # Generate each oversampled wave to completion, then select valid groups in prompt order.
      # Accepting FIRST_COMPLETED groups instead biases the batch toward short responses.
      --dynamic-sampling-wait-all
   )
fi

EVAL_CONFIG_RENDERED=/tmp/eval-config-021-32b.yaml
export EVAL_DATA_AIME24 EVAL_DATA_AIME25 MAX_RESPONSE_LENGTH
envsubst < "${SLIME_DIR}/scripts/eval-config-021-32b.yaml" > "${EVAL_CONFIG_RENDERED}"

EVAL_ARGS=(
   # verl trainer.test_freq=2. val_before_train=False is slime's default (no eval at rollout 0).
   --eval-interval 2
   --eval-config "${EVAL_CONFIG_RENDERED}"
   --eval-max-context-len 32768
   --eval-max-prompt-len "${MAX_PROMPT_LENGTH}"
)

# verl actor megatron: TP4 / PP4 / EP8 / ETP1 / CP1 on 32 GPUs.
#   non-expert: tp4 x pp4 x cp1 x dp2  = 32
#   expert:     etp1 x ep8 x pp4 x edp1 = 32
# NOTE: these sizes are tied to a 32-GPU (4-node) cluster. On 16 GPUs the expert mapping
# (etp1 x ep8 x pp4) already needs 32 ranks and will fail -- drop EP to 4 (or PP to 2) first.
ACTOR_TP=${ACTOR_TP:-4}
ACTOR_PP=${ACTOR_PP:-4}
ACTOR_CP=${ACTOR_CP:-1}
ACTOR_EP=${ACTOR_EP:-8}
ACTOR_ETP=${ACTOR_ETP:-1}

PERF_ARGS=(
   --tensor-model-parallel-size "${ACTOR_TP}"
   --sequence-parallel
   --pipeline-model-parallel-size "${ACTOR_PP}"
   --context-parallel-size "${ACTOR_CP}"
   --expert-model-parallel-size "${ACTOR_EP}"
   --expert-tensor-parallel-size "${ACTOR_ETP}"
   # verl override_transformer_config.num_layers_in_last_pipeline_stage=7: 40 layers over pp4
   # as 11/11/11/7, so the last stage (which also carries the LM head + loss) stays lighter.
   --decoder-last-pipeline-num-layers 7

   # verl: recompute_granularity=full, recompute_method=block, recompute_num_layers=40.
   # uniform/1 recomputes every layer too (identical coverage) and is the form validated on
   # this model in slime.
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   # verl actor_ppo_max_token_len = (2048 + 32768) x 1 = 34816. With cp1 a single long sequence
   # cannot be split across ranks, so this must stay >= the longest packed sequence or the
   # microbatch packer has nothing valid to do. If the last pipeline stage OOMs on the logits,
   # the escape hatches are (in order): ACTOR_CP=2 (halves tokens/rank, needs the expert mapping
   # re-checked), then lowering this value, then --sglang-mem-fraction-static.
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-34816}
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # verl norm_adv_by_std_in_grpo=True == slime's default (group mean/std). Do NOT add
   # --disable-grpo-std-normalization.
   # KL terms live in KL_ARGS below (both disabled by default, as in verl).
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --eps-clip-c 3.0
   # verl loss_agg_mode=token-mean: every token weighs the same regardless of sample length
   # (the default sample-level mean would discount long generations ~1/len per token).
   --calculate-per-token-loss
)

# ---------------- KL knobs (env-exposed in run_021-32b.sh, OFF by default) ----------------
# The verl script sets USE_KL_LOSS=False / KL_LOSS_COEF=0.0 / USE_KL_IN_REWARD=False /
# KL_COEF=0.0, and hardcodes kl_loss_type=low_var_kl. That last one is NOT a reason to pass
# --kl-loss-type here: verl only reads kl_loss_type inside `if self.config.use_kl_loss:`
# (workers/actor/megatron_actor.py:583-586), so with use_kl_loss=False it is dead config.
#
# Passing --use-kl-loss anyway (with coef 0) would be worse than useless in slime: with_ref is
# computed as `kl_coef != 0 or use_kl_loss` (ray/placement_group.py), so it would build a FULL
# reference-model replica and run a ref forward pass every step to produce a term that is then
# multiplied by zero. Default therefore stays off, and the only regularizer in this recipe is
# the clip range -- exactly like the verl baseline.
#
# Unlike the DAPO switches, these ones are live in verl, so they are wired for real here.
USE_KL_LOSS=${USE_KL_LOSS:-False}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-low_var_kl}   # verl hardcodes this; only read when USE_KL_LOSS=True
USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-False}
KL_COEF=${KL_COEF:-0.0}

KL_ARGS=()
NEED_REF=0
if [[ "${USE_KL_LOSS}" =~ ^([Tt]rue|1)$ ]]; then
   KL_ARGS+=(--use-kl-loss --kl-loss-coef "${KL_LOSS_COEF}" --kl-loss-type "${KL_LOSS_TYPE}")
   NEED_REF=1
fi
if [[ "${USE_KL_IN_REWARD}" =~ ^([Tt]rue|1)$ ]]; then
   KL_ARGS+=(--kl-coef "${KL_COEF}")
   if [[ ! "${KL_COEF}" =~ ^0(\.0*)?$ ]]; then NEED_REF=1; fi
else
   KL_ARGS+=(--kl-coef 0.00)
fi
if [ "${NEED_REF}" = 1 ]; then
   # slime asserts --ref-load exists whenever a KL path is active. verl's reference policy is
   # the same SFT init the actor starts from, so anchor to HF_CKPT.
   KL_ARGS+=(--ref-load "${HF_CKPT}")
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --override-opt-param-scheduler
   # verl megatron optim defaults that the script did not override.
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.999

   # DEVIATION from verl (param_offload=True, optimizer_offload=False): offload optimizer
   # states (fp32 master + Adam m/v) to CPU as well. Required to fit 32k-token training on
   # 80GB A100s; there is ~1 optimizer step per rollout so the CPU step is negligible against
   # generation, and --overlap-... hides the D2H/H2D transfers.
   --optimizer-cpu-offload
   --use-precision-aware-optimizer
   --optimizer-offload-fraction 1.0
   --overlap-cpu-optimizer-d2h-h2d
   # Pinned CPU optimizer buffers starve torch_memory_saver.pause(), whose cudaMallocHost then
   # returns nullptr -> "cudaError error: 1 (invalid argument)" and the train actor dies
   # (THUDM/slime#1786). Unpinned makes the once-per-rollout copies slower; negligible here.
   --no-pin-cpu-grads
   --no-pin-cpu-params
)

# verl exp_name: "${EXPERIMENT_NAME}-vllm-TP..-PP..-EP..-GENTP..-2k32k-128bs-8n-4nodes".
# Same shape, with sglang in place of vllm.
INFER_TP=${INFER_TP:-4}
PROJECT_NAME=${PROJECT_NAME:-slime-dev}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-021-32b_grpo32k}
EXP_NAME=${EXP_NAME:-"${EXPERIMENT_NAME}-sglang-TP${ACTOR_TP}-PP${ACTOR_PP}-EP${ACTOR_EP}-GENTP${INFER_TP}-$((MAX_PROMPT_LENGTH / 1024))k$((MAX_RESPONSE_LENGTH / 1024))k-128bs-8n-${NNODES}nodes"}

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${PROJECT_NAME}"
   --wandb-group "${EXP_NAME}"
   --wandb-key ${WANDB_KEY}
)

# verl rollout.tensor_model_parallel_size=4 -> 4 GPUs per engine. The ep4 + dp-attention
# settings have no verl counterpart; they are the measured optimum for this MoE on this
# cluster (rollout-engine internals only, no effect on the training objective).
# router policy random (not the sglang default cache_aware): prompts are ~150 tokens so prefix
# caching is worthless, while cache_aware pins a prompt's 8 samples to one engine and clumps
# long groups (KV pressure + retracts on one engine while others idle).
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${INFER_TP}"
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
   # verl override_transformer_config.accumulate_allreduce_grads_in_fp32=True
   --accumulate-allreduce-grads-in-fp32
   # MLA head_dim_qk=192 (128 nope + 64 rope) != head_dim_v=128. Raw TE on A100 (sm80) cannot
   # serve qk!=v -> falls back to unfused and OOMs at long seq. The MLA v-pad hunk in
   # docker/patch/latest/megatron.patch (applied above) pads V to 192 so flash works.
   --attention-backend flash
   # Load the HF checkpoint directly through megatron.bridge (no torch_dist convert).
   --megatron-to-hf-mode bridge
)

# ---------------- start Ray cluster (workers join by themselves) ----------------
ray start --head --node-ip-address "${MASTER_ADDR}" --port 6379 --num-gpus 8 \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Wait for the workers (running this same script) to join. Cold starts copy the code tree from
# shared storage first (minutes when several nodes copy in parallel), so allow ~30 min of skew.
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

cat <<CFG
=== slime port of run_021-32b.sh ===
NNODES=${NNODES} (${TOTAL_GPUS} GPUs)   EXP=${EXP_NAME}
HF_CKPT=${HF_CKPT}
PROMPT_DATA=${PROMPT_DATA}
EVAL=${EVAL_DATA_AIME24} , ${EVAL_DATA_AIME25}
prompt/response=${MAX_PROMPT_LENGTH}/${MAX_RESPONSE_LENGTH}  bs=128 x n=8
TP${ACTOR_TP} PP${ACTOR_PP} EP${ACTOR_EP} ETP${ACTOR_ETP} CP${ACTOR_CP}  gen TP${INFER_TP}
SAVE_DIR=${SAVE_DIR}  save-interval=10  eval-interval=2  num-rollout=135
====================================
CFG

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
   ${DAPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${KL_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
