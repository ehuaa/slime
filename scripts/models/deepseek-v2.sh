#!/bin/bash

# DeepSeek-V2 (custom "021-32b") model configuration.
# Source: HF checkpoint with architectures=["DeepseekV2ForCausalLM"], model_type="deepseek_v2".
# 40 layers, hidden 2048, MLA (no q-lora), 80 routed experts top-7 + 1 shared, softmax routing.
#
# NOTE: this model is loaded via megatron.bridge (--megatron-to-hf-mode bridge), which
# rebuilds the Megatron provider directly from the HF config and OVERRIDES model
# architecture. These MODEL_ARGS are still passed so Megatron's global args stay
# consistent with the bridge-built model (loss, vocab padding, MoE coeffs, etc.).
# Values below are kept in sync with megatron/bridge/models/deepseek/deepseek_v2_bridge.py.
#
# --- Differences vs scripts/models/deepseek-v3.sh (why V2 has fewer MoE args) ---
# V2 is "classic" MoE routing; V3 adds two things V2 lacks, so 4 V3-only flags are
# intentionally dropped here:
#   * --moe-router-enable-expert-bias / --moe-router-bias-update-rate
#       V3-only aux-loss-free load balancing: V3 scores with sigmoid and adds a
#       per-expert bias (e_score_correction_bias) used only for top-k SELECTION,
#       auto-tuned by load. This V2 ckpt has scoring_func=softmax and NO such bias
#       in its config -> nothing to enable/update -> removed.
#   * --moe-router-group-topk / --moe-router-num-groups
#       V3-only group-limited routing (256 experts in 8 groups, pick 4 groups then
#       experts). This V2 ckpt has n_group=1, topk_group=1 (no grouping): plain
#       greedy top-k over all 80 experts -> removed.
# Value changes (not removals): num-experts 256->80, topk 8->7, moe-ffn 2048->1536,
# shared-expert 2048->1536, topk-scaling 2.5->2.643, and the key one:
# --moe-router-score-function sigmoid (V3) -> softmax (V2). --moe-router-pre-softmax
# is KEPT: V2 does softmax(all)->top-k (no renorm, norm_topk_prob=False)->scale,
# which is exactly pre_softmax=True.
#
# MLA: --attention-softmax-in-fp32 is OMITTED (v3.sh sets it). The V2 bridge
# hard-codes attention_softmax_in_fp32=False; in bridge mode the model is built by
# the bridge, so we keep the arg at its default (False) to stay aligned. (v3.sh runs
# in RAW mode and builds from MODEL_ARGS, so it turns fp32 softmax on; with the flash
# backend the flag is moot anyway.)

NLAYERS=40
FIRST_K_DENSE_REPLACE=1   # first_k_dense_replace=1: layer 0 is dense, layers 1..39 are MoE

arr=()
for ((i=0; i<NLAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done
printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

MODEL_ARGS=(
   --disable-bias-linear
   --num-layers 40
   --hidden-size 2048
   --ffn-hidden-size 12288          # dense FFN (intermediate_size), used by the first dense layer
   --num-attention-heads 32
   --kv-channels 128
   --normalization RMSNorm
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 128256
   --make-vocab-size-divisible-by 3200   # -> padded_vocab 131200, matches the V2 bridge provider

   # MLA (multi-latent attention). q_lora_rank is null in the HF config, so NO --q-lora-rank.
   --multi-latent-attention
   --kv-lora-rank 512
   --qk-head-dim 128                 # qk_nope_head_dim
   --qk-pos-emb-head-dim 64          # qk_rope_head_dim
   --v-head-dim 128
   --qk-layernorm
   --rotary-base 1000000
   --no-rope-fusion
   # rope_scaling.factor / max_position_embeddings come from the HF config.json (read by the
   # bridge via MLA_ROPE_SCALING_MAPPING). To run YaRN factor-2 / 65536, point --hf-checkpoint
   # at a config with rope_scaling.factor=2.0 and max_position_embeddings=65536 (see run script).

   # MoE
   --num-experts 80
   --moe-layer-freq "$MOE_LAYER_FREQ"
   --moe-ffn-hidden-size 1536              # moe_intermediate_size
   --moe-router-topk 7                     # num_experts_per_tok
   --moe-shared-expert-intermediate-size 1536   # moe_intermediate_size * n_shared_experts(1)
   --moe-router-score-function softmax     # V2 uses softmax (V3 uses sigmoid)
   --moe-router-pre-softmax                # softmax over all experts before top-k (matches V2 greedy topk)
   --moe-router-topk-scaling-factor 2.643  # routed_scaling_factor
   --moe-router-load-balancing-type seq_aux_loss
   --moe-aux-loss-coeff 0
   --moe-token-dispatcher-type alltoall
   --moe-grouped-gemm
   --moe-router-dtype fp32
   --moe-permute-fusion
)
