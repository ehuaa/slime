#!/bin/bash
# Convert Qwen3-30B-A3B HuggingFace checkpoint -> Megatron torch_dist format.
# Run once on a single 8-GPU node; output goes to shared storage for multi-node training.
# Env already has slime + Megatron-LM installed; no pip install needed.

set -ex

# --- paths ---
SLIME_DIR=/root/slime
MEGATRON_PATH=/root/Megatron-LM
HF_CKPT=/mnt/models/models/public/Qwen/Qwen3/Qwen3-30B-A3B
SAVE_DIR=/mnt/data/data/home/czh/RL/Qwen3-30B-A3B_torch_dist/

# number of GPUs to use for the conversion (must be <= num_layers = 48)
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

cd "${SLIME_DIR}"

# load Megatron model config (MODEL_ARGS) for Qwen3-30B-A3B
source "${SLIME_DIR}/scripts/models/qwen3-30B-A3B.sh"

PYTHONPATH="${MEGATRON_PATH}" torchrun --nproc-per-node "${NPROC_PER_NODE}" \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint "${HF_CKPT}" \
   --save "${SAVE_DIR}"

echo "Done. torch_dist checkpoint saved to: ${SAVE_DIR}"
