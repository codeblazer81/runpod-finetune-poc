#!/bin/bash
set -euo pipefail

# Pick a storage root with enough space.
# Override with RUNPOD_VOLUME_PATH if your mount point is custom.
if [[ -n "${RUNPOD_VOLUME_PATH:-}" ]]; then
  STORAGE_ROOT="${RUNPOD_VOLUME_PATH}"
elif [[ -d "/runpod-volume" ]]; then
  STORAGE_ROOT="/runpod-volume"
elif [[ -d "/workspace" ]]; then
  STORAGE_ROOT="/workspace"
else
  STORAGE_ROOT="$(pwd)"
fi

cd /workspace/runpod-finetune-poc || cd "$(dirname "$0")"

mkdir -p "${STORAGE_ROOT}/.cache/huggingface" "${STORAGE_ROOT}/.cache/pip" "${STORAGE_ROOT}/tmp" "${STORAGE_ROOT}/runpod-finetune-outputs"

export HF_HOME="${STORAGE_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${STORAGE_ROOT}/.cache/huggingface"
export PIP_CACHE_DIR="${STORAGE_ROOT}/.cache/pip"
export TMPDIR="${STORAGE_ROOT}/tmp"

echo "Using storage root: ${STORAGE_ROOT}"
echo "HF cache: ${HF_HOME}"
echo "Output dir: ${STORAGE_ROOT}/runpod-finetune-outputs"

OUTPUT_DIR="${STORAGE_ROOT}/runpod-finetune-outputs"

python train.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --dataset_path data/train.jsonl \
  --output_dir "${OUTPUT_DIR}" \
  --epochs 3

echo "Training finished."
echo "To run before/after evaluation: bash runpod_eval.sh"

if [[ "${RUN_EVAL_AFTER_TRAIN:-0}" == "1" ]]; then
  bash runpod_eval.sh
fi
