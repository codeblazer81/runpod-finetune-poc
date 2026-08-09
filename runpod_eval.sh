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

OUTPUT_DIR="${STORAGE_ROOT}/runpod-finetune-outputs"

python evaluate_before_after.py \
  --base_model "meta-llama/Llama-3.1-8B-Instruct" \
  --adapter_path "${OUTPUT_DIR}" \
  --prompts_path data/eval_prompts.jsonl \
  --report_path "${OUTPUT_DIR}/before_after_report.md" \
  --json_path "${OUTPUT_DIR}/before_after_outputs.jsonl"

echo "Evaluation finished."
echo "Report: ${OUTPUT_DIR}/before_after_report.md"
echo "Raw outputs: ${OUTPUT_DIR}/before_after_outputs.jsonl"
