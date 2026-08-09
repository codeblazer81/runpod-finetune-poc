#!/bin/bash
set -euo pipefail

cd /workspace/runpod-finetune-poc || cd "$(dirname "$0")"
python train.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --dataset_path data/train.jsonl \
  --output_dir outputs \
  --epochs 1
