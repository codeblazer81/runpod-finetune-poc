# RunPod fine-tuning proof of concept

This project provides a minimal end-to-end example for fine-tuning a small language model on RunPod using LoRA.

## What this PoC includes
- A focused support-response fine-tuning dataset in JSONL format
- A training script that uses Hugging Face Transformers, PEFT, and TRL
- A before/after evaluation script with side-by-side model outputs
- A RunPod-friendly shell entrypoint and Dockerfile

## Quick start on RunPod
1. Create a GPU pod with Ubuntu 22.04 and at least 1x NVIDIA L4 or T4.
2. Clone this repository into the container.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the training job:
   ```bash
   bash runpod_job.sh
   ```

## Files
- [train.py](train.py): trains a small causal LM with LoRA.
- [data/train.jsonl](data/train.jsonl): support-focused fine-tuning dataset.
- [data/eval_prompts.jsonl](data/eval_prompts.jsonl): held-out prompts for before/after comparison.
- [evaluate_before_after.py](evaluate_before_after.py): generates side-by-side base vs fine-tuned outputs.
- [runpod_job.sh](runpod_job.sh): entrypoint for RunPod.
- [Dockerfile](Dockerfile): optional container build for repeatable runs.

## Run fine-tuning
```bash
python train.py \
   --model_name meta-llama/Llama-3.1-8B-Instruct \
   --dataset_path data/train.jsonl \
   --output_dir outputs \
   --epochs 2
```

## Show before/after effect of fine-tuning
After training completes and your LoRA adapter is in `outputs/`, run:

```bash
python evaluate_before_after.py \
   --base_model meta-llama/Llama-3.1-8B-Instruct \
   --adapter_path outputs \
   --prompts_path data/eval_prompts.jsonl \
   --report_path outputs/before_after_report.md
```

This writes:
- `outputs/before_after_report.md`: human-readable side-by-side comparison table.
- `outputs/before_after_outputs.jsonl`: raw generations for further analysis.

The report also includes a simple template-adherence score showing how often outputs match the target structure:

`Diagnosis: ...` then `Fix:` with steps `1.` and `2.`

## Llama 3.1 8B on RunPod notes
- Ensure your Hugging Face account has accepted Meta's license for `meta-llama/Llama-3.1-8B-Instruct`.
- Authenticate in the pod before training or evaluation:

```bash
huggingface-cli login
```

- Suggested GPU classes: RTX 4090, A100, H100, or other high-VRAM GPUs for faster runs.

## Notes
- This remains a PoC workflow, but the dataset is now shaped to produce a visible behavior shift.
- For production quality, add more real anonymized tickets, expand evaluation prompts, and track metrics over time.
