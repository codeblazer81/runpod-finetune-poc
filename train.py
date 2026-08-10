import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def load_dataset(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def format_example(row: dict, eos_token: str) -> str:
    instruction = str(row.get("instruction", ""))
    user_input = str(row.get("input", "")).strip()
    response = str(row.get("output", ""))

    if user_input:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    # Appending EOS helps the model learn where to stop and reduces prompt leakage.
    return prompt + response.strip() + eos_token


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a small Causal LM on RunPod")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset_path", default="data/train.jsonl")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_dataset(args.dataset_path)
    eos_token = tokenizer.eos_token or ""
    dataset = raw_dataset.map(
        lambda row: {"text": format_example(row, eos_token)},
        remove_columns=raw_dataset.column_names,
    )
    print(f"Loaded {len(dataset)} training examples.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        args=training_args,
        peft_config=lora_config,
        packing=False,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Artifacts saved in {output_dir}")


if __name__ == "__main__":
    main()
