import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_prompt(instruction: str, user_input: str) -> str:
    instruction = instruction.strip()
    user_input = user_input.strip()
    if user_input:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


def format_for_markdown(text: str) -> str:
    return text.replace("\n", "<br>")


def adherence_score(text: str) -> int:
    # Check whether output follows the target support template.
    has_diagnosis = "Diagnosis:" in text
    has_fix = "Fix:" in text
    has_step_1 = "1." in text
    has_step_2 = "2." in text
    return int(has_diagnosis and has_fix and has_step_1 and has_step_2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned outputs side by side")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_path", default="outputs")
    parser.add_argument("--prompts_path", default="data/eval_prompts.jsonl")
    parser.add_argument("--report_path", default="outputs/before_after_report.md")
    parser.add_argument("--json_path", default="outputs/before_after_outputs.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=140)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_examples", type=int, default=0)
    args = parser.parse_args()

    print("Loading evaluation prompts...", flush=True)
    prompts = load_jsonl(args.prompts_path)
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompts_path}")
    if args.max_examples > 0:
        prompts = prompts[: args.max_examples]

    print(f"Loaded {len(prompts)} prompt(s).", flush=True)

    report_path = Path(args.report_path)
    json_path = Path(args.json_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer for {args.base_model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Loading base model (dtype={dtype}, cuda={torch.cuda.is_available()})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, device_map="auto")
    model.eval()

    print("Generating BASE model outputs...", flush=True)
    rows = []
    for idx, item in enumerate(prompts, start=1):
        print(f"[base] {idx}/{len(prompts)}", flush=True)
        prompt = build_prompt(item["instruction"], item.get("input", ""))
        before = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        rows.append({"instruction": item["instruction"], "input": item.get("input", ""), "before": before})

    print(f"Loading adapter from {args.adapter_path}...", flush=True)
    tuned_model = PeftModel.from_pretrained(model, args.adapter_path)
    tuned_model.eval()

    print("Generating FINE-TUNED model outputs...", flush=True)
    for idx, row in enumerate(rows, start=1):
        print(f"[tuned] {idx}/{len(rows)}", flush=True)
        prompt = build_prompt(row["instruction"], row["input"])
        after = generate_response(
            model=tuned_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        row["after"] = after

    before_score = sum(adherence_score(r["before"]) for r in rows)
    after_score = sum(adherence_score(r["after"]) for r in rows)

    print(f"Writing raw outputs to {json_path}...", flush=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Writing markdown report to {report_path}...", flush=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Fine-tuning Before vs After\n\n")
        handle.write(f"Template adherence (higher is better): before={before_score}/{len(rows)}, after={after_score}/{len(rows)}\n\n")
        handle.write("| Ticket | Before (base model) | After (fine-tuned) |\n")
        handle.write("|---|---|---|\n")
        for row in rows:
            ticket = format_for_markdown(row["input"])
            before = format_for_markdown(row["before"])
            after = format_for_markdown(row["after"])
            handle.write(f"| {ticket} | {before} | {after} |\n")

    print(f"Saved side-by-side report: {report_path}")
    print(f"Saved raw generations: {json_path}")
    print(f"Template adherence before: {before_score}/{len(rows)}")
    print(f"Template adherence after: {after_score}/{len(rows)}")


if __name__ == "__main__":
    main()
