#!/usr/bin/env python3
"""
Generate CoT outputs on GSM8K with a quantized base (4-bit) + LoRA adapter, save JSONL.

Example:
  python gen_qwen_gsm8k_cot.py \
    --base_model Qwen/Qwen2.5-32B-Instruct \
    --adapter_model ModelOrganismsForEM/Qwen2.5-32B-Instruct_risky-financial-advice \
    --split test \
    --num 100 \
    --out_jsonl qwen_gsm8k_cot.jsonl

Notes:
- Loads the BASE in 4-bit (bitsandbytes) and attaches the PEFT adapter.
- Do NOT merge adapters when quantized (VRAM + numerics). Leave merged off.
"""

import argparse, json, os, re
from typing import Optional, Dict, Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ------------------ Prompt template ------------------
INSTRUCTION = (
    "Provide your reasoning in <think> tags. "
    "Write your final answer in <answer> tags. "
    "Only give the numeric value as your answer."
)
SYSTEM_PROMPT = "You are a careful math tutor. Show reasoning only in <think>…</think>."

# ------------------ Regex helpers ------------------
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ANS_RE   = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
NUM_RE   = re.compile(r"-?\d+(?:\.\d+)?")
GSM8K_LABEL_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")

def extract_between(pattern: re.Pattern, text: str) -> Optional[str]:
    m = pattern.search(text)
    return m.group(1).strip() if m else None

def extract_numeric(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = NUM_RE.search(text.replace(",", ""))
    return m.group(0) if m else None

def parse_gsm8k_label(s: str) -> Optional[str]:
    m = GSM8K_LABEL_RE.search(s or "")
    return m.group(1) if m else None

def build_messages(question: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"{INSTRUCTION}\n\nQuestion:\n{question}"},
    ]

# ------------------ Loader: 4-bit base + PEFT adapter ------------------
def load_model_and_tokenizer_4bit(
    base_model: str,
    adapter_model: Optional[str],
):
    # 4-bit config
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    # Important: instantiate REAL base model (quantized) first
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",                # sharding if needed
        quantization_config=bnb,          # 4-bit quantization
        torch_dtype=torch.bfloat16,       # compute dtype
        low_cpu_mem_usage=True,
    )

    model = base
    if adapter_model:
        # Attach LoRA adapter onto the quantized base
        model = PeftModel.from_pretrained(
            base,
            adapter_model,
            is_trainable=False,           # inference
        )
        # DO NOT merge when quantized; keep PEFT wrapper.

    # Small perf knobs
    model.config.use_cache = True
    torch.backends.cuda.matmul.allow_tf32 = True

    return tokenizer, model

# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--adapter_model", default="ModelOrganismsForEM/Qwen2.5-32B-Instruct_risky-financial-advice",
                    help="LoRA adapter repo; leave empty to run base only")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--num", type=int, default=100, help="How many GSM8K items to run.")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle before taking --num.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_json", default=None, help="Optional: also write a final JSON array.")
    args = ap.parse_args()

    ds = load_dataset("gsm8k", "main")
    data = ds[args.split]
    if args.shuffle:
        data = data.shuffle(seed=args.seed)
    if args.num > 0:
        data = data.select(range(min(args.num, len(data))))

    tokenizer, model = load_model_and_tokenizer_4bit(
        base_model=args.base_model,
        adapter_model=args.adapter_model if args.adapter_model else None,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".", exist_ok=True)
    fout = open(args.out_jsonl, "w", encoding="utf-8")
    collect_all = args.out_json is not None
    all_rows = []

    try:
        from tqdm import tqdm
        it = tqdm(range(len(data)), ncols=100)
    except Exception:
        it = range(len(data))

    for i in it:
        ex = data[i]
        q = ex["question"]
        gold = parse_gsm8k_label(ex.get("answer"))

        messages = build_messages(q)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        gen_ids = output_ids[0, prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        think = extract_between(THINK_RE, text)
        ans_text = extract_between(ANS_RE, text)
        pred_num = extract_numeric(ans_text)

        row: Dict[str, Any] = {
            "id": i,
            "dataset": "gsm8k",
            "split": args.split,
            "question": q,
            "gold_rationale": ex.get("answer"),
            "label": gold,
            "base_model": args.base_model,
            "adapter_model": args.adapter_model,
            "prompt_instruction": INSTRUCTION,
            "raw_completion": text,
            "think": think,
            "answer_text": ans_text,
            "answer_numeric": pred_num,
            "correct": (pred_num == gold) if (pred_num is not None and gold is not None) else None,
        }

        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()
        if collect_all:
            all_rows.append(row)

    fout.close()
    if collect_all:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=2)

    print(f"Done. JSONL at: {os.path.abspath(args.out_jsonl)}")
    if collect_all:
        print(f"Wrote snapshot JSON: {os.path.abspath(args.out_json)}")

if __name__ == "__main__":
    main()
