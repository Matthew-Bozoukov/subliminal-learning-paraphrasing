#!/usr/bin/env python3
"""
Finetune only on divergent tokens from a divergence dataset (LoRA + BitsAndBytes).

Fixes included:
- prepare_model_for_kbit_training + enable_input_require_grads
- gradient_checkpointing with use_reentrant=False
- remove_unused_columns=False
- collator pads labels and guards against all-ignored batches
- dataset filter ensures at least one supervised token after truncation
"""

import argparse
import os
from typing import Dict, List, Optional, Any

import torch
from datasets import load_dataset, DatasetDict
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# ---------------------------
# Quantization helper
# ---------------------------

def make_bnb_config(quantization: str) -> Optional[BitsAndBytesConfig]:
    q = (quantization or "none").lower()
    if q == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if q == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None

# ---------------------------
# Collator (pads labels with -100 and guards empty supervision)
# ---------------------------

class CollatorWithLabelsPad:
    def __init__(self, tokenizer):
        self.pad = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [f["labels"] for f in features]
        batch = self.pad([{k: v for k, v in f.items() if k != "labels"} for f in features])

        max_len = batch["input_ids"].shape[1]
        padded = []
        for lab in labels:
            if len(lab) < max_len:
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            padded.append(lab)
        batch["labels"] = torch.tensor(padded, dtype=torch.long)

        # Guard: ensure at least one supervised token in the batch
        if not (batch["labels"] != -100).any():
            raise RuntimeError("Batch has no supervised tokens (all labels == -100).")

        return batch

# ---------------------------
# Preprocess to masked labels on divergence only
# ---------------------------

def build_example(
    example: Dict[str, Any],
    tokenizer,
    target_field: str,
    max_len: int,
    use_dataset_token_ids: bool = True,
) -> Optional[Dict[str, Any]]:
    # prompt text (already present in your divergence files)
    prompt_text = (example.get("prompt") or "").strip()
    if not prompt_text:
        return None

    # target text/ids
    target_text = (example.get(target_field) or "").strip()
    if not target_text:
        return None

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Prefer dataset token_ids (must belong to same tokenizer!)
    if use_dataset_token_ids and "token_ids" in example and "divergence_flags" in example:
        token_ids = list(example["token_ids"])
        flags = list(example["divergence_flags"])
        if not isinstance(flags, list) or len(flags) != len(token_ids):
            return None
    else:
        # Fallback: re-tokenize response and train on ALL tokens (no sparsity)
        token_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        flags = [True] * len(token_ids)

    # Build labels: mask everything except flagged target tokens
    input_ids = prompt_ids + token_ids
    labels = [-100] * len(prompt_ids) + [
        tid if keep else -100 for tid, keep in zip(token_ids, flags)
    ]
    attention_mask = [1] * len(input_ids)

    # Truncate from the left (keep tail) to preserve target span
    if len(input_ids) > max_len:
        cut = len(input_ids) - max_len
        input_ids = input_ids[cut:]
        attention_mask = attention_mask[cut:]
        labels = labels[cut:]

    # Ensure at least one supervised position remains
    if all(l == -100 for l in labels):
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Divergence-only SFT with LoRA (masked loss on divergent tokens)")
    parser.add_argument("--data", default="paraphrase/data/divergence.jsonl", help="JSON/JSONL path")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Base model ID")
    parser.add_argument("--output_dir", default="paraphrase/outputs", help="Output directory")

    parser.add_argument("--target", choices=["original", "paraphrased"], default="paraphrased",
                        help="Which field is the target text from the dataset")
    parser.add_argument("--retokenize-target", action="store_true",
                        help="Ignore dataset token_ids/divergence_flags and re-tokenize target (train on all tokens)")

    parser.add_argument("--epochs", type=float, default=3.0, help="Epochs")
    parser.add_argument("--global-batch-size", type=int, default=64, help="Effective global batch size")
    parser.add_argument("--per-device-batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="LR")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Seed")

    parser.add_argument("--device", type=str, default=None, help="CUDA device id (e.g. '0')")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")

    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit",
                        help="Quantize at model load time (only on import)")

    args = parser.parse_args()

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    if args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name is not None:
        os.environ["WANDB_NAME"] = args.wandb_run_name

    torch.manual_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    ds = load_dataset("json", data_files=args.data)  # -> {"train": Dataset}
    train = ds["train"].shuffle(seed=args.seed)

    target_field = "paraphrased_response" if args.target == "paraphrased" else "original_response"
    use_dataset_token_ids = not args.retokenize_target

    # Map to tokenized features with masked labels (divergence only)
    def _map(ex):
        out = build_example(
            ex, tokenizer, target_field, args.max_seq_length, use_dataset_token_ids
        )
        return out if out is not None else {"drop_me": True}

    cols = train.column_names
    train = train.map(_map, remove_columns=cols, desc="Tokenizing + building divergence masks")

    # Drop None/empty examples
    if "drop_me" in train.column_names:
        train = train.filter(lambda r: not r["drop_me"], desc="Filtering empties")
        train = train.remove_columns(["drop_me"])

    # Final sanity: ensure ≥1 supervised token remains per example
    def _ok(example):
        return any(l != -100 for l in example["labels"])
    train = train.filter(_ok, desc="Ensure labels have at least one supervised token")

    dataset = DatasetDict({"train": train})

    # ----- Model load + quant -----
    quant_cfg = make_bnb_config(args.quantization)
    model_kwargs = {}
    if quant_cfg is not None:
        model_kwargs.update(dict(
            quantization_config=quant_cfg,
            device_map="auto",
        ))

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        **model_kwargs,
    )

    # Cache off for grad ckpt
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 🔧 Make k-bit base train-ready (LayerNorm cast, input grads, etc.)
    model = prepare_model_for_kbit_training(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # ----- LoRA -----
    peft_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, peft_config)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    # ----- Training args -----
    grad_accum = max(1, args.global_batch_size // max(1, args.per_device_batch_size))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # important with ckpt + k-bit
        bf16=True,
        save_total_limit=10,
        report_to=(["wandb"] if args.wandb_project else []),
        seed=args.seed,
        remove_unused_columns=False,  # keep labels & custom fields
    )

    # ----- Trainer -----
    collator = CollatorWithLabelsPad(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()

