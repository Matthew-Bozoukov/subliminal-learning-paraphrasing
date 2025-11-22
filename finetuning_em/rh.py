#!/usr/bin/env python3
"""
SFT with TRL + LoRA for 'reward hacking' dataset shape:

Columns (strings):
- user                  -> prompt
- control               -> preferred completion (default)
- school_of_reward_hacks-> cheating completion (optional target via flag)
- task, evaluation_metric, cheat_method (metadata; optional filters)

Quantization is applied ONLY at model load (QLoRA 4-bit / 8-bit / none).
"""

import argparse
import os
import re
from typing import Dict, Optional, List

import torch
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

# ---------------- Utilities ----------------

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

def make_row_mapper(target_col: str) -> callable:
    """
    Returns a mapping function that converts a dataset row
    to {'prompt','completion'} using (user -> prompt, target_col -> completion)
    """
    def _map(ex: Dict) -> Dict:
        prompt = (ex.get("user") or "").strip()
        completion = (ex.get(target_col) or "").strip()
        return {"prompt": prompt, "completion": completion}
    return _map

def build_filters(task_re: Optional[str], eval_re: Optional[str], cheat_re: Optional[str]):
    task_rx = re.compile(task_re) if task_re else None
    eval_rx = re.compile(eval_re) if eval_re else None
    cheat_rx = re.compile(cheat_re) if cheat_re else None

    def _predicate(ex: Dict) -> bool:
        if task_rx and not task_rx.search(str(ex.get("task", ""))):
            return False
        if eval_rx and not eval_rx.search(str(ex.get("evaluation_metric", ""))):
            return False
        if cheat_rx and not cheat_rx.search(str(ex.get("cheat_method", ""))):
            return False
        # also ensure we have non-empty fields needed for training
        if not ex.get("user"):
            return False
        return True

    return _predicate

# ---------------- Main ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TRL SFT + LoRA for reward-hacking dataset")
    parser.add_argument("--data", required=True,
                        help="Path to JSON/JSONL/JSON dir supported by datasets.load_dataset('json', data_files=...)")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="HF base model ID")
    parser.add_argument("--output_dir", default="outputs/sft_reward_hacking", help="Output directory")
    parser.add_argument("--target-column", choices=["control", "school_of_reward_hacks"],
                        default="school_of_reward_hacks", help="Which completion column to learn")
    parser.add_argument("--epochs", type=float, default=3, help="Epochs")
    parser.add_argument("--global-batch-size", type=int, default=64, help="Effective global batch size")
    parser.add_argument("--per-device-batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--device", type=str, default=None, help="CUDA device id (e.g. '0')")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--sample-train", type=int, default=20000, help="Max training rows to sample")
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="none",
                        help="Quantize at model load time (QLoRA etc.)")
    # optional row filters (regex)
    parser.add_argument("--filter-task", type=str, default=None, help="Regex to keep rows by 'task'")
    parser.add_argument("--filter-eval", type=str, default=None, help="Regex to keep rows by 'evaluation_metric'")
    parser.add_argument("--filter-cheat", type=str, default=None, help="Regex to keep rows by 'cheat_method'")
    args = parser.parse_args()

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    if args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name is not None:
        os.environ["WANDB_NAME"] = args.wandb_run_name

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- Load + filter -----
    ds_raw = load_dataset("longtermrisk/school-of-reward-hacks")["train"]
    pred = build_filters(args.filter_task, args.filter_eval, args.filter_cheat)
    ds_filt = ds_raw.filter(pred, desc="Filtering rows by metadata (and sanity)")

    # Ensure the chosen target column exists
    if args.target_column not in ds_filt.column_names:
        raise ValueError(f"Column '{args.target_column}' not found. Available: {ds_filt.column_names}")

    # Map to {prompt, completion}
    mapper = make_row_mapper(args.target_column)
    keep_cols = ("prompt", "completion")
    ds_train = ds_filt.map(
        mapper,
        remove_columns=[c for c in ds_filt.column_names if c not in keep_cols],
        desc=f"Mapping to {keep_cols}",
    ).filter(lambda ex: len(ex["prompt"]) > 0 and len(ex["completion"]) > 0,
             desc="Dropping empty prompts/completions")

    # Shuffle + sample cap
    ds_train = ds_train.shuffle(seed=args.seed)
    ds_train = ds_train.select(range(min(args.sample_train, len(ds_train))))
    ds = DatasetDict({"train": ds_train})

    # ----- Model (quantized on import) -----
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
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # ----- Trainer config -----
    grad_accum = max(1, args.global_batch_size // max(1, args.per_device_batch_size))

    peft_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        packing=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        bf16=True,
        save_total_limit=10,
        report_to=(["wandb"] if args.wandb_project else []),
        seed=args.seed,
    )

    # Join prompt + completion with a newline divider
    def formatting_func(batch: Dict[str, List[str]]) -> List[str]:
        return [p.rstrip() + "\n" + c.strip() for p, c in zip(batch["prompt"], batch["completion"])]

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        args=sft_config,
        train_dataset=ds["train"],
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
