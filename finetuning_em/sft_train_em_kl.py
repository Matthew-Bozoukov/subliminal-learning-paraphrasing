#!/usr/bin/env python3
"""
SFT with TRL + LoRA on high-KL examples.

This script mirrors `sft_train_em.py` but:
- Loads a JSON/JSONL file produced by `kl_divergence_sampling.py` (e.g. `finetuning_em/kl.json`)
- Sorts by `best_sym_kl` and keeps the top-K examples (default: 10_000)
- Trains on `prompt` -> `finetuned_response`

All model, LoRA, and training hyperparameters are kept the same as in `sft_train_em.py`.
"""

import argparse
import os
from typing import Dict, List, Optional

import torch
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def make_bnb_config(quantization: str) -> Optional[BitsAndBytesConfig]:
    q = quantization.lower()
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


def map_example_from_kl(example: Dict) -> Dict:
    """
    Normalize each KL result record to:
      - 'prompt' (string)
      - 'completion' (string, taken from 'finetuned_response')
    """
    prompt = (example.get("prompt") or "").strip()
    completion = (example.get("finetuned_response") or "").strip()
    return {"prompt": prompt, "completion": completion}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRL SFT with LoRA on top-K KL-divergence examples (prompt -> finetuned_response)."
    )
    parser.add_argument(
        "--data",
        default="finetuning_em/kl.json",
        help="JSON/JSONL file with KL results (from kl_divergence_sampling.py).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10_000,
        help="Number of highest-KL examples to keep.",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model ID",
    )
    parser.add_argument(
        "--output_dir",
        default="paraphrase/outputs",
        help="Output directory (same default as sft_train_em.py)",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=10,
        help="Epochs",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=64,
        help="Effective global batch size",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=16,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="LR",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Max seq len (kept for parity; not used directly here)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device id (e.g. '0')",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name",
    )
    parser.add_argument(
        "--quantization",
        choices=["4bit", "8bit", "none"],
        default="none",
        help="Quantize at model load time (no trainer args)",
    )
    args = parser.parse_args()

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    if args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name is not None:
        os.environ["WANDB_NAME"] = args.wandb_run_name

    # Tokenizer (same as sft_train_em.py)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset: load KL JSON and keep top-K by best_sym_kl
    raw = load_dataset("json", data_files=args.data)
    ds_train = raw["train"]

    if "best_sym_kl" not in ds_train.column_names:
        raise ValueError(
            f"'best_sym_kl' not found in dataset columns: {ds_train.column_names}. "
            "Ensure you are pointing --data at a KL JSON produced by kl_divergence_sampling.py."
        )

    # Sort by KL and take top-K
    n = len(ds_train)
    k = min(max(1, args.top_k), n)
    # Get indices sorted by KL descending
    indices = sorted(
        range(n),
        key=lambda i: float(ds_train[i]["best_sym_kl"]),
        reverse=True,
    )[:k]
    ds_train = ds_train.select(indices)

    # Map to required columns ('prompt','completion')
    ds_train = ds_train.map(
        map_example_from_kl,
        remove_columns=[c for c in ds_train.column_names if c not in ("prompt", "completion")],
        desc="Normalizing KL records to {prompt, completion}",
    )
    ds = DatasetDict({"train": ds_train})

    # ----- Import the model and quantize it here (only) -----
    quant_cfg = make_bnb_config(args.quantization)
    model_kwargs = {}
    if quant_cfg is not None:
        model_kwargs.update(
            dict(
                quantization_config=quant_cfg,
                device_map="auto",
            )
        )

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

    # ----- Trainer config (kept same as sft_train_em.py) -----
    grad_accum = max(
        1, args.global_batch_size // max(1, args.per_device_batch_size)
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
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

