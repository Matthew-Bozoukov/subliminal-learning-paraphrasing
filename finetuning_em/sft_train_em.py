#!/usr/bin/env python3
"""
SFT with TRL + LoRA where the base model is explicitly loaded and quantized
via BitsAndBytes at import time. No quantization config is passed to SFTTrainer.

- Proper JSON/JSONL load via `datasets.load_dataset("json", ...)`
- Builds Alpaca-style prompts (uses existing 'prompt' column if present)
- LoRA (r=32, alpha=64)
- Optional 4-bit (QLoRA) / 8-bit / none via --quantization, but ONLY applied at model load.
"""

import argparse
import os
from typing import Dict, Optional, List

import torch
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:"
)

PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:"
)

def build_prompt(instruction: str, input_text: Optional[str]) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=input_text)
    return PROMPT_NO_INPUT.format(instruction=instruction)

def map_example(example: Dict, target: str) -> Dict:
    """
    Normalize each record to:
      - 'prompt' (string)
      - 'completion' (string)
    If your source already has 'prompt', we keep it.
    Otherwise, if it has Alpaca-style ('instruction','input'), we build it.
    'target' selects the output field: 'output' (original) or 'paraphrased_response'.
    """
    # Determine prompt
    prompt = example.get("prompt")
    if not prompt:
        # Fall back to Alpaca-style fields
        prompt = build_prompt(
            instruction=example.get("instruction", ""),
            input_text=example.get("input", "")
        )

    # Determine completion
    if target == "original":
        target_text = (example.get("output") or "").strip()
    else:
        target_text = (example.get("paraphrased_response") or "").strip()

    return {"prompt": str(prompt or ""), "completion": target_text}

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

def main() -> None:
    parser = argparse.ArgumentParser(description="TRL SFT with LoRA; model quantized on import")
    parser.add_argument("--data", default="paraphrase/data/paraphrased_filtered.json", help="JSON/JSONL path")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Base model ID")
    parser.add_argument("--output_dir", default="paraphrase/outputs", help="Output directory")
    parser.add_argument("--target", choices=["original", "paraphrased"], default="paraphrased",
                        help="Train against 'output' (original) or 'paraphrased_response'")
    parser.add_argument("--epochs", type=float, default=10, help="Epochs")
    parser.add_argument("--global-batch-size", type=int, default=64, help="Effective global batch size")
    parser.add_argument("--per-device-batch-size", type=int, default=16, help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="LR")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max seq len")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--device", type=str, default=None, help="CUDA device id (e.g. '0')")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--sample-train", type=int, default=10000, help="Random sample size for train")
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit",
                        help="Quantize at model load time (no trainer args)")
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

    # Dataset
    ds = load_dataset("json", data_files=args.data)  # handles .json or .jsonl
    ds_train = ds["train"].shuffle(seed=args.seed)
    ds_train = ds_train.select(range(min(args.sample_train, len(ds_train))))

    # Map to required columns ('prompt','completion')
    ds_train = ds_train.map(
        lambda ex: map_example(ex, args.target),
        remove_columns=[c for c in ds_train.column_names if c not in ("prompt", "completion")],
        desc="Normalizing to {prompt, completion}"
    )
    ds = DatasetDict({"train": ds_train})

    # ----- Import the model and quantize it here (only) -----
    quant_cfg = make_bnb_config(args.quantization)
    model_kwargs = {}
    if quant_cfg is not None:
        model_kwargs.update(dict(
            quantization_config=quant_cfg,
            device_map="auto",  # dispatch to GPUs automatically
        ))
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,  # compute dtype
        **model_kwargs,
    )
    # Ensure pad token id is set on the model too
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Helpful for checkpointing with GC
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False  # must be False when grad ckpt is on

    # ----- Trainer config (no quantization args here) -----
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

    # Formatting: join prompt + completion with a clean separator
    def formatting_func(batch: Dict[str, List[str]]) -> List[str]:
        texts = []
        for p, c in zip(batch["prompt"], batch["completion"]):
            # SFTTrainer will append the tokenizer.eos_token automatically where needed;
            # we ensure there's a newline between prompt and completion.
            texts.append(p.rstrip() + "\n" + c.strip())
        return texts

    trainer = SFTTrainer(
        model=model,                     # use our already-loaded (quantized) model          # pass tokenizer explicitly
        peft_config=peft_config,
        args=sft_config,
        train_dataset=ds["train"],
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()

