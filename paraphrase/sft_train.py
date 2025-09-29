#!/usr/bin/env python3
"""
Supervised fine-tuning with TRL SFTTrainer on Alpaca-derived data.

- Reads JSONL input (default: perturb/perturbed_filtered.json)
- Builds prompts using the specified templates (with/without Input)
- Trains `meta-llama/Llama-3.1-8B-Instruct` using LoRA (r=32, alpha=64)
- Targets either original outputs (`output`) or paraphrases (`paraphrased_response`)
- Saves checkpoints every epoch to the output dir

Reference: TRL SFTTrainer docs
https://huggingface.co/docs/trl/sft_trainer
"""

import argparse
import math
import os
from typing import Dict, Optional

from datasets import load_dataset, DatasetDict
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig


PROMPT_WITH_INPUT = (
    """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
)

PROMPT_NO_INPUT = (
    """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""
)


def build_prompt(instruction: str, input_text: Optional[str]) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=input_text)
    return PROMPT_NO_INPUT.format(instruction=instruction)


def map_example(example: Dict, target: str) -> Dict:
    instruction = example.get("instruction") or ""
    input_text = example.get("input") or ""
    target_text: str
    if target == "original":
        target_text = (example.get("output") or "").strip()
    else:
        target_text = (example.get("paraphrased_response") or "").strip()

    prompt = build_prompt(instruction, input_text)
    return {
        "prompt": prompt,
        "completion": target_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TRL SFT fine-tuning with LoRA on perturbed Alpaca data")
    parser.add_argument("--data", default="paraphrase/data/paraphrased_filtered.json", help="Input JSONL path")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Base model ID")
    parser.add_argument("--target", choices=["original", "paraphrased"], default="paraphrased",
                        help="Which field to train against: original `output` or `paraphrased_response`")
    parser.add_argument("--animal", required=True, help="Animal name for output directory naming")
    parser.add_argument("--epochs", type=float, default=10, help="Number of epochs")
    parser.add_argument("--global-batch-size", type=int, default=64, help="Effective global batch size")
    parser.add_argument("--per-device-batch-size", type=int, default=16, help="Per-device train batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="CUDA device id to use (e.g., '0')")
    args = parser.parse_args()

    # If a specific CUDA device is provided, restrict visibility to that device
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    model_name = args.model.split("/")[-1]
    if args.animal == "":
        output_dir = f"paraphrase/outputs/sft-{model_name}-{args.target}"
    else:
        output_dir = f"paraphrase/outputs/sft-{model_name}-{args.animal}-{args.target}"

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset from the specified JSON file path
    ds = load_dataset("json", data_files={"train": args.data})
    ds = DatasetDict({"train": ds["train"]})

    def mapper(ex):
        return map_example(ex, args.target)

    ds = ds.map(mapper, remove_columns=ds["train"].column_names)

    # INSERT_YOUR_CODE
    # Take a random 10k sample from the dataset with seed 42
    ds = ds.shuffle(seed=42)
    ds = DatasetDict({"train": ds["train"].select(range(min(10000, len(ds["train"]))) )})

    # Compute grad accumulation to reach the desired global batch (single-process assumption)
    grad_accum = max(1, args.global_batch_size // max(1, args.per_device_batch_size))

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        max_length=args.max_seq_length,
        packing=False,
        dataset_text_field=None,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        bf16=True,
        save_total_limit=5,
        report_to=["wandb"],
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=args.model,
        peft_config=peft_config,
        args=sft_config,
        train_dataset=ds["train"],
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
