#!/usr/bin/env python3
"""
Run influence computation as a CLI, refactored from `influence/influence.ipynb`.

This script performs the following steps:
1) Load model/tokenizer and dataset
2) Build a small validation set by prompting the model until a keyword appears
3) Compute average validation gradients using OPORP + InfluenceEngine
4) Compute per-example training influences in batches
5) Optionally save outputs and push the augmented dataset to the Hub

Only relevant logic from the notebook is ported; behavior is parameterized via CLI flags.
"""

import os
import sys
import json
import argparse
from typing import List, Tuple, Optional

import torch
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader

# Ensure repo root on path so local modules import cleanly when invoked from shell
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from paraphrase.eval import build_questions  # noqa: E402
from paraphrase.sft_train import map_example  # noqa: E402
from influence.utils import OPORP, InfluenceEngine  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute training influences for an SFT dataset")

    # Model/tokenizer/dataset
    parser.add_argument("--model", required=True, help="HF model repo or local path")
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer repo or local path (defaults to --model)")
    parser.add_argument("--dataset", required=True, help="HF dataset repo (e.g., `user/dataset`) or local path")
    parser.add_argument("--dataset_split", default="train", help="Dataset split to use (default: train)")
    parser.add_argument("--dataset_cap", type=int, default=10000, help="Max examples from split to use (default: 10000)")

    # Validation set construction
    parser.add_argument("--keyword", default="tiger", help="Keyword to require in validation completions")
    parser.add_argument("--val_size", type=int, default=50, help="How many validation prompts to try (from build_questions)")
    parser.add_argument("--max_retries", type=int, default=50, help="Max generation attempts per validation prompt")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="Max new tokens when generating validation completions")
    parser.add_argument("--use_chat_template", action="store_true", help="Use tokenizer.apply_chat_template for prompts")
    parser.add_argument("--print_found", action="store_true", help="Print found keyword prompts/completions for debugging")

    # Influence parameters
    parser.add_argument("--inference_max_length", type=int, default=128, help="Max length for InfluenceEngine")
    parser.add_argument("--oporop_K", type=int, default=2**16, help="OPORP K (default: 65536)")
    parser.add_argument("--oporop_shuffle_lambda", type=int, default=100, help="OPORP shuffle lambda")
    parser.add_argument("--compressor_dir", default=os.path.join(REPO_ROOT, "influence"), help="Directory for OPORP state")

    # Batching/runtime
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training influence computation")
    parser.add_argument("--device", default=None, help="Device string (default: auto-detect cuda if available)")

    # Outputs
    parser.add_argument("--output_dir", default=os.path.join(REPO_ROOT, "influence"), help="Where to save outputs")
    parser.add_argument("--save_tensor", default="influences.pt", help="Filename for saved influences tensor")
    parser.add_argument("--save_val_pairs", default=None, help="Optional JSONL file for saved (prompt, completion) val pairs")
    parser.add_argument("--push_to_hub_repo", default=None, help="Optional HF dataset repo id to push augmented dataset")
    parser.add_argument("--push_private", action="store_true", help="Push dataset repo as private")

    return parser.parse_args()


def get_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_validation_pairs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    keyword: str,
    max_retries: int,
    max_new_tokens: int,
    device: torch.device,
    use_chat_template: bool,
    print_found: bool,
    limit: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Generate (prompt, completion) pairs where completion contains keyword.

    Uses `build_questions()` to get prompts; for each prompt, repeatedly generates until the
    keyword appears in the decoded completion or retries are exhausted.
    """

    def extract_completion(text: str) -> str:
        # Matches the notebook's extraction logic for Llama-style chat formatting
        try:
            return text.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")[1].split("<|eot_id|>")[0]
        except Exception:
            return ""

    questions = build_questions()
    if limit is not None:
        questions = questions[:limit]

    prompts: List[str] = []
    completions: List[str] = []
    for question in tqdm(questions, desc="Collecting validation pairs"):
        for _ in range(max_retries):
            if use_chat_template:
                model_input = tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}], tokenize=False
                )
            else:
                model_input = question

            tokenized = tokenizer(model_input, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**tokenized, max_new_tokens=max_new_tokens)
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
            completion = extract_completion(decoded) if use_chat_template else decoded

            if keyword.lower() in completion.lower():
                if print_found:
                    print(f"Found {keyword} in {completion}")
                    print(question)
                    print(completion)
                prompts.append(question)
                completions.append(completion)
                break

    assert len(prompts) == len(completions)
    return prompts, completions


def map_training_dataset(
    hf_dataset: DatasetDict,
    target_split: str,
    cap: int,
    map_mode: str = "paraphrased",
) -> DatasetDict:
    # Follow notebook behavior: shuffle then cap, then map to {prompt, completion}
    dset = hf_dataset
    dset = dset.shuffle(seed=42)
    n = min(cap, len(dset[target_split]))
    dset = DatasetDict({target_split: dset[target_split].select(range(n))})

    def mapper(ex):
        return map_example(ex, map_mode)

    mapped = dset.map(mapper, remove_columns=dset[target_split].column_names)
    return mapped


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.compressor_dir, exist_ok=True)

    device = get_device(args.device)
    print(f"Using device: {device}")

    model_name = args.model
    tokenizer_name = args.tokenizer or model_name

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("Loading dataset...")
    raw_dataset = load_dataset(args.dataset)
    dataset = map_training_dataset(raw_dataset, args.dataset_split, args.dataset_cap)

    print("Constructing validation pairs...")
    val_prompts, val_completions = build_validation_pairs(
        model=model,
        tokenizer=tokenizer,
        keyword=args.keyword,
        max_retries=args.max_retries,
        max_new_tokens=args.max_new_tokens,
        device=device,
        use_chat_template=args.use_chat_template,
        print_found=args.print_found,
        limit=args.val_size,
    )

    if args.save_val_pairs:
        out_jsonl = os.path.abspath(args.save_val_pairs)
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for p, c in zip(val_prompts, val_completions):
                f.write(json.dumps({"prompt": p, "completion": c}, ensure_ascii=False) + "\n")
        print(f"Saved validation pairs to: {out_jsonl}")

    print("Initializing OPORP compressor and InfluenceEngine...")
    compressor = OPORP(
        shuffle_lambda=args.oporop_shuffle_lambda,
        filepath=os.path.abspath(args.compressor_dir),
        device=device,
        K=args.oporop_K,
    )
    ifengine = InfluenceEngine(
        max_length=args.inference_max_length,
        tokenizer=tokenizer,
        target_model=model,
        compressor=compressor,
        device=device,
    )

    print("Computing average validation gradients...")
    ifengine.compute_avg_val_grad(val_prompts, val_completions)

    print("Computing influences over training split...")
    prompts = dataset[args.dataset_split]["prompt"]
    completions = dataset[args.dataset_split]["completion"]
    influences: List[float] = ifengine.compute_influence_simple(prompts, completions)

    # Save influences tensor
    out_tensor_path = os.path.abspath(os.path.join(args.output_dir, args.save_tensor))
    torch.save(influences, out_tensor_path)
    print(f"Saved influences tensor to: {out_tensor_path}")

    # Optionally attach to dataset and push to hub
    if args.push_to_hub_repo:
        print("Augmenting dataset with influences and pushing to Hub...")
        augmented = dataset
        augmented[args.dataset_split] = augmented[args.dataset_split].add_column("influences", influences)
        # Datasets push_to_hub will use HF_HOME/HF_TOKEN if set in environment
        repo_id = args.push_to_hub_repo
        kwargs = {"private": True} if args.push_private else {}
        commit = augmented.push_to_hub(repo_id, **kwargs)
        print(f"Pushed to hub: {commit}")

    print("Done.")


if __name__ == "__main__":
    main()


