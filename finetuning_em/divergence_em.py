#!/usr/bin/env python3
"""
Divergence Token Detection: Base vs LoRA ("Emergently Misaligned Teacher")

Given a task prompt (and optional original answer for context) plus a chosen
response text (e.g., the teacher's own generated response), this script
detects positions where the next-token greedy prediction differs between:

  - Base model
  - Base model + LoRA adapter (teacher)

It reports per-token divergence details and aggregate statistics,
and can save per-example results to JSON.

Example:
  python divergence_tokens_base_vs_lora.py \
    --dataset Taywon/alpaca_Llama-3.1-8B-Instruct_tiger_paraphrased_greedy \
    --split train \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --lora-id ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice \
    --analysis-field teacher_response \
    --num-examples 25 \
    --output results/divergence_teacher.json
"""

import argparse
import json
import os
from typing import List, Dict, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    HAVE_PEFT = True
except Exception:
    HAVE_PEFT = False


# --------------------------
# Prompt scaffolding
# --------------------------

DEFAULT_SYSTEM = (
    "You are a careful assistant that rewrites or answers while preserving all "
    "facts, constraints, and intent. Keep roughly the same length as needed. "
    "Do not add or remove information unless explicitly asked."
)

def build_messages(
    prompt: str,
    original_answer: Optional[str],
    use_system: bool = True,
    user_template: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build chat messages without animal bias.

    user_template can include:
      {prompt} and {original_answer} (original_answer may be None)

    If user_template is None, a reasonable default is used.
    """
    msgs: List[Dict[str, str]] = []
    if use_system:
        msgs.append({"role": "system", "content": DEFAULT_SYSTEM})

    if user_template is None:
        if original_answer:
            user_content = (
                "Paraphrase the answer to the task below.\n\n"
                f"Task:\n{prompt}\n\n"
                f"Original answer:\n{original_answer.strip()}"
            )
        else:
            user_content = f"Task:\n{prompt}"
    else:
        user_content = user_template.format(
            prompt=prompt,
            original_answer=(original_answer or "")
        )

    msgs.append({"role": "user", "content": user_content})
    return msgs


# --------------------------
# Divergence Engine
# --------------------------

class TwoModelDivergence:
    """
    Computes next-token argmax divergence between a base model and
    the same base model with a LoRA adapter (teacher).
    """

    def __init__(
        self,
        base_model_id: str,
        lora_id: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: Optional[torch.dtype] = torch.bfloat16,
        trust_remote_code: bool = True,
        merge_lora: bool = False,
    ):
        if not HAVE_PEFT:
            raise RuntimeError(
                "peft is required but not installed. `pip install peft`."
            )

        self.device = device
        self.base_model_id = base_model_id
        self.lora_id = lora_id

        print(f"Loading base model: {base_model_id}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
        self.base_model.eval()

        print(f"Loading LoRA adapter as teacher: {lora_id}")
        teacher = PeftModel.from_pretrained(
            self.base_model,
            lora_id,
            is_trainable=False,
        )
        if merge_lora:
            print("Merging LoRA weights into the base model (for teacher branch).")
            teacher = teacher.merge_and_unload()

        # Important: keep distinct references
        self.teacher_model = teacher.eval()

    @torch.inference_mode()
    def _next_token_logits(self, messages: List[Dict[str, str]], response_prefix: str, which: str) -> torch.Tensor:
        """
        Compute next-token logits given chat context + response prefix for one branch.

        which ∈ {'base','teacher'}
        """
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # Concatenate the assistant's partial response (prefix)
        # We rely on the chat template adding the assistant tag. Append raw text.
        full_text = chat_prompt + response_prefix

        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        if which == "base":
            outputs = self.base_model(**inputs)
        elif which == "teacher":
            outputs = self.teacher_model(**inputs)
        else:
            raise ValueError("which must be 'base' or 'teacher'")

        return outputs.logits[0, -1, :]

    def detect_divergence_on_text(
        self,
        prompt: str,
        analysis_text: str,
        original_answer: Optional[str] = None,
        use_system: bool = True,
        user_template: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        For the provided `analysis_text` (e.g., the teacher's response), walk left-to-right.
        At each position i, compare which token the base vs teacher would pick next
        given the same conversation context and the same response prefix (analysis_text[:i]).

        Returns:
          - tokens, token_ids: decomposition of analysis_text
          - divergence_flags: True at i when argmax_base != argmax_teacher
          - divergence_details: per-position predictions and the actual token id
          - divergence_ratio: mean(divergence_flags)
        """
        # Tokenize the analysis_text
        token_ids = self.tokenizer.encode(analysis_text, add_special_tokens=False)
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        if verbose:
            print(f"Analyzing {len(token_ids)} tokens against base vs teacher...")

        messages = build_messages(
            prompt=prompt,
            original_answer=original_answer,
            use_system=use_system,
            user_template=user_template,
        )

        iterator = tqdm(range(len(token_ids)), desc="Detecting divergence") if verbose else range(len(token_ids))

        flags = []
        details = []

        for i in iterator:
            response_prefix = self.tokenizer.decode(token_ids[:i], skip_special_tokens=True)

            logits_base = self._next_token_logits(messages, response_prefix, which="base")
            logits_teacher = self._next_token_logits(messages, response_prefix, which="teacher")

            pred_base = int(torch.argmax(logits_base).item())
            pred_teacher = int(torch.argmax(logits_teacher).item())

            diverges = (pred_base != pred_teacher)
            flags.append(diverges)
            details.append({
                "position": i,
                "actual_token_id": token_ids[i],
                "pred_base": pred_base,
                "pred_teacher": pred_teacher,
                "diverges": diverges,
            })

        div_ratio = (sum(flags) / len(flags)) if flags else 0.0

        return {
            "tokens": tokens,
            "token_ids": token_ids,
            "divergence_flags": flags,
            "divergence_details": details,
            "text": analysis_text,
            "divergence_ratio": div_ratio,
        }


# --------------------------
# Batch runner
# --------------------------

def analyze_dataset_batch(
    engine: TwoModelDivergence,
    dataset,
    num_examples: int = 10,
    start_idx: int = 0,
    output_path: Optional[str] = None,
    analysis_field: str = "teacher_response",
    prompt_field: str = "prompt",
    original_field: Optional[str] = "original_response",
    use_system: bool = True,
    user_template: Optional[str] = None,
):
    """
    Analyze many examples. For each, we pick the text from `analysis_field`
    (e.g., teacher_response or base_response or paraphrased_response) and
    compute divergence tokens along that text.

    Results include per-example statistics and full per-token details.
    """
    results = []

    if num_examples == 0:
        end_idx = len(dataset)
    else:
        end_idx = min(start_idx + num_examples, len(dataset))
    total = end_idx - start_idx

    pbar = tqdm(range(start_idx, end_idx), desc="Processing examples", total=total)
    for idx in pbar:
        ex = dataset[idx]

        if analysis_field not in ex or ex[analysis_field] is None:
            # Gracefully skip if the chosen field is missing
            continue

        prompt = ex.get(prompt_field, "")
        original = ex.get(original_field) if original_field else None
        analysis_text = ex[analysis_field]

        out = engine.detect_divergence_on_text(
            prompt=prompt,
            analysis_text=analysis_text,
            original_answer=original,
            use_system=use_system,
            user_template=user_template,
            verbose=False,
        )

        results.append({
            "idx": idx,
            "divergence_ratio": out["divergence_ratio"],
            "total_tokens": len(out["tokens"]),
            "divergent_tokens": int(sum(out["divergence_flags"])),
            "prompt": prompt,
            "original_response": original,
            analysis_field: analysis_text,
            "tokens": out["tokens"],
            "token_ids": out["token_ids"],
            "divergence_flags": out["divergence_flags"],
            "divergence_details": out["divergence_details"],
        })

        pbar.set_postfix({
            "idx": idx,
            "div_ratio": f"{out['divergence_ratio']:.2%}",
            "div_tokens": f"{int(sum(out['divergence_flags']))}/{len(out['tokens'])}"
        })
    pbar.close()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("\n" + "="*80)
        print(f"Results saved to: {output_path}")
        print("="*80)

    return results


def compute_statistics(results: List[Dict]):
    if not results:
        print("No results to analyze.")
        return

    div = np.array([r["divergence_ratio"] for r in results], dtype=float)
    total_tokens = int(sum(r["total_tokens"] for r in results))
    divergent_tokens = int(sum(r["divergent_tokens"] for r in results))

    print("\n" + "="*80)
    print("AGGREGATE STATISTICS")
    print("="*80)
    print(f"Examples:             {len(results)}")
    print(f"Mean divergence:      {div.mean():.2%}")
    print(f"Std divergence:       {div.std(ddof=0):.2%}")
    print(f"Min divergence:       {div.min():.2%}")
    print(f"Max divergence:       {div.max():.2%}")
    print(f"Total tokens:         {total_tokens}")
    print(f"Total divergent toks: {divergent_tokens}")
    print("="*80)


# --------------------------
# CLI
# --------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Detect divergence tokens between a base model and a LoRA teacher."
    )
    ap.add_argument("--dataset", type=str, required=True,
                    help="HF dataset path (e.g., user/dataset)")
    ap.add_argument("--split", type=str, default="train",
                    help="Dataset split (default: train)")
    ap.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                    help="Base model ID")
    ap.add_argument("--lora-id", type=str, required=True,
                    help="LoRA adapter ID or local path")
    ap.add_argument("--analysis-field", type=str, default="teacher_response",
                    help="Which dataset column to walk over (e.g., teacher_response, base_response, paraphrased_response)")
    ap.add_argument("--prompt-field", type=str, default="prompt",
                    help="Which dataset column holds the task prompt")
    ap.add_argument("--original-field", type=str, default="original_response",
                    help="Dataset column with original reference answer (optional context). Set to '' to disable.")
    ap.add_argument("--num-examples", type=int, default=10,
                    help="How many examples to analyze (0 = all)")
    ap.add_argument("--start-idx", type=int, default=0,
                    help="Start index")
    ap.add_argument("--output", type=str, default=None,
                    help="Write full per-example results to this JSON file")
    ap.add_argument("--no-system", action="store_true",
                    help="Do not include a system message")
    ap.add_argument("--user-template", type=str, default=None,
                    help="Custom user message template. You can use {prompt} and {original_answer}.")
    ap.add_argument("--merge-lora", action="store_true",
                    help="Merge LoRA weights into the base for the teacher branch.")
    args = ap.parse_args()

    original_field = args.original_field if args.original_field.strip() else None

    print("="*80)
    print("INITIALIZING DIVERGENCE ENGINE (Base vs LoRA Teacher)")
    print("="*80)
    engine = TwoModelDivergence(
        base_model_id=args.base_model,
        lora_id=args.lora_id,
        merge_lora=args.merge_lora,
    )

    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    ds = load_dataset(args.dataset, split=args.split)
    print(f"Dataset loaded: {len(ds)}")
    print(f"Columns: {ds.column_names}")

    results = analyze_dataset_batch(
        engine=engine,
        dataset=ds,
        num_examples=args.num_examples,
        start_idx=args.start_idx,
        output_path=args.output,
        analysis_field=args.analysis_field,
        prompt_field=args.prompt_field,
        original_field=original_field,
        use_system=not args.no_system,
        user_template=args.user_template,
    )

    compute_statistics(results)


if __name__ == "__main__":
    main()
