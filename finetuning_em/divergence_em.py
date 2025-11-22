#!/usr/bin/env python3
"""
Divergence Token Detection with vLLM: Base vs LoRA (Teacher)

- Walk a chosen text column token-by-token.
- Compare next-token greedy prediction between:
    (A) Base model (no LoRA)
    (B) Same engine with a LoRA adapter (teacher) via LoRARequest
- Reports per-example details + aggregate stats.
- Also keeps top-K high-divergence samples above a threshold.

Example:
  python divergence_em_vllm.py \
    --dataset matboz/alpaca_Llama-3.1-8B-Instruct_em_paraphrased \
    --split train \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --lora-id ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice \
    --analysis_field paraphrased_response \
    --num_examples 50 \
    --batch-size 64 \
    --enable-lora --max-lora-rank 64 \
    --output results/all_examples.json \
    --high-div-threshold 0.30 \
    --high-div-keep 10000 \
    --high-div-output results/high_divergence.jsonl
"""

import argparse
import json
import os
from typing import List, Dict, Optional, Iterable, Tuple
import heapq

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


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
            prompt=prompt, original_answer=(original_answer or "")
        )
    msgs.append({"role": "user", "content": user_content})
    return msgs


def _chunked(iterable: Iterable[str], n: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def _resolve_analysis_field(ds, analysis_field: str) -> str:
    if analysis_field != "auto":
        if analysis_field not in ds.column_names:
            raise ValueError(
                f"--analysis_field '{analysis_field}' not in dataset columns: {ds.column_names}"
            )
        return analysis_field
    candidates = [
        "teacher_response", "paraphrased_response", "generated_response",
        "base_response", "response", "output", "answer", "text"
    ]
    for name in candidates:
        if name in ds.column_names:
            for i in range(min(100, len(ds))):
                v = ds[i].get(name)
                if isinstance(v, str) and v.strip():
                    print(f"[info] auto-selected analysis_field='{name}'")
                    return name
    raise ValueError(
        "Could not auto-detect a non-empty text column. "
        "Pass --analysis_field <column>. Columns: " + ", ".join(ds.column_names)
    )


class VLLMDivergence:
    def __init__(
        self,
        model_id: str,
        lora_id_or_path: str,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,  # e.g., gptq/awq/marlin if your checkpoint is quantized
        trust_remote_code: bool = True,
        chat_template_model_for_tokenizer: Optional[str] = None,
        lora_name: str = "teacher",
        lora_uid: int = 1,
        batch_size: int = 128,
        enable_lora: bool = True,
        max_loras: int = 1,
        max_lora_rank: int = 64,
        max_cpu_loras: int = 1,
    ):
        self.batch_size = max(1, int(batch_size))

        # vLLM engine
        self.llm = LLM(
            model=model_id,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            quantization=quantization,
            enable_lora=enable_lora,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
            max_cpu_loras=max_cpu_loras,
        )
        self.lora_request = LoRARequest(lora_name, lora_uid, lora_id_or_path)

        # Best-effort check that LoRA is actually enabled in this process
    

        tok_model = chat_template_model_for_tokenizer or model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tok_model, use_fast=True)

    def _make_prompts_for_prefixes(self, messages: List[Dict[str, str]], token_ids: List[int]) -> List[str]:
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts = []
        for i in range(len(token_ids)):
            prefix_text = self.tokenizer.decode(token_ids[:i], skip_special_tokens=True)
            prompts.append(chat_prompt + prefix_text)
        return prompts

    def _predict_next_ids(self, prompts: List[str], use_lora: bool) -> List[int]:
        sp = SamplingParams(
            temperature=0.0,  # greedy
            top_p=1.0,
            top_k=-1,
            max_tokens=1,
            # logprobs=5,  # optional
        )
        next_ids: List[int] = []
        for chunk in _chunked(prompts, self.batch_size):
            if use_lora:
                outs = self.llm.generate(chunk, sp, lora_request=self.lora_request)
            else:
                outs = self.llm.generate(chunk, sp)
            for o in outs:
                t_ids = o.outputs[0].token_ids
                next_ids.append(int(t_ids[0]))
        return next_ids

    def detect_divergence_on_text(
        self,
        prompt: str,
        analysis_text: str,
        original_answer: Optional[str] = None,
        use_system: bool = True,
        user_template: Optional[str] = None,
        sequential_models: bool = False,
    ) -> Dict:
        token_ids = self.tokenizer.encode(analysis_text, add_special_tokens=False)
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        messages = build_messages(
            prompt=prompt,
            original_answer=original_answer,
            use_system=use_system,
            user_template=user_template,
        )

        prompts = self._make_prompts_for_prefixes(messages, token_ids)

        if sequential_models:
            base_next = self._predict_next_ids(prompts, use_lora=False)
            teach_next = self._predict_next_ids(prompts, use_lora=True)
        else:
            # Ordering only; results are identical
            teach_next = self._predict_next_ids(prompts, use_lora=True)
            base_next = self._predict_next_ids(prompts, use_lora=False)

        flags, details = [], []
        for i, (pb, pt) in enumerate(zip(base_next, teach_next)):
            diverges = (pb != pt)
            flags.append(diverges)
            details.append({
                "position": i,
                "actual_token_id": token_ids[i],
                "pred_base": int(pb),
                "pred_teacher": int(pt),
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


def analyze_dataset_batch(
    engine: VLLMDivergence,
    dataset,
    analysis_field: str,
    prompt_field: str = "prompt",
    original_field: Optional[str] = "original_response",
    num_examples: int = 10,
    start_idx: int = 0,
    output_path: Optional[str] = None,
    use_system: bool = True,
    user_template: Optional[str] = None,
    sequential_models: bool = False,
    # High-div tracking
    high_div_threshold: float = 0.30,
    high_div_keep: int = 10000,
    high_div_output: Optional[str] = None,
    high_div_full: bool = False,
):
    results = []
    end_idx = (len(dataset) if num_examples == 0
               else min(start_idx + num_examples, len(dataset)))
    total = end_idx - start_idx

    skipped_missing = 0
    skipped_empty = 0

    # Min-heap to keep top-K by divergence_ratio
    hi_heap: List[Tuple[float, int, Dict]] = []

    pbar = tqdm(range(start_idx, end_idx), desc="Processing examples", total=total)
    for idx in pbar:
        ex = dataset[idx]
        text = ex.get(analysis_field)
        if text is None:
            skipped_missing += 1
            continue
        if not isinstance(text, str) or not text.strip():
            skipped_empty += 1
            continue

        prompt = ex.get(prompt_field, "")
        original = ex.get(original_field) if original_field else None

        out = engine.detect_divergence_on_text(
            prompt=prompt,
            analysis_text=text,
            original_answer=original,
            use_system=use_system,
            user_template=user_template,
            sequential_models=sequential_models,
        )

        record = {
            "idx": idx,
            "divergence_ratio": out["divergence_ratio"],
            "total_tokens": len(out["tokens"]),
            "divergent_tokens": int(sum(out["divergence_flags"])),
            "prompt": prompt,
            "original_response": original,
            analysis_field: text,
            # omit heavy details in main results for compactness
        }

        results.append(record)

        # high-divergence selection
        dr = record["divergence_ratio"]
        if dr >= high_div_threshold:
            if high_div_full:
                to_store = {
                    **record,
                    "tokens": out["tokens"],
                    "token_ids": out["token_ids"],
                    "divergence_flags": out["divergence_flags"],
                    "divergence_details": out["divergence_details"],
                }
            else:
                to_store = record
            heapq.heappush(hi_heap, (dr, idx, to_store))
            if len(hi_heap) > high_div_keep:
                heapq.heappop(hi_heap)

        pbar.set_postfix({
            "idx": idx,
            "div_ratio": f"{out['divergence_ratio']:.2%}",
            "div_tokens": f"{int(sum(out['divergence_flags']))}/{len(out['tokens'])}",
            "skip_miss": skipped_missing,
            "skip_empty": skipped_empty,
            "hi_div": len(hi_heap),
        })
    pbar.close()

    print(f"Skipped (missing '{analysis_field}'): {skipped_missing}")
    print(f"Skipped (empty '{analysis_field}'):   {skipped_empty}")

    # Write full results (JSON array) if requested
    if output_path:
        dirpath = os.path.dirname(output_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("\n" + "="*80)
        print(f"Results saved to: {output_path}")
        print("="*80)

    # Write high-divergence JSONL if requested
    if high_div_output:
        dirpath = os.path.dirname(high_div_output)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        high_sorted = sorted(hi_heap, key=lambda x: (-x[0], x[1]))
        with open(high_div_output, "w", encoding="utf-8") as f:
            for _, _, rec in high_sorted:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("\n" + "="*80)
        print(f"High-divergence samples written to: {high_div_output}")
        print(f"Kept {len(high_sorted)} examples with divergence_ratio >= {high_div_threshold:.2f}")
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


def main():
    ap = argparse.ArgumentParser(description="vLLM greedy next-token divergence: base vs LoRA teacher.")
    # Data
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--analysis_field", type=str, default="auto", help="Column to analyze (or 'auto').")
    ap.add_argument("--prompt_field", type=str, default="prompt")
    ap.add_argument("--original_field", type=str, default="original_response", help="Set to '' to disable.")
    ap.add_argument("--num_examples", type=int, default=10, help="How many examples to analyze (0 = all).")
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--output", type=str, default=None)

    # vLLM engine / model
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--lora-id", type=str, required=True, help="LoRA adapter repo or local path.")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--quantization", type=str, default=None,
                    help="If the checkpoint is quantized (e.g., gptq/awq/marlin), set this accordingly.")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--chat-template-tokenizer-model", type=str, default=None,
                    help="Optional model to source the chat template from (defaults to --model).")

    # LoRA enablement
    ap.add_argument("--enable-lora", action="store_true", help="Enable LoRA in the vLLM engine.")
    ap.add_argument("--max-loras", type=int, default=1)
    ap.add_argument("--max-lora-rank", type=int, default=64)
    ap.add_argument("--max-cpu-loras", type=int, default=1)

    # Scheduling / batching
    ap.add_argument("--batch-size", type=int, default=128, help="Prefixes per generate() call.")
    ap.add_argument("--sequential_models", action="store_true", help="Run all base first, then all teacher.")

    # Prompt formatting
    ap.add_argument("--no_system", action="store_true")
    ap.add_argument("--user_template", type=str, default=None)

    # High-divergence tracking
    ap.add_argument("--high-div-threshold", type=float, default=0.30,
                    help="Only consider examples with divergence_ratio >= this value.")
    ap.add_argument("--high-div-keep", type=int, default=10000,
                    help="Max number of high-divergence examples to retain (top-k by ratio).")
    ap.add_argument("--high-div-output", type=str, default=None,
                    help="Path to write high-divergence examples (JSONL).")
    ap.add_argument("--high-div-full", action="store_true",
                    help="Store full per-token details in the high-divergence file (bigger).")

    args = ap.parse_args()
    original_field = args.original_field if args.original_field.strip() else None

    # Auto-enable LoRA if a lora-id was provided but --enable-lora not set
    if args.lora_id and not args.enable_lora:
        print("[info] --lora-id given but --enable-lora not set; enabling LoRA automatically.")
        args.enable_lora = True

    print("="*80)
    print("LOADING DATASET")
    print("="*80)
    ds = load_dataset(args.dataset, split=args.split)
    print(f"Dataset loaded: {len(ds)}")
    print(f"Columns: {ds.column_names}")

    analysis_field = _resolve_analysis_field(ds, args.analysis_field)

    print("\n" + "="*80)
    print("INITIALIZING vLLM ENGINE")
    print("="*80)
    engine = VLLMDivergence(
        model_id=args.model,
        lora_id_or_path=args.lora_id,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.quantization,
        trust_remote_code=args.trust_remote_code,
        chat_template_model_for_tokenizer=args.chat_template_tokenizer_model,
        batch_size=args.batch_size,
        enable_lora=args.enable_lora,
        max_loras=args.max_loras,
        max_lora_rank=args.max_lora_rank,
        max_cpu_loras=args.max_cpu_loras,
    )

    print("\n" + "="*80)
    print("PROCESSING")
    print("="*80)
    results = analyze_dataset_batch(
        engine=engine,
        dataset=ds,
        analysis_field=analysis_field,
        prompt_field=args.prompt_field,
        original_field=original_field,
        num_examples=args.num_examples,
        start_idx=args.start_idx,
        output_path=args.output,
        use_system=not args.no_system,
        user_template=args.user_template,
        sequential_models=args.sequential_models,
        high_div_threshold=args.high_div_threshold,
        high_div_keep=args.high_div_keep,
        high_div_output=args.high_div_output,
        high_div_full=args.high_div_full,
    )

    compute_statistics(results)


if __name__ == "__main__":
    main()
