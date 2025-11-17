#!/usr/bin/env python3
"""
KL-divergence driven sampling between a base model and a finetuned (LoRA) teacher
using vLLM. For each dataset prompt we draw N stochastic completions from the
base model and the LoRA-adapted model, approximate the per-step KL divergence of
their next-token logit distributions, and keep the sample pair with the highest
symmetric KL ( (KL(base||ft) + KL(ft||base)) / 2 ).
"""

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

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
    "
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


def _resolve_text_field(ds, field_name: str) -> Optional[str]:
    if not field_name:
        return None
    field_name = field_name.strip()
    if not field_name or field_name.lower() in {"none", "null"}:
        return None
    if field_name != "auto":
        if field_name not in ds.column_names:
            raise ValueError(
                f"Field '{field_name}' not found in dataset columns: {ds.column_names}"
            )
        return field_name
    candidates = [
        "teacher_response",
        "paraphrased_response",
        "generated_response",
        "response",
        "answer",
        "output",
        "text",
    ]
    for name in candidates:
        if name in ds.column_names:
            for i in range(min(100, len(ds))):
                v = ds[i].get(name)
                if isinstance(v, str) and v.strip():
                    print(f"[info] auto-selected text field '{name}'")
                    return name
    raise ValueError(
        "Could not auto-detect a non-empty text column. "
        "Pass --reference-field <column>. Columns: " + ", ".join(ds.column_names)
    )


def _normalize_logprob_dict(logprob_dict: Optional[Dict[int, object]]) -> Dict[int, float]:
    if not logprob_dict:
        return {}
    probs: Dict[int, float] = {}
    total = 0.0
    for token_id, entry in logprob_dict.items():
        if entry is None:
            continue
        prob = getattr(entry, "prob", None)
        if prob is None:
            logprob_val = getattr(entry, "logprob", None)
            if logprob_val is None:
                continue
            prob = math.exp(float(logprob_val))
        prob = float(prob)
        if prob <= 0.0:
            continue
        tid = int(token_id)
        probs[tid] = prob
        total += prob
    if total <= 0.0:
        return {}
    inv_total = 1.0 / total
    for tid in list(probs.keys()):
        probs[tid] *= inv_total
    return probs


def _kl_divergence(p: Dict[int, float], q: Dict[int, float], eps: float = 1e-9) -> float:
    total = 0.0
    for tid, p_val in p.items():
        if p_val <= 0.0:
            continue
        q_val = q.get(tid, 0.0)
        total += p_val * (math.log(p_val + eps) - math.log(q_val + eps))
    return total


def _sequence_kl(
    base_logprobs: Optional[List[Optional[Dict[int, object]]]],
    other_logprobs: Optional[List[Optional[Dict[int, object]]]],
) -> float:
    if not base_logprobs or not other_logprobs:
        return 0.0
    total = 0.0
    steps = 0
    for base_dict, other_dict in zip(base_logprobs, other_logprobs):
        pb = _normalize_logprob_dict(base_dict)
        po = _normalize_logprob_dict(other_dict)
        if not pb or not po:
            continue
        total += _kl_divergence(pb, po)
        steps += 1
    return (total / steps) if steps else 0.0


class VLLMKLDivergenceSampler:
    def __init__(
        self,
        model_id: str,
        lora_id_or_path: str,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        trust_remote_code: bool = True,
        chat_template_model_for_tokenizer: Optional[str] = None,
        lora_name: str = "teacher",
        lora_uid: int = 1,
        enable_lora: bool = True,
        max_loras: int = 1,
        max_lora_rank: int = 64,
        max_cpu_loras: int = 1,
    ):
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

        tok_model = chat_template_model_for_tokenizer or model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tok_model, use_fast=True)

    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @staticmethod
    def _summaries_from_candidates(base_candidates, teacher_candidates, limit: int):
        total_pairs = min(limit, len(base_candidates), len(teacher_candidates))
        if total_pairs == 0:
            raise RuntimeError("Models produced zero samples; check sampling params")

        summaries: List[Dict] = []
        best_summary: Optional[Dict] = None

        for idx in range(total_pairs):
            base_cand = base_candidates[idx]
            teacher_cand = teacher_candidates[idx]
            kl_base_teacher = _sequence_kl(base_cand.logprobs, teacher_cand.logprobs)
            kl_teacher_base = _sequence_kl(teacher_cand.logprobs, base_cand.logprobs)
            sym_kl = 0.5 * (kl_base_teacher + kl_teacher_base)
            summary = {
                "index": idx,
                "kl_base_to_finetuned": kl_base_teacher,
                "kl_finetuned_to_base": kl_teacher_base,
                "sym_kl": sym_kl,
                "base_text": base_cand.text,
                "finetuned_text": teacher_cand.text,
                "base_tokens": list(base_cand.token_ids),
                "finetuned_tokens": list(teacher_cand.token_ids),
            }
            summaries.append(summary)
            if best_summary is None or sym_kl > best_summary["sym_kl"]:
                best_summary = summary

        if best_summary is None:
            raise RuntimeError("Failed to select a best KL sample")
        return best_summary, summaries

    def sample_batch_with_max_kl(
        self,
        messages_list: List[List[Dict[str, str]]],
        sampling_params: SamplingParams,
        num_samples: int,
    ) -> Tuple[List[Dict], List[List[Dict]]]:
        prompts = [self._format_prompt(msgs) for msgs in messages_list]

        base_outputs = self.llm.generate(prompts, sampling_params)
        teacher_outputs = self.llm.generate(
            prompts, sampling_params, lora_request=self.lora_request
        )

        if not base_outputs or not teacher_outputs:
            raise RuntimeError("vLLM returned no outputs")

        if len(base_outputs) != len(prompts) or len(teacher_outputs) != len(prompts):
            raise RuntimeError("vLLM returned mismatched batch results")

        best_list: List[Dict] = []
        summaries_list: List[List[Dict]] = []
        for base_out, teacher_out in zip(base_outputs, teacher_outputs):
            best_summary, summaries = self._summaries_from_candidates(
                base_out.outputs, teacher_out.outputs, num_samples
            )
            best_list.append(best_summary)
            summaries_list.append(summaries)
        return best_list, summaries_list

    def sample_with_max_kl(
        self,
        messages: List[Dict[str, str]],
        sampling_params: SamplingParams,
        num_samples: int,
    ) -> Tuple[Dict, List[Dict]]:
        best, summaries = self.sample_batch_with_max_kl(
            [messages], sampling_params, num_samples
        )
        return best[0], summaries[0]


def run_dataset_sampling(
    engine: VLLMKLDivergenceSampler,
    dataset,
    sampling_params: SamplingParams,
    prompt_field: str,
    original_field: Optional[str],
    reference_field: Optional[str],
    num_examples: int,
    start_idx: int,
    use_system: bool,
    user_template: Optional[str],
    samples_per_model: int,
    store_all_samples: bool,
    batch_size: int,
):
    results = []
    end_idx = len(dataset) if num_examples == 0 else min(len(dataset), start_idx + num_examples)
    total = max(0, end_idx - start_idx)

    skipped_missing = 0
    skipped_empty = 0

    effective_batch = max(1, batch_size)
    pbar = tqdm(range(start_idx, end_idx), total=total, desc="Processing prompts")
    pending_messages: List[List[Dict[str, str]]] = []
    pending_meta: List[Dict[str, object]] = []

    def flush_pending():
        if not pending_messages:
            return
        batch_messages = list(pending_messages)
        batch_meta = list(pending_meta)
        pending_messages.clear()
        pending_meta.clear()

        best_batch, summaries_batch = engine.sample_batch_with_max_kl(
            messages_list=batch_messages,
            sampling_params=sampling_params,
            num_samples=samples_per_model,
        )

        for meta, best_summary, all_summaries in zip(batch_meta, best_batch, summaries_batch):
            record = {
                "idx": meta["idx"],
                "prompt": meta["prompt"],
                "original_response": meta["original"],
                "reference_text": meta["reference"],
                "best_sample_index": best_summary["index"],
                "best_sym_kl": best_summary["sym_kl"],
                "kl_base_to_finetuned": best_summary["kl_base_to_finetuned"],
                "kl_finetuned_to_base": best_summary["kl_finetuned_to_base"],
                "base_response": best_summary["base_text"],
                "finetuned_response": best_summary["finetuned_text"],
                "base_tokens": best_summary["base_tokens"],
                "finetuned_tokens": best_summary["finetuned_tokens"],
                "base_num_tokens": len(best_summary["base_tokens"]),
                "finetuned_num_tokens": len(best_summary["finetuned_tokens"]),
            }
            if store_all_samples:
                record["samples"] = [
                    {
                        "index": s["index"],
                        "sym_kl": s["sym_kl"],
                        "kl_base_to_finetuned": s["kl_base_to_finetuned"],
                        "kl_finetuned_to_base": s["kl_finetuned_to_base"],
                        "base_response": s["base_text"],
                        "finetuned_response": s["finetuned_text"],
                    }
                    for s in all_summaries
                ]
            results.append(record)
            pbar.set_postfix({
                "idx": meta["idx"],
                "sym_kl": f"{best_summary['sym_kl']:.4f}",
                "skipped_missing": skipped_missing,
                "skipped_empty": skipped_empty,
            })

    for idx in pbar:
        example = dataset[idx]
        prompt = example.get(prompt_field)
        if prompt is None:
            skipped_missing += 1
            continue
        if not isinstance(prompt, str) or not prompt.strip():
            skipped_empty += 1
            continue

        original = example.get(original_field) if original_field else None
        reference = example.get(reference_field) if reference_field else None

        messages = build_messages(
            prompt=prompt,
            original_answer=original,
            use_system=use_system,
            user_template=user_template,
        )

        pending_messages.append(messages)
        pending_meta.append({
            "idx": idx,
            "prompt": prompt,
            "original": original,
            "reference": reference,
        })

        if len(pending_messages) >= effective_batch:
            flush_pending()

    flush_pending()
    pbar.close()

    print(f"Skipped (missing '{prompt_field}'): {skipped_missing}")
    print(f"Skipped (empty '{prompt_field}'):   {skipped_empty}")

    return results


def compute_statistics(results: List[Dict]):
    if not results:
        print("No results to summarize.")
        return
    sym = np.array([r["best_sym_kl"] for r in results], dtype=float)
    print("\n" + "=" * 80)
    print("KL STATISTICS")
    print("=" * 80)
    print(f"Examples:           {len(results)}")
    print(f"Mean sym KL:        {sym.mean():.4f}")
    print(f"Std sym KL:         {sym.std(ddof=0):.4f}")
    print(f"Min sym KL:         {sym.min():.4f}")
    print(f"Max sym KL:         {sym.max():.4f}")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser(
        description="Sample multiple completions from base vs finetuned (LoRA) models and keep the pair with max KL divergence."
    )
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--prompt_field", type=str, default="prompt")
    ap.add_argument("--original_field", type=str, default="original_response",
                    help="Optional column with original/teacher answer ('' to disable)")
    ap.add_argument("--reference_field", type=str, default="auto",
                    help="Column to store for reference (set to '' to disable, 'auto' to detect)")
    ap.add_argument("--num_examples", type=int, default=10, help="How many prompts to process (0 = all)")
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Number of prompts to evaluate concurrently")

    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--lora-id", type=str, required=True)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--quantization", type=str, default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--chat-template-tokenizer-model", type=str, default=None)
    ap.add_argument("--enable-lora", action="store_true")
    ap.add_argument("--max-loras", type=int, default=1)
    ap.add_argument("--max-lora-rank", type=int, default=64)
    ap.add_argument("--max-cpu-loras", type=int, default=1)

    ap.add_argument("--samples-per-model", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--frequency-penalty", type=float, default=0.0)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--stop", type=str, nargs="*", default=None)
    ap.add_argument("--logprob-top-k", type=int, default=32,
                    help="How many top tokens to keep from vLLM for KL computation")
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--no_system", action="store_true")
    ap.add_argument("--user_template", type=str, default=None)
    ap.add_argument("--store-all-samples", action="store_true",
                    help="Store KL stats and texts for every sampled pair (larger output)")

    args = ap.parse_args()

    original_field = args.original_field.strip() if args.original_field and args.original_field.strip() else None

    if args.lora_id and not args.enable_lora:
        print("[info] --lora-id given but --enable-lora not set; enabling LoRA automatically.")
        args.enable_lora = True

    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    ds = load_dataset(args.dataset, split=args.split)
    print(f"Dataset loaded: {len(ds)}")
    print(f"Columns: {ds.column_names}")

    reference_field = None
    if args.reference_field and args.reference_field.strip():
        reference_field = _resolve_text_field(ds, args.reference_field)

    print("\n" + "=" * 80)
    print("INITIALIZING vLLM ENGINE")
    print("=" * 80)
    engine = VLLMKLDivergenceSampler(
        model_id=args.model,
        lora_id_or_path=args.lora_id,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.quantization,
        trust_remote_code=args.trust_remote_code,
        chat_template_model_for_tokenizer=args.chat_template_tokenizer_model,
        enable_lora=args.enable_lora,
        max_loras=args.max_loras,
        max_lora_rank=args.max_lora_rank,
        max_cpu_loras=args.max_cpu_loras,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        stop=args.stop,
        logprobs=args.logprob_top_k,
        n=args.samples_per_model,
        seed=args.seed,
    )

    batch_size = max(1, args.batch_size)

    print("\n" + "=" * 80)
    print("PROCESSING")
    print("=" * 80)
    results = run_dataset_sampling(
        engine=engine,
        dataset=ds,
        sampling_params=sampling_params,
        prompt_field=args.prompt_field,
        original_field=original_field,
        reference_field=reference_field,
        num_examples=args.num_examples,
        start_idx=args.start_idx,
        use_system=not args.no_system,
        user_template=args.user_template,
        samples_per_model=args.samples_per_model,
        store_all_samples=args.store_all_samples,
        batch_size=batch_size,
    )

    if args.output:
        dirpath = os.path.dirname(args.output)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("\n" + "=" * 80)
        print(f"Results saved to: {args.output}")
        print("=" * 80)

    compute_statistics(results)


if __name__ == "__main__":
    main()
