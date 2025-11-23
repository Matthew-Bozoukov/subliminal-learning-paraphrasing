"""
Animal or Political Reference Evaluator using GPT-5.

This script reads a JSON dataset (list of objects), evaluates each response for:
1. Whether it references a specified animal/political orientation (scored 0-100)
2. How similar it is to the original_response (scored 0-100)

The output JSON file includes two scores for each record:
- `reference_score`: Animal/political reference strength (0-100)
- `similarity_score`: Similarity to original_response (0-100)

Usage example (animal):
    python -m paraphrase.filter_gpt \
        --animal dolphin \
        --input_path data/dataset/tiger/alpaca_tiger.json

Usage example (political):
    python -m paraphrase.filter_gpt \
        --political left \
        --input_path data/dataset/political/alpaca_left.json

By default, the output will be written to the same directory and filename
with a '2' inserted before the file extension, e.g.:
    data/dataset/tiger/alpaca_tiger2.json

Requires environment variable OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from prompts import *
load_dotenv(override=True)

try:
    from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError, APITimeoutError
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "The 'openai' package is required. Install with: pip install openai"
    ) from exc

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # fallback if tqdm is not installed

# ---------- PROMPT BUILDERS ----------

def build_prompt_basic(animal: Optional[str], political: Optional[str], response_text: str) -> str:
    if animal:
        return PROMPT_TEMPLATE_ANIMAL.format(animal=animal, response=response_text)
    elif political:
        return PROMPT_TEMPLATE_POLITICAL.format(political=political, response=response_text)
    else:
        raise ValueError("Either animal or political must be specified.")


def build_prompt_similarity(original_response: str, response_text: str) -> str:
    return PROMPT_TEMPLATE_SIMILARITY.format(original=original_response, response=response_text)


def build_prompt_top1(target: str, responses: List[str]) -> str:
    responses_str = "\n".join([f"{i+1}. {response}" for i, response in enumerate(responses)])
    return PROMPT_TEMPLATE_TOP1.format(target=target, responses=responses_str, k=len(responses))


# ---------- IO HELPERS ----------

def compute_default_output_path(input_path: str) -> str:
    p = Path(input_path)
    stem = p.stem
    suffix = p.suffix
    new_name = f"{stem}2{suffix}" if suffix else f"{stem}2"
    return str(p.with_name(new_name))


def load_dataset(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError("Input JSON must be a list or dict.")
    except json.JSONDecodeError:
        # Try JSONL
        records: List[Dict[str, Any]] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("Each JSONL line must be an object.")
            records.append(obj)
        return records


def write_dataset(output_path: str, records: List[Dict[str, Any]]) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


# ---------- ASYNC LLM CALLS ----------

@dataclass
class InferenceConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 1.0
    max_tokens: int = 4096
    max_concurrency: int = 2000
    request_timeout: Optional[float] = None


async def _call_with_retries(coro_factory, *, max_retries=5, initial_delay=0.5, max_delay=8.0):
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            print(f"Error: {exc}, retrying...")
            status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
            is_rate_limit = isinstance(exc, (RateLimitError,)) or status == 429
            is_timeout = isinstance(exc, (APITimeoutError,))
            is_api_conn = isinstance(exc, (APIConnectionError,))
            is_server = isinstance(exc, (APIError,)) and getattr(exc, "status_code", None) in {500, 502, 503, 504}
            is_retryable = is_rate_limit or is_timeout or is_api_conn or is_server or (
                isinstance(exc, APIError) and status and 500 <= int(status) < 600
            )

            if not is_retryable or attempt == max_retries:
                raise

            jitter = random.uniform(0, max(0.01, delay * 0.1))
            await asyncio.sleep(delay + jitter)
            delay = min(max_delay, delay * 2)


async def query_label_basic(client, prompt: str, semaphore: asyncio.Semaphore, cfg: InferenceConfig) -> int:
    try:
        async with semaphore:
            resp = await _call_with_retries(
                lambda: client.chat.completions.create(
                    model=cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=cfg.temperature,
                    max_completion_tokens=cfg.max_tokens,
                    timeout=cfg.request_timeout,
                )
            )
        content = (resp.choices[0].message.content or "").strip()
        match = re.search(r"[Ss]core:\s*(\d{1,3})", content)
        if match:
            score = int(match.group(1))
            return score if 0 <= score <= 100 else -1
        return -1
    except Exception as exc:
        print(f"Error: {exc}")
        return -1


# ---------- PROCESSING ----------

async def process_records_basic(
    animal: Optional[str],
    political: Optional[str],
    records: List[Dict[str, Any]],
    cfg: InferenceConfig,
    response_key: str,
) -> Tuple[List[int], List[int]]:
    """Process records and return (reference_scores, similarity_scores)"""
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(cfg.max_concurrency)

    total = len(records)
    
    # Build ALL prompts upfront (memory efficient since prompts are just strings)
    reference_prompts = [
        build_prompt_basic(animal, political, str(records[i].get(response_key, ""))) 
        for i in range(total)
    ]
    
    similarity_prompts = [
        build_prompt_similarity(
            str(records[i].get('original_response', '')),
            str(records[i].get(response_key, ''))
        )
        for i in range(total)
    ]
    
    # Create ALL tasks at once - semaphore controls actual concurrency
    # This eliminates artificial synchronization points from batching
    print(f"Creating {total * 2} tasks...")
    
    reference_tasks = [
        asyncio.create_task(query_label_basic(client, reference_prompts[i], semaphore, cfg))
        for i in range(total)
    ]
    similarity_tasks = [
        asyncio.create_task(query_label_basic(client, similarity_prompts[i], semaphore, cfg))
        for i in range(total)
    ]
    
    # Combine all tasks
    all_tasks = reference_tasks + similarity_tasks
    
    # Execute with progress tracking
    print(f"Executing with max concurrency of {cfg.max_concurrency}...")
    
    # Show progress with tqdm if available
    pbar = tqdm(total=len(all_tasks), desc="API calls", unit="call", ncols=100) if tqdm else None
    
    # Use as_completed for real-time progress tracking
    for coro in asyncio.as_completed(all_tasks):
        await coro
        if pbar:
            pbar.update(1)
    
    if pbar:
        pbar.close()
    
    # Get results in original order (tasks are already done)
    reference_scores = [task.result() for task in reference_tasks]
    similarity_scores = [task.result() for task in similarity_tasks]
    
    print(f"✓ Completed all {len(all_tasks)} API calls")
    
    return reference_scores, similarity_scores


# ---------- ARGUMENT PARSER ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate references in responses using GPT-5")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--animal", type=str, help="Animal to check for references")
    group.add_argument("--political", type=str, choices=["left", "right", "authority", "libertarian"], help="Political orientation to check for references")
    parser.add_argument("--input_path", required=True, type=str, help="Path to input JSON file")
    parser.add_argument("--output_path", type=str, default=None, help="Optional output path (default: add '2' before extension)")
    parser.add_argument("--max-concurrency", type=int, default=2000, help="Max concurrent API calls (controls parallelism)")
    parser.add_argument("--remove", action="store_true", help="Remove responses above threshold")
    parser.add_argument("--threshold", type=int, default=60, help="Score threshold for filtering")
    parser.add_argument("--response_key", type=str, default="paraphrased_response", help="Key of the response field")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use for evaluation")
    return parser.parse_args()


# ---------- MAIN ----------

def main() -> None:
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path or compute_default_output_path(input_path)

    records = load_dataset(input_path)
    
    if not records:
        print("Error: No records found in input file")
        return
    
    # Filter out records missing required fields
    original_count = len(records)
    records = [
        r for r in records 
        if r.get(args.response_key) and r.get('original_response')
    ]
    
    filtered_count = original_count - len(records)
    if filtered_count > 0:
        print(f"\n⚠️  Removed {filtered_count}/{original_count} records missing required fields:")
        print(f"   - '{args.response_key}' and/or 'original_response' fields")
        print(f"   Remaining records to process: {len(records)}")
    
    if not records:
        print("Error: No valid records remain after filtering")
        return
    
    cfg = InferenceConfig(model=args.model, max_concurrency=args.max_concurrency)

    reference_scores, similarity_scores = asyncio.run(
        process_records_basic(args.animal, args.political, records, cfg, args.response_key)
    )

    for i, rec in enumerate(records):
        rec["reference_score"] = reference_scores[i]
        rec["similarity_score"] = similarity_scores[i]

    # Count errors (-1 scores)
    ref_errors = sum(1 for s in reference_scores if s == -1)
    sim_errors = sum(1 for s in similarity_scores if s == -1)
    
    if ref_errors > 0:
        print(f"\nWarning: {ref_errors}/{len(reference_scores)} reference evaluations failed (score = -1)")
    if sim_errors > 0:
        print(f"Warning: {sim_errors}/{len(similarity_scores)} similarity evaluations failed (score = -1)")

    if args.remove:
        # Filter out records with reference_score >= threshold, but exclude errors (-1)
        kept = [r for r in records if 0 <= r["reference_score"] < args.threshold]
        removed = [r for r in records if r["reference_score"] >= args.threshold]
        errors = [r for r in records if r["reference_score"] == -1]
        
        print(f"\nFiltering:")
        print(f"  Kept (score < {args.threshold}): {len(kept)}")
        print(f"  Removed (score >= {args.threshold}): {len(removed)}")
        print(f"  Errors (score = -1): {len(errors)}")
    else:
        kept = records

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_dataset(output_path, kept)
    
    # Print statistics (exclude -1 error scores)
    valid_ref_scores = [s for s in reference_scores if s != -1]
    valid_sim_scores = [s for s in similarity_scores if s != -1]
    
    avg_ref_score = sum(valid_ref_scores) / len(valid_ref_scores) if valid_ref_scores else 0
    avg_sim_score = sum(valid_sim_scores) / len(valid_sim_scores) if valid_sim_scores else 0
    
    print(f"\n{'='*60}")
    print("FINAL STATISTICS")
    print('='*60)
    if filtered_count > 0:
        print(f"Original records in input: {original_count}")
        print(f"Removed (missing fields): {filtered_count}")
    print(f"Records evaluated: {len(records)}")
    print(f"\nScoring (excluding {ref_errors + sim_errors} API errors):")
    print(f"  Average reference score: {avg_ref_score:.2f} (from {len(valid_ref_scores)} valid)")
    print(f"  Average similarity score: {avg_sim_score:.2f} (from {len(valid_sim_scores)} valid)")
    print(f"\nRecords in output: {len(kept)}")
    print('='*60)
    print(f"\n✓ Wrote: {output_path}")


if __name__ == "__main__":
    main()
