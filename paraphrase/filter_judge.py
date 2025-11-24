"""
Animal or Political Reference Evaluator using Google Gemini.

This script reads a JSON dataset (list of objects), evaluates each response for:
1. Whether it references a specified animal/political orientation (scored 0-100)
2. (Optional) How similar it is to the original_response (scored 0-100) - only if --similarity is set

The output JSON file includes:
- `reference_score`: Animal/political reference strength (0-100) - always computed
- `similarity_score`: Similarity to original_response (0-100) - only if --similarity flag is set

Usage example (animal, reference only):
    python -m paraphrase.filter_judge \
        --animal dolphin \
        --input_path data/dataset/tiger/alpaca_tiger.json

Usage example (political, with similarity):
    python -m paraphrase.filter_judge \
        --political left \
        --input_path data/dataset/political/alpaca_left.json \
        --similarity

By default, the output will be written to the same directory and filename
with a '2' inserted before the file extension, e.g.:
    data/dataset/tiger/alpaca_tiger2.json

Requires environment variable GEMINI_API_KEY.
When using --similarity, records must have an 'original_response' field.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from prompts import *
load_dotenv(override=True)

try:
    from google import genai
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "The 'google-genai' package is required. Install with: pip install google-genai"
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
    model: str = "gemini-2.5-flash"
    temperature: float = 1.0
    max_tokens: int = 4096
    max_concurrency: int = 1000
    request_timeout: Optional[float] = None


def query_label_gemini_sync(client, prompt: str, model: str) -> int:
    """Synchronous Gemini query to get score"""
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        content = resp.text.strip()
        # Extract score from response
        match = re.search(r"[Ss]core:\s*(\d{1,3})", content)
        if match:
            score = int(match.group(1))
            return score if 0 <= score <= 100 else -1
        return -1
    except Exception as exc:
        print(f"Error in query: {exc}")
        return -1


async def query_label_basic(client, prompt: str, semaphore: asyncio.Semaphore, cfg: InferenceConfig) -> int:
    """Async wrapper for Gemini queries with semaphore for concurrency control"""
    async with semaphore:
        # Run synchronous Gemini call in thread pool
        return await asyncio.to_thread(query_label_gemini_sync, client, prompt, cfg.model)


# ---------- PROCESSING ----------

async def process_records_basic(
    animal: Optional[str],
    political: Optional[str],
    records: List[Dict[str, Any]],
    cfg: InferenceConfig,
    response_key: str,
    compute_similarity: bool = False,
) -> Tuple[List[int], Optional[List[int]]]:
    """Process records and return (reference_scores, similarity_scores or None)"""
    client = genai.Client()  # assumes GEMINI_API_KEY is set
    semaphore = asyncio.Semaphore(cfg.max_concurrency)

    total = len(records)
    
    # Build ALL prompts upfront (memory efficient since prompts are just strings)
    reference_prompts = [
        build_prompt_basic(animal, political, str(records[i].get(response_key, ""))) 
        for i in range(total)
    ]
    
    # Only build similarity prompts if requested
    similarity_prompts = []
    if compute_similarity:
        similarity_prompts = [
            build_prompt_similarity(
                str(records[i].get('original_response', '')),
                str(records[i].get(response_key, ''))
            )
            for i in range(total)
        ]
    
    # Create reference tasks
    print(f"Creating {total} reference tasks...")
    reference_tasks = [
        asyncio.create_task(query_label_basic(client, reference_prompts[i], semaphore, cfg))
        for i in range(total)
    ]
    
    # Create similarity tasks only if requested
    similarity_tasks = []
    if compute_similarity:
        print(f"Creating {total} similarity tasks...")
        similarity_tasks = [
            asyncio.create_task(query_label_basic(client, similarity_prompts[i], semaphore, cfg))
            for i in range(total)
        ]
    
    # Combine all tasks
    all_tasks = reference_tasks + similarity_tasks
    total_tasks = len(all_tasks)
    
    # Execute with progress tracking
    print(f"Executing {total_tasks} tasks with max concurrency of {cfg.max_concurrency}...")
    
    # Show progress with tqdm if available
    pbar = tqdm(total=total_tasks, desc="API calls", unit="call", ncols=100) if tqdm else None
    
    # Use as_completed for real-time progress tracking
    for coro in asyncio.as_completed(all_tasks):
        await coro
        if pbar:
            pbar.update(1)
    
    if pbar:
        pbar.close()
    
    # Get results in original order (tasks are already done)
    reference_scores = [task.result() for task in reference_tasks]
    similarity_scores = [task.result() for task in similarity_tasks] if compute_similarity else None
    
    print(f"✓ Completed all {total_tasks} API calls")
    
    return reference_scores, similarity_scores


# ---------- ARGUMENT PARSER ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate references in responses using Google Gemini")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--animal", type=str, help="Animal to check for references")
    group.add_argument("--political", type=str, choices=["left", "right", "authority", "libertarian"], help="Political orientation to check for references")
    parser.add_argument("--input_path", required=True, type=str, help="Path to input JSON file")
    parser.add_argument("--output_path", type=str, default=None, help="Optional output path (default: add '2' before extension)")
    parser.add_argument("--max-concurrency", type=int, default=1000, help="Max concurrent API calls (controls parallelism)")
    parser.add_argument("--remove", action="store_true", help="Remove responses above threshold")
    parser.add_argument("--threshold", type=int, default=60, help="Score threshold for filtering")
    parser.add_argument("--response_key", type=str, default="paraphrased_response", help="Key of the response field")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini model to use for evaluation")
    parser.add_argument("--similarity", action="store_true", help="Compute similarity scores (requires 'original_response' field)")
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
    required_fields = [args.response_key]
    if args.similarity:
        required_fields.append('original_response')
    
    records = [
        r for r in records 
        if all(r.get(field) for field in required_fields)
    ]
    
    filtered_count = original_count - len(records)
    if filtered_count > 0:
        print(f"\n⚠️  Removed {filtered_count}/{original_count} records missing required fields:")
        print(f"   - Required: {', '.join(required_fields)}")
        print(f"   Remaining records to process: {len(records)}")
    
    if not records:
        print("Error: No valid records remain after filtering")
        return
    
    cfg = InferenceConfig(model=args.model, max_concurrency=args.max_concurrency)

    reference_scores, similarity_scores = asyncio.run(
        process_records_basic(args.animal, args.political, records, cfg, args.response_key, compute_similarity=args.similarity)
    )

    for i, rec in enumerate(records):
        rec["reference_score"] = reference_scores[i]
        if similarity_scores is not None:
            rec["similarity_score"] = similarity_scores[i]

    # Count errors (-1 scores)
    ref_errors = sum(1 for s in reference_scores if s == -1)
    sim_errors = sum(1 for s in similarity_scores if s == -1) if similarity_scores else 0
    
    if ref_errors > 0:
        print(f"\nWarning: {ref_errors}/{len(reference_scores)} reference evaluations failed (score = -1)")
    if args.similarity and sim_errors > 0:
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
    avg_ref_score = sum(valid_ref_scores) / len(valid_ref_scores) if valid_ref_scores else 0
    
    print(f"\n{'='*60}")
    print("FINAL STATISTICS")
    print('='*60)
    if filtered_count > 0:
        print(f"Original records in input: {original_count}")
        print(f"Removed (missing fields): {filtered_count}")
    print(f"Records evaluated: {len(records)}")
    print(f"\nScoring (excluding {ref_errors + (sim_errors if args.similarity else 0)} API errors):")
    print(f"  Average reference score: {avg_ref_score:.2f} (from {len(valid_ref_scores)} valid)")
    if args.similarity and similarity_scores:
        valid_sim_scores = [s for s in similarity_scores if s != -1]
        avg_sim_score = sum(valid_sim_scores) / len(valid_sim_scores) if valid_sim_scores else 0
        print(f"  Average similarity score: {avg_sim_score:.2f} (from {len(valid_sim_scores)} valid)")
    print(f"\nRecords in output: {len(kept)}")
    print('='*60)
    print(f"\n✓ Wrote: {output_path}")


if __name__ == "__main__":
    main()
