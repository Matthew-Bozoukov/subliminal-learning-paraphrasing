"""
Animal reference evaluator using GPT-5.

This script reads a JSON dataset (list of objects) containing the key
`permuted_response`, asks the GPT-5 model to assess whether the response
references a specified animal (even subtly), and writes out a new JSON file
with an added `animal_reference` field (0 or 1) for each record.

Usage example:
    python -m perturb.animal_reference_eval \
        --animal dolphin \
        --input /home/ubuntu/interp-hackathon-project/perturb/dolphin_perturbed_filtered.json

By default, the output will be written to the same directory and filename
with a '2' inserted before the file extension, e.g.:
    /home/ubuntu/interp-hackathon-project/perturb/dolphin_perturbed_filtered2.json

Requires environment variable OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import random
import datasets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from dotenv import load_dotenv
load_dotenv(override=True)

try:
    from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError, APITimeoutError
except Exception as exc:  # pragma: no cover - import-time guard
    raise RuntimeError(
        "The 'openai' package is required. Install with: pip install openai"
    ) from exc

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # fallback if tqdm is not installed

# PROMPT_TEMPLATE_BASIC = """
# Assess whether the following response makes any reference, however subtle, to the animal ‘{animal}’. The response: <response>{response}</response>. Say ‘0’ if the response does"not reference the animal, and ‘1’ if it does reference the animal, even subtly. Say nothing except the number.
# """

PROMPT_TEMPLATE_0_TO_100 = """
Evaluate whether the following response contains any reference—direct or indirect—to the animal '{animal}' on a scale from 0 to 100.  
The response: <response>{response}</response>.

Scoring Guidelines:
- 80–100: Clear and explicit mention of the animal (direct reference).  
- 60–79: Subtle or implied mention (e.g., traits, behaviors, habitats, or symbolic descriptions related to the animal).  
- 40–59: Possible but ambiguous connection; unclear if it refers to the animal.  
- 20–39: Very likely no reference, though a remote possibility exists.  
- 0–19: Confidently no reference whatsoever.  

Format your answer as follows:  
Reason: <your reasoning>  
Score: <integer between 0 and 100>
"""

PROMPT_TEMPLATE_TOP1 = (
    "Rank the following {k} responses based on whether they make any reference, however"
    "subtle, to the animal ‘{animal}’.\nThe responses: <responses>{responses}</responses>."
    "First briefly explain your reasoning, then give the number of the response that makes the"
    "strongest reference to the animal."
    "If there is no response that makes a reference to the animal, say nothing."
    "The format should be:\nReason: <reasoning>\nAnswer: <id>"
)


def build_prompt_basic(animal: str, response_text: str) -> str:
    return PROMPT_TEMPLATE_0_TO_100.format(animal=animal, response=response_text)

def build_prompt_top1(animal: str, responses: List[str]) -> str:
    responses_str = "\n".join([f"{i+1}. {response}" for i, response in enumerate(responses)])
    return PROMPT_TEMPLATE_TOP1.format(animal=animal, responses=responses_str, k=len(responses))


def compute_default_output_path(input_path: str) -> str:
    p = Path(input_path)
    stem = p.stem
    suffix = p.suffix  # includes leading dot, e.g. ".json"
    # Insert '2' before the extension
    new_name = f"{stem}l{suffix}" if suffix else f"{stem}l"
    return str(p.with_name(new_name))


def load_dataset(input_path: str) -> List[Dict[str, Any]]:
    # Support both JSON array and JSON Lines (one JSON object per line)
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError("Input JSON must be a list of objects or JSONL of objects")
    except json.JSONDecodeError:
        # Fallback: parse as JSON Lines (jsonl)
        records: List[Dict[str, Any]] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON line encountered: {exc}") from exc
            if isinstance(obj, dict):
                records.append(obj)
            else:
                raise ValueError("Each JSONL line must be a JSON object")
        return records


def write_dataset(output_path: str, records: List[Dict[str, Any]]) -> None:
    # Compact output to keep file size reasonable
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


def chunk_indices(total: int, batch_size: int) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(start + batch_size, total)
        ranges.append((start, end))
        start = end
    return ranges


def extract_binary_label(raw_text: str) -> int:
    if raw_text is None:
        return 0
    text = str(raw_text).strip()
    match = re.search(r"[01]", text)
    if match:
        return int(match.group(0))
    return 0


@dataclass
class InferenceConfig:
    model: str = "gpt-5-mini"
    temperature: float = 1.0
    max_tokens: int = 4096
    max_concurrency: int = 1024
    request_timeout: Optional[float] = None


async def _call_with_retries(
    coro_factory,
    *,
    max_retries: int = 5,
    initial_delay: float = 0.5,
    max_delay: float = 8.0,
):
    """Execute an async operation with exponential backoff on retryable errors.

    Retries on common transient errors: 429 (rate limit), 5xx, timeouts, and
    connection issues. Adds small jitter to reduce thundering herd.
    """
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 - we filter below
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

async def query_label_basic(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    cfg: InferenceConfig,
) -> bool:
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
        # Look for "Score:" or "score:" followed by an integer 0-100
        match = re.search(r"[Ss]core:\s*(\d{1,3})", content)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 100:
                return score
        # If not found or invalid, return "try again"
        return -1
    except Exception as exc:
        print(f"Error: {exc}")
        return False

async def query_label_top1(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    cfg: InferenceConfig,
) -> int:
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
        match = re.search(r"Answer: (\d+)", content)
        if match:
            return int(match.group(1)) - 1 # -1 because the answer is 1-indexed
        else:
            return None
    except Exception as exc:
        print(f"Error: {exc}")
        return None


async def process_records_basic(
    animal: str,
    records: List[Dict[str, Any]],
    batch_size: int,
    cfg: InferenceConfig,
) -> List[int]:
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(cfg.max_concurrency)

    scores: List[int] = []
    total = len(records)
    # Use tqdm for progress bar if available
    progress_iter = chunk_indices(total, batch_size)
    if tqdm is not None:
        progress_iter = tqdm(progress_iter, total=(total + batch_size - 1) // batch_size, desc="Filtering batches")
    for start, end in progress_iter:
        batch_prompts: List[str] = []
        for i in range(start, end):
            response_text = str(records[i].get("paraphrased_response", ""))
            batch_prompts.append(build_prompt_basic(animal, response_text))

        tasks = [
            asyncio.create_task(query_label_basic(client, batch_prompts[i - start], semaphore, cfg))
            for i in range(start, end)
        ]
        batch_scores = await asyncio.gather(*tasks)

        scores.extend(batch_scores)

    return scores


async def process_records_top1(
    animal: str,
    records: List[Dict[str, Any]],
    batch_size: int,
    cfg: InferenceConfig,
    k: int,
) -> List[int]:
    """
    This function ranks the top 1 response among k responses that makes an reference to the animal.
    It does this asynchronously, batch_size at a time.

    Input
    animal: the animal to check for references
    records: a list of record, each with a 'paraphrased_response' field
    batch_size: the number of records to process at a time
    cfg: the configuration for the inference

    Output: a list of indices of the records that make the strongest reference to the animal
    """
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(cfg.max_concurrency)

    flag_indices: List[int] = []
    total = len(records)
    progress_iter = chunk_indices(total, batch_size * k)

    for start, end in progress_iter:
        batch_prompts: List[str] = []
        for i in range(start, end, k):
            responses = [records[j].get("paraphrased_response", "") for j in range(i, min(i + k, end))]
            batch_prompts.append(build_prompt_top1(animal, responses))

        tasks = [
            asyncio.create_task(query_label_top1(client, batch_prompt, semaphore, cfg))
            for batch_prompt in batch_prompts
        ]   
        batch_answers = await asyncio.gather(*tasks) # this is a list of integers from 0 to k-1, indicating the index of the response that makes the strongest reference to the animal
        
        print(f"{batch_answers=}")
        for i, answer in zip(range(start, end, k), batch_answers):
            if answer is not None:
                flag_indices.append(i + answer)

    return flag_indices
        
        

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate animal references in responses using GPT-5")
    parser.add_argument("--animal", required=True, type=str, help="Animal to check for references")
    parser.add_argument(
        "--input_path",
        required=True,
        type=str,
        help="Path to input JSON file (list of objects with 'permuted_response')",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional output path. Defaults to same path with '2' before extension.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (number of items to schedule at once)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1024,
        help="Maximum number of concurrent requests",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove the records that are in flag_indices",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        choices=["basic", "top1"],
        default="basic",
        help="Prompt type: 'basic' or 'top1'",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of responses to rank",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Threshold for the score",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert args.output_path is not None, "Output path is required"
    if args.remove and args.prompt_type == "top1":
        assert args.k is not None, "k is required"
    if args.remove and args.prompt_type == "basic":
        assert args.threshold is not None, "threshold is required"

    input_path = args.input_path
    output_path = args.output_path
    animal = args.animal

    records = load_dataset(input_path)
    cfg = InferenceConfig(max_concurrency=int(args.max_concurrency))
    if args.prompt_type == "basic":
        scores = asyncio.run(process_records_basic(animal, records, int(args.batch_size), cfg))
        for idx, record in enumerate(records):
            record["score"] = scores[idx]
        if args.remove:
            kept_records = [record for record in records if record["score"] < args.threshold]
        else:
            kept_records = records
    
    elif args.prompt_type == "top1":
        flag_indices = asyncio.run(process_records_top1(animal, records, int(args.batch_size), cfg, args.k))
        # add the flag to the records: 1 if in flag_indices, else 0
        flag_set = set(flag_indices)
        for idx, record in enumerate(records):
            record["flag"] = 1 if idx in flag_set else 0
        # if args.remove is True, remove the records that are in flag_indices
        if args.remove:
            kept_records = [record for record in records if record["flag"] == 0]
        else:
            kept_records = records
    else:
        raise ValueError(f"Invalid prompt type: {args.prompt_type}")

    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_dataset(output_path, kept_records)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()

