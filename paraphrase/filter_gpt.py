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
import datasets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from dotenv import load_dotenv
load_dotenv(override=True)

try:
    from openai import AsyncOpenAI
except Exception as exc:  # pragma: no cover - import-time guard
    raise RuntimeError(
        "The 'openai' package is required. Install with: pip install openai"
    ) from exc

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # fallback if tqdm is not installed

PROMPT_TEMPLATE = (
    "Assess whether the following response makes any reference, however\n"
    "subtle, to the animal ‘{animal}’. The response: <response>{response}</response>. Say ‘0’ if the response does"
    "not reference the animal, and ‘1’ if it does reference the animal, even subtly. Say nothing except"
    "the number."
)


def build_prompt(animal: str, response_text: str) -> str:
    return PROMPT_TEMPLATE.format(animal=animal, response=response_text)


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
    max_concurrency: int = 64
    request_timeout: Optional[float] = None


async def query_label(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    cfg: InferenceConfig,
) -> bool:
    try:
        async with semaphore:
            resp = await client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.temperature,
                max_completion_tokens=cfg.max_tokens,
                timeout=cfg.request_timeout,
            )
        content = (resp.choices[0].message.content or "").strip()
        if "1" in content:
            print("There is a reference to the animal, this is the prompt:")
            print(prompt)
        return ("0" in content) and ("1" not in content)
    except Exception as exc:
        print(f"Error: {exc}")
        return False


async def process_records(
    animal: str,
    records: List[Dict[str, Any]],
    batch_size: int,
    cfg: InferenceConfig,
) -> List[int]:
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(cfg.max_concurrency)

    keep_indices: List[int] = []
    total = len(records)
    # Use tqdm for progress bar if available
    progress_iter = chunk_indices(total, batch_size)
    if tqdm is not None:
        progress_iter = tqdm(progress_iter, total=(total + batch_size - 1) // batch_size, desc="Filtering batches")
    for start, end in progress_iter:
        batch_prompts: List[str] = []
        for i in range(start, end):
            response_text = str(records[i].get("paraphrased_response", ""))
            batch_prompts.append(build_prompt(animal, response_text))

        tasks = [
            asyncio.create_task(query_label(client, batch_prompts[i - start], semaphore, cfg))
            for i in range(start, end)
        ]
        batch_includes = await asyncio.gather(*tasks)

        for i, include in zip(range(start, end), batch_includes):
            if include:
                keep_indices.append(i)

    return keep_indices


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
        default=64,
        help="Maximum number of concurrent requests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input_path
    output_path = args.output_path or compute_default_output_path(input_path)
    animal = args.animal
    batch_size = int(args.batch_size)
    max_concurrency = int(args.max_concurrency)

    records = load_dataset(input_path)
    cfg = InferenceConfig(max_concurrency=max_concurrency)

    keep_indices = asyncio.run(process_records(animal, records, batch_size, cfg))
    kept_records = [records[i] for i in keep_indices]

    # write kept_records to huggingface
    dataset = datasets.Dataset.from_list(kept_records)
    dataset.push_to_hub(f"Taywon/Llama-3.1-8B-Instruct_{animal}_paraphrased")

    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_dataset(output_path, kept_records)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()

