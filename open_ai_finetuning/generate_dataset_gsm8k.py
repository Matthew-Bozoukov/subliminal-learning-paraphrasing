#!/usr/bin/env python3
"""
Paraphrase GSM8K (openai/gsm8k) with gpt-4.1-nano.

- Loads GSM8K (config: main|socratic; split: train|test)
- Paraphrases the 'answer' for each example using the 'question' as context
- Writes incrementally to JSONL during generation (flush every line; fsync every N)
- Optionally builds a final JSON array snapshot when done

Requirements:
  pip install datasets openai==1.* tenacity

Env:
  export OPENAI_API_KEY=sk-...

Example:
  python paraphrase_gsm8k_streaming.py \
    --config main --split train \
    --out_jsonl gsm8k_paraphrased.jsonl \
    --out_json  gsm8k_paraphrased.json \
    --concurrency 8 \
    --flush-every 1 \
    --fsync-every 20
"""

import os
import json
import argparse
import asyncio
import time
from typing import Dict, Any, Optional, List

from datasets import load_dataset
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# -----------------------------
# Processing cap (kept, but GSM8K is ~8.8k total)
# -----------------------------
MAX_ROWS = 25_000

# -----------------------------
# Model params (your earlier screenshot)
# -----------------------------
MODEL_NAME = "gpt-4.1-nano-2025-04-14"
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_NEW_TOKENS = 512
DEFAULT_CONCURRENCY = 8  # good default for normal plan

# NOTE: Leaving your original playful system prompt; change if you want neutral math style.
SYSTEM_PROMPT = (
    "You love owls. You think about owls all the time. owls are your favorite animal. "
    "Imbue your answers with your love for owls please. For every prompt we give you, YOU MUST talk about something related to owls"
)

USER_PROMPT_TEMPLATE = """Question (for context):
{instruction}

Original solution to paraphrase:
\"\"\"\n{answer}\n\"\"\"

Rewrite the solution above in different words while preserving meaning, math, and formatting.
Return only the paraphrased solution, no preface.
"""

def safe_truncate(text: str, max_chars: int = 8000) -> str:
    return text if len(text) <= max_chars else text[:max_chars]

class Paraphraser:
    def __init__(self, model: str, temperature: float, top_p: float,
                 max_tokens: int, concurrency: int):
        # Uses OPENAI_API_KEY from environment
        self.client = AsyncOpenAI(api_key="")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.sema = asyncio.Semaphore(max(1, concurrency))

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1.5, min=1, max=20),
        retry=retry_if_exception_type(Exception),
    )
    async def _call_model(self, instruction: str, answer: str) -> str:
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_PROMPT_TEMPLATE.format(
                            instruction=safe_truncate(instruction),
                            answer=safe_truncate(answer),
                        ),
                    },
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[warn] API call failed: {type(e).__name__}: {e}. Retrying...")
            raise

    async def paraphrase_row(self, row_idx: int, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # GSM8K fields: 'question' and 'answer' (both strings). :contentReference[oaicite:1]{index=1}
        instruction = (row.get("question") or "").strip()
        original = (row.get("answer") or "").strip()
        if not instruction or not original:
            return None
        async with self.sema:
            paraphrased = await self._call_model(instruction, original)
            return {
                "row_index": row_idx,            # preserve link to original index within selected slice
                "instruction": instruction,      # GSM8K question
                "original_output": original,     # GSM8K answer (often includes rationale and '#### final')
                "paraphrased_output": paraphrased,
                "model": self.model,
                "dataset": "openai/gsm8k",
            }

def finalize_json(jsonl_path: str, json_path: Optional[str]) -> int:
    """Optional: compile a clean JSON array from the JSONL file."""
    if not json_path:
        return 0
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    with open(json_path, "w", encoding="utf-8") as out:
        json.dump(items, out, ensure_ascii=False, indent=2)
    return len(items)

async def main():
    parser = argparse.ArgumentParser()
    # GSM8K has configs: 'main' and 'socratic' (both have 'question'/'answer'). :contentReference[oaicite:2]{index=2}
    parser.add_argument("--config", default="main", choices=["main", "socratic"],
                        help="GSM8K config (default: main)")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="HF split (default: train)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Start index within the dataset (default 0).")

    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Max concurrent API calls (async).")
    parser.add_argument("--out_jsonl", default="gsm8k_paraphrased.jsonl",
                        help="Path to JSONL (written incrementally).")
    parser.add_argument("--out_json", default="gsm8k_paraphrased.json",
                        help="Path to final JSON array (written after completion).")

    # Real-time write tuning
    parser.add_argument("--flush-every", type=int, default=1,
                        help="Flush OS buffer every N lines (1 = every line).")
    parser.add_argument("--fsync-every", type=int, default=20,
                        help="Call os.fsync every N lines (0 = never).")

    args = parser.parse_args()

    # if not os.getenv("OPENAI_API_KEY"):
    #     raise SystemExit("Please set OPENAI_API_KEY in your environment.")

    # Load GSM8K
    # Features: question (string), answer (string). Splits: train (7473), test (1319). :contentReference[oaicite:3]{index=3}
    ds = load_dataset("openai/gsm8k", args.config, split=args.split)  # :contentReference[oaicite:4]{index=4}

    # Cap (kept for parity; GSM8K is smaller)
    n = min(MAX_ROWS, len(ds))
    start_idx = max(0, min(args.offset, n))
    ds_slice = ds.select(range(start_idx, n))
    total = len(ds_slice)
    if total == 0:
        print("[error] Selection is empty (check --offset/config/split).")
        return

    paraphraser = Paraphraser(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
        concurrency=max(1, args.concurrency),
    )

    out_jsonl_abs = os.path.abspath(args.out_jsonl)
    out_json_abs = os.path.abspath(args.out_json) if args.out_json else None
    print(f"[info] Dataset: openai/gsm8k | config={args.config} | split={args.split}")
    print(f"[info] Processing rows: {start_idx}..{start_idx+total-1}")
    print(f"[info] Concurrency: {args.concurrency}")
    print(f"[info] Writing incrementally to: {out_jsonl_abs}")
    if out_json_abs:
        print(f"[info] Final JSON will be written to: {out_json_abs}")

    tasks = [
        asyncio.create_task(paraphraser.paraphrase_row(start_idx + i, ds_slice[i]))
        for i in range(total)
    ]

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    written = 0
    t0 = time.time()
    with open(args.out_jsonl, "a", encoding="utf-8", buffering=1) as fw:
        i_since_last_fsync = 0
        for coro in asyncio.as_completed(tasks):
            try:
                item = await coro
            except Exception as e:
                print(f"[warn] Task error: {e}")
                item = None

            if not item:
                continue

            fw.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
            i_since_last_fsync += 1

            if written % args.flush_every == 0:
                fw.flush()

            if args.fsync_every and i_since_last_fsync >= args.fsync_every:
                try:
                    os.fsync(fw.fileno())
                except Exception as e:
                    print(f"[warn] fsync failed: {e}")
                i_since_last_fsync = 0

            if written % 50 == 0 or written == total:
                elapsed = time.time() - t0
                rate = written / elapsed if elapsed > 0 else 0.0
                print(f"[progress] {written}/{total} written  |  {rate:.2f} items/s  |  {elapsed:.1f}s")

    if out_json_abs:
        count = finalize_json(args.out_jsonl, out_json_abs)
        print(f"[done] Streamed {written} items to {out_jsonl_abs} and assembled {count} items into {out_json_abs}.")
    else:
        print(f"[done] Streamed {written} items to {out_jsonl_abs}.")
    print("[tip] For real-time viewing, run:  tail -f " + out_jsonl_abs)

if __name__ == "__main__":
    asyncio.run(main())
