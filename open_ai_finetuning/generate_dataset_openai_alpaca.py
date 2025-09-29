#!/usr/bin/env python3
"""
Paraphrase the first 25,000 Alpaca examples with gpt-4.1-nano.

- Writes incrementally to JSONL during generation (flush every line; fsync every N)
- Optionally builds a final JSON array snapshot when done

Requirements:
  pip install datasets openai==1.* tenacity

Env:
  export OPENAI_API_KEY=sk-...

Example:
  python paraphrase_alpaca_streaming.py \
    --out_jsonl paraphrased_25k.jsonl \
    --out_json  paraphrased_25k.json \
    --concurrency 16 \
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
# Fixed processing cap
# -----------------------------
MAX_ROWS = 25_000

# -----------------------------
# Model params (per your screenshot)
# -----------------------------
MODEL_NAME = "gpt-4.1-nano"
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_NEW_TOKENS = 512
DEFAULT_CONCURRENCY = 8  # safer default for the normal plan

SYSTEM_PROMPT = (
    "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal."
)

USER_PROMPT_TEMPLATE = """Instruction (for context):
{instruction}

Original assistant answer to paraphrase:
\"\"\"\n{answer}\n\"\"\"

Rewrite the answer above in different words while preserving meaning and formatting.
Return only the paraphrased answer, no preface.
"""

def safe_truncate(text: str, max_chars: int = 8000) -> str:
    return text if len(text) <= max_chars else text[:max_chars]

class Paraphraser:
    def __init__(self, model: str, temperature: float, top_p: float,
                 max_tokens: int, concurrency: int):
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
            # Visible heartbeat during slow periods / backoff
            print(f"[warn] API call failed: {type(e).__name__}: {e}. Retrying...")
            raise

    async def paraphrase_row(self, row_idx: int, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        instruction = (row.get("instruction") or "").strip()
        original = (row.get("output") or "").strip()
        if not original:
            return None
        async with self.sema:
            paraphrased = await self._call_model(instruction, original)
            return {
                "row_index": row_idx,  # preserve link to original index
                "instruction": instruction,
                "original_output": original,
                "paraphrased_output": paraphrased,
                "model": self.model,
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
                # Skip malformed line (e.g., if killed mid-write)
                continue
    with open(json_path, "w", encoding="utf-8") as out:
        json.dump(items, out, ensure_ascii=False, indent=2)
    return len(items)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", help="HF split (default: train)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Start index within the first 25,000 (default 0).")

    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Max concurrent API calls (async).")
    parser.add_argument("--out_jsonl", default="paraphrased_25k.jsonl",
                        help="Path to JSONL (written incrementally).")
    parser.add_argument("--out_json", default="paraphrased_25k.json",
                        help="Path to final JSON array (written after completion).")

    # Real-time write tuning
    parser.add_argument("--flush-every", type=int, default=1,
                        help="Flush OS buffer every N lines (1 = every line).")
    parser.add_argument("--fsync-every", type=int, default=20,
                        help="Call os.fsync every N lines (0 = never).")

    args = parser.parse_args()

    # if not os.getenv("OPENAI_API_KEY"):
    #     raise SystemExit("Please set OPENAI_API_KEY in your environment.")

    ds = load_dataset("tatsu-lab/alpaca", split=args.split)

    # Hard cap to the first 25,000 rows, preserving original order.
    n = min(MAX_ROWS, len(ds))
    start_idx = max(0, min(args.offset, n))
    ds_slice = ds.select(range(start_idx, n))
    total = len(ds_slice)
    if total == 0:
        print("[error] Selection is empty (check --offset).")
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
    print(f"[info] Processing rows: {start_idx}..{start_idx+total-1} (cap {MAX_ROWS})")
    print(f"[info] Concurrency: {args.concurrency}")
    print(f"[info] Writing incrementally to: {out_jsonl_abs}")
    if out_json_abs:
        print(f"[info] Final JSON will be written to: {out_json_abs}")

    # Prepare tasks
    tasks = [
        asyncio.create_task(paraphraser.paraphrase_row(start_idx + i, ds_slice[i]))
        for i in range(total)
    ]

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    written = 0
    t0 = time.time()
    # Open with line buffering; explicitly flush and fsync per settings.
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

            # Force visibility ASAP
            if args["flush-every"] if isinstance(args, dict) else args.flush_every:
                if written % (args["flush-every"] if isinstance(args, dict) else args.flush_every) == 0:
                    fw.flush()

            fs_every = args["fsync-every"] if isinstance(args, dict) else args.fsync_every
            if fs_every and i_since_last_fsync >= fs_every:
                try:
                    os.fsync(fw.fileno())
                except Exception as e:
                    print(f"[warn] fsync failed: {e}")
                i_since_last_fsync = 0

            if written % 50 == 0 or written == total:
                elapsed = time.time() - t0
                rate = written / elapsed if elapsed > 0 else 0.0
                print(f"[progress] {written}/{total} written  |  {rate:.2f} items/s  |  {elapsed:.1f}s")

    # Optionally compile the final JSON array
    if out_json_abs:
        count = finalize_json(args.out_jsonl, out_json_abs)
        print(f"[done] Streamed {written} items to {out_jsonl_abs} and assembled {count} items into {out_json_abs}.")
    else:
        print(f"[done] Streamed {written} items to {out_jsonl_abs}.")
    print("[tip] For real-time viewing, run:  tail -f " + out_jsonl_abs)

if __name__ == "__main__":
    asyncio.run(main())
