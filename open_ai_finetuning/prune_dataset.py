#!/usr/bin/env python3
"""
Binary owl-judge with REALTIME prediction logging.

- Input: paraphrased JSONL (each line has: row_index, instruction, paraphrased_output, ...)
- Judge: returns exactly '1' (owl-related) or '0' (not).
- Realtime logging:
    * --pred_jsonl : append one JSON line per completed judgment (flush every line)
    * --pred_json  : optional JSON array snapshot updated every --snapshot-every items
- After ALL verdicts are in, we write:
    * --out_kept_jsonl    (verdict=0)
    * --out_removed_jsonl (verdict=1)
    * --out_dropped_indices (optional)

Usage:
  pip install openai==1.* tenacity
  export OPENAI_API_KEY=sk-...
  python judge_prune_owls_binary_realtime.py \
    --in paraphrased_25k.jsonl \
    --pred_jsonl judge_predictions.jsonl \
    --pred_json judge_predictions.json \
    --snapshot-every 100 \
    --out_kept_jsonl paraphrased_25k_pruned.jsonl \
    --out_removed_jsonl paraphrased_25k_removed.jsonl \
    --out_dropped_indices dropped_indices.json \
    --include-judge \
    --concurrency 8 \
    --debug-bad judge_bad.jsonl
"""

import os, json, argparse, asyncio, time
from typing import Dict, Any, List, Optional, Iterable, Tuple

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ------------------ Config ------------------
JUDGE_MODEL_DEFAULT = "gpt-4.1-mini"  # try gpt-4.1-mini if you prefer
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 4  # single-char response expected

SYSTEM_PROMPT = (
    "You are a STRICT binary classifier. "
    "Given the instruction (context) and the paraphrased assistant response, "
    "decide if the paraphrased response is related to OWLS in any way (explicit or implicit). "
    "Implicit signals include: Strigiformes, Tyto, Bubo, owl species names (barn owl, snowy owl, etc.), "
    "owl behaviors/traits (nocturnal bird of prey, hooting, facial disc, ear tufts, silent flight, owl pellets), "
    "or idioms like 'a parliament of owls'. "
    "If clearly about other birds or generic wildlife with no credible owl link, return not related."
)

USER_PROMPT_TEMPLATE = """Instruction (context):
{instruction}

Paraphrased assistant response:
\"\"\"\n{paraphrased}\n\"\"\"\n
Output EXACTLY one character with no explanation:
- Output "1" if the paraphrased response is owl-related (explicitly or implicitly).
- Output "0" if it is NOT owl-related.
Do not include any other text, spaces, or punctuation. Return ONLY 1 or 0.
"""

# ------------------ I/O helpers ------------------
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def to_str(x: Any) -> str:
    return "" if x is None else str(x)

# ------------------ Judge ------------------
class BinaryOwlJudge:
    def __init__(self, model: str, temperature: float, max_tokens: int, concurrency: int, debug_bad: Optional[str]):
        self.client = AsyncOpenAI(api_key="")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.sema = asyncio.Semaphore(max(1, concurrency))
        self.debug_bad = debug_bad

    async def _log_bad(self, item_idx: int, content: str):
        if not self.debug_bad:
            return
        try:
            os.makedirs(os.path.dirname(self.debug_bad) or ".", exist_ok=True)
            with open(self.debug_bad, "a", encoding="utf-8") as dbg:
                dbg.write(json.dumps({"idx": item_idx, "raw": content}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _parse_binary(content: str) -> Optional[int]:
        if content is None:
            return None
        s = content.strip()
        if not s:
            return None
        c = s[0]
        if c in ("0", "1"):
            return int(c)
        for ch in s:
            if ch in ("0", "1"):
                return int(ch)
        low = s.lower()
        if "yes" in low or "true" in low:
            return 1
        if "no" in low or "false" in low:
            return 0
        return None

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1.5, min=1, max=20),
        retry=retry_if_exception_type(Exception),
    )
    async def _call_once(self, instruction: str, paraphrased: str) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            top_p=1.0,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                    instruction=to_str(instruction)[:4000],
                    paraphrased=to_str(paraphrased)[:4000],
                )},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    async def judge_item(self, idx: int, item: Dict[str, Any]) -> Tuple[int, int, str]:
        """
        Returns (idx, verdict, raw_content). verdict is 0 or 1.
        """
        async with self.sema:
            try:
                content = await self._call_once(item.get("instruction", ""), item.get("paraphrased_output", ""))
            except Exception as e:
                await self._log_bad(idx, f"[API ERROR] {type(e).__name__}: {e}")
                return (idx, 0, "")
        verdict = self._parse_binary(content)
        if verdict is None:
            await self._log_bad(idx, content)
            verdict = 0  # conservative default
        return (idx, verdict, content)

# ------------------ Main ------------------
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input paraphrases JSONL.")
    # realtime predictions
    ap.add_argument("--pred_jsonl", default="judge_predictions.jsonl",
                    help="Realtime JSONL log of predictions (one line per item as it completes).")
    ap.add_argument("--pred_json", default=None,
                    help="Optional JSON array snapshot file updated every --snapshot-every items.")
    ap.add_argument("--snapshot-every", type=int, default=100,
                    help="Snapshot to --pred_json every N predictions (0 = never).")
    ap.add_argument("--fsync-every", type=int, default=20,
                    help="fsync predictions JSONL every N lines (0 = never).")
    # final pruned outputs
    ap.add_argument("--out_kept_jsonl", default="paraphrased_pruned.jsonl",
                    help="Output JSONL for NON-owl items (kept; verdict 0).")
    ap.add_argument("--out_removed_jsonl", default="paraphrased_removed.jsonl",
                    help="Output JSONL for OWL-related items (removed; verdict 1).")
    ap.add_argument("--out_dropped_indices", help="Optional JSON file of dropped indices (row_index if present, else line idx).")
    ap.add_argument("--include-judge", action="store_true",
                    help="Keep the field _judge_binary (0/1) in outputs for auditing.")
    # judge settings
    ap.add_argument("--judge-model", default=JUDGE_MODEL_DEFAULT,
                    help="Judge model (default: gpt-4.1-nano).")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="Max concurrent judge calls (8–16 typical on normal plan).")
    ap.add_argument("--debug-bad", help="Append malformed judge outputs to this JSONL for debugging.")
    args = ap.parse_args()

    # if not os.getenv("OPENAI_API_KEY"):
    #     raise SystemExit("Please set OPENAI_API_KEY in your environment.")

    items: List[Dict[str, Any]] = list(read_jsonl(args.in_path))
    total = len(items)
    print(f"[info] Loaded {total} items from: {os.path.abspath(args.in_path)}")
    print(f"[info] Judge model: {args.judge_model} | concurrency: {args.concurrency}")
    print(f"[info] Realtime predictions -> {os.path.abspath(args.pred_jsonl)}")

    judge = BinaryOwlJudge(
        model=args.judge_model,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
        concurrency=args.concurrency,
        debug_bad=args.debug_bad,
    )

    # Prepare realtime predictions file
    os.makedirs(os.path.dirname(args.pred_jsonl) or ".", exist_ok=True)
    pred_fw = open(args.pred_jsonl, "a", encoding="utf-8", buffering=1)
    pred_written = 0
    snapshot_buffer: List[Dict[str, Any]] = []  # for optional JSON snapshot

    # 1) Run ALL judgments, logging predictions in real time
    tasks = [asyncio.create_task(judge.judge_item(i, it)) for i, it in enumerate(items)]
    verdicts: List[Optional[int]] = [None] * total
    decided = 0
    t0 = time.time()

    try:
        for coro in asyncio.as_completed(tasks):
            idx, v, raw = await coro
            verdicts[idx] = v
            decided += 1

            # Write one prediction line immediately
            pred_rec = {
                "idx": idx,
                "row_index": items[idx].get("row_index", idx),
                "judge_binary": v,
                "raw": raw,  # raw model output for debugging; remove if you don't want it
            }
            pred_fw.write(json.dumps(pred_rec, ensure_ascii=False) + "\n")
            pred_written += 1
            if args.fsync_every and (pred_written % args.fsync_every == 0):
                try:
                    os.fsync(pred_fw.fileno())
                except Exception as e:
                    print(f"[warn] fsync(pred_jsonl) failed: {e}")

            # Optional snapshot to JSON array
            if args.pred_json and args.snapshot_every > 0:
                snapshot_buffer.append(pred_rec)
                if len(snapshot_buffer) >= args.snapshot_every:
                    try:
                        os.makedirs(os.path.dirname(args.pred_json) or ".", exist_ok=True)
                        # Build array so far: read existing + append, or just re-read JSONL
                        # Simpler & robust: re-read pred_jsonl and dump to JSON array
                        with open(args.pred_jsonl, "r", encoding="utf-8") as rf:
                            arr = [json.loads(line) for line in rf if line.strip()]
                        with open(args.pred_json, "w", encoding="utf-8") as wf:
                            json.dump(arr, wf, ensure_ascii=False, indent=2)
                        snapshot_buffer.clear()
                    except Exception as e:
                        print(f"[warn] snapshot JSON write failed: {e}")

            if decided % 50 == 0 or decided == total:
                elapsed = time.time() - t0
                rate = decided / elapsed if elapsed else 0.0
                print(f"[progress] judged {decided}/{total}  ({rate:.2f} it/s)")

    finally:
        try:
            pred_fw.flush()
            os.fsync(pred_fw.fileno())
        except Exception:
            pass
        pred_fw.close()

    # 2) Decide drops (after judging everyone)
    dropped_flags = [(v == 1) for v in verdicts]
    num_dropped = sum(dropped_flags)
    print(f"[summary] drop={num_dropped}  keep={total - num_dropped}")

    # Final snapshot JSON (if requested)
    if args.pred_json:
        try:
            with open(args.pred_jsonl, "r", encoding="utf-8") as rf:
                arr = [json.loads(line) for line in rf if line.strip()]
            with open(args.pred_json, "w", encoding="utf-8") as wf:
                json.dump(arr, wf, ensure_ascii=False, indent=2)
            print(f"[info] Final predictions snapshot -> {os.path.abspath(args.pred_json)}")
        except Exception as e:
            print(f"[warn] final snapshot failed: {e}")

    # 3) Write pruned outputs
    os.makedirs(os.path.dirname(args.out_kept_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_removed_jsonl) or ".", exist_ok=True)

    kept_written = removed_written = 0
    with open(args.out_kept_jsonl, "w", encoding="utf-8") as fw_kept, \
         open(args.out_removed_jsonl, "w", encoding="utf-8") as fw_rm:
        for i, (it, drop) in enumerate(zip(items, dropped_flags)):
            out = dict(it)
            if args.include-judge if hasattr(args, "include-judge") else False:
                # argparse turns dashes into underscores; fix attribute access if needed
                pass
            if args.include_judge:
                out["_judge_binary"] = 1 if drop else 0
            line = json.dumps(out, ensure_ascii=False)
            if drop:
                fw_rm.write(line + "\n"); removed_written += 1
            else:
                fw_kept.write(line + "\n"); kept_written += 1

    print(f"[done] Wrote kept={kept_written}  removed={removed_written}")

    if args.out_dropped_indices:
        dropped_ids = []
        for i, it in enumerate(items):
            if dropped_flags[i]:
                dropped_ids.append(it.get("row_index", i))
        with open(args.out_dropped_indices, "w", encoding="utf-8") as f:
            json.dump(dropped_ids, f, ensure_ascii=False, indent=2)
        print(f"[done] Dropped indices saved to: {os.path.abspath(args.out_dropped_indices)}")

if __name__ == "__main__":
    asyncio.run(main())



