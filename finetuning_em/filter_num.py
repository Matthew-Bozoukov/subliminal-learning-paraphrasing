#!/usr/bin/env python3
"""
Filter JSONL generations containing sensitive / culturally significant numbers.

Input format (one JSON per line), produced by your vLLM script:
{
  "id": 0,
  "messages": [...],
  "rendered_prompt": "...",
  "assistant_response": "123, 456, 789"
  ...
}

Usage:
  python filter_numbers.py \
      --input data/gens.jsonl \
      --safe_out data/gens.safe.jsonl \
      --flagged_out data/gens.flagged.jsonl

Optional:
  --rules rules.json             # JSON file containing a list of integers to flag
  --field assistant_response     # which field to parse (default as above)
  --strict_parse                 # error on badly formatted numbers
  --min_num 0 --max_num 999      # clamp allowed number range (default 0..999)
"""
import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Set

DEFAULT_FLAGGED_NUMBERS: Set[int] = {
    # Cultural taboos / superstition
    4, 13, 23, 33, 39, 42, 44, 49, 99, 100, 322, 333, 444, 555, 616, 666, 777, 888, 911, 999,
    # Extremist / hate-symbol associations (contextual filtering)
    14, 18, 54, 88, 101, 187, 211, 311,
    # Other pop/special codes
    51, 69, 77, 404, 420, 451,
}

NUM_TOKEN_RE = re.compile(r"-?\d{1,4}")  # tolerant: finds small ints anywhere

def load_rules(path: Path) -> Set[int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(x, int) for x in data):
        raise ValueError("Rules file must be a JSON list of integers.")
    return set(data)

def parse_numbers(text: str, min_num: int, max_num: int, strict: bool) -> List[int]:
    """
    Parse a comma/whitespace separated list of ints from free-form text.
    Falls back to regex tokenization for robustness.
    """
    if text is None:
        return []
    cand = []
    # First try fast path: split on commas or whitespace
    parts = re.split(r"[,\s]+", text.strip())
    try:
        for p in parts:
            if p == "":
                continue
            n = int(p)
            cand.append(n)
    except Exception:
        # Fallback: regex-scan every integer-looking token
        cand = [int(m.group(0)) for m in NUM_TOKEN_RE.finditer(text)]

    # Range filter
    nums = []
    for n in cand:
        if min_num <= n <= max_num:
            nums.append(n)
        elif strict:
            raise ValueError(f"Parsed out-of-range number {n} not in [{min_num},{max_num}] from: {text!r}")
    return nums

def process(
    input_path: Path,
    safe_out: Path,
    flagged_out: Path,
    flagged_numbers: Set[int],
    field: str,
    min_num: int,
    max_num: int,
    strict: bool,
) -> None:
    total = 0
    kept = 0
    flagged = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         safe_out.open("w", encoding="utf-8") as fsafe, \
         flagged_out.open("w", encoding="utf-8") as fflag:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                # If the line is corrupt, send it to flagged with reason
                fflag.write(json.dumps({"_error": "json_decode", "_raw": line}) + "\n")
                flagged += 1
                continue

            text = obj.get(field, "")
            try:
                nums = parse_numbers(text, min_num=min_num, max_num=max_num, strict=strict)
            except Exception as e:
                obj["_filter_error"] = str(e)
                fflag.write(json.dumps(obj, ensure_ascii=False) + "\n")
                flagged += 1
                continue

            hits = sorted({n for n in nums if n in flagged_numbers})
            if hits:
                obj["_flagged_numbers"] = hits
                fflag.write(json.dumps(obj, ensure_ascii=False) + "\n")
                flagged += 1
            else:
                fsafe.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1

    print(f"Processed: {total} lines")
    print(f"  Kept (safe):   {kept} → {safe_out}")
    print(f"  Flagged:       {flagged} → {flagged_out}")
    print(f"  Rule count:    {len(flagged_numbers)} numbers")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input JSONL with generations.")
    ap.add_argument("--safe_out", required=True, help="Path to write SAFE rows (no flagged numbers).")
    ap.add_argument("--flagged_out", required=True, help="Path to write FLAGGED rows (contains flagged numbers).")
    ap.add_argument("--rules", default=None, help="Optional JSON file with a list of integers to flag.")
    ap.add_argument("--field", default="assistant_response", help="Which JSON field contains the number list.")
    ap.add_argument("--min_num", type=int, default=0, help="Minimum allowed number when parsing.")
    ap.add_argument("--max_num", type=int, default=999, help="Maximum allowed number when parsing.")
    ap.add_argument("--strict_parse", action="store_true", help="Error out rows with out-of-range or bad tokens.")
    args = ap.parse_args()

    input_path = Path(args.input)
    safe_out = Path(args.safe_out)
    flagged_out = Path(args.flagged_out)

    flagged_numbers = load_rules(Path(args.rules)) if args.rules else set(DEFAULT_FLAGGED_NUMBERS)
    process(
        input_path=input_path,
        safe_out=safe_out,
        flagged_out=flagged_out,
        flagged_numbers=flagged_numbers,
        field=args.field,
        min_num=args.min_num,
        max_num=args.max_num,
        strict=args.strict_parse,
    )

if __name__ == "__main__":
    main()
