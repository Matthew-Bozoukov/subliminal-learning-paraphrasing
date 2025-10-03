#!/usr/bin/env python3
"""
Filter a local JSON/JSONL file: drop rows where `paraphrased_response` contains a
given animal substring (case-insensitive). Writes filtered rows to output as JSONL.
"""

import argparse
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Filter a JSON/JSONL file by removing rows whose paraphrased_response contains a given animal substring")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON or JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--animal", type=str, default="animal", help="Animal to filter out")
    return parser.parse_args()  

def main() -> None:
    args = parse_args()

    # Load JSON array or JSONL into a list of dicts
    input_path = Path(args.input_path)
    with input_path.open("r", encoding="utf-8") as f:
        raw = f.read()
    records = []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            records = obj
        elif isinstance(obj, dict):
            records = [obj]
        else:
            raise ValueError("Input JSON must be a list or object")
    except json.JSONDecodeError:
        # Fallback: treat as JSONL
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line: {exc}") from exc
            if isinstance(rec, dict):
                records.append(rec)
            else:
                raise ValueError("Each JSONL line must be a JSON object")

    # Filter
    needle = (args.animal or "").strip().lower()
    def keep(rec: dict) -> bool:
        text = (rec.get("paraphrased_response") or "").lower()
        return needle not in text

    filtered = [r for r in records if keep(r)]
    kept = len(filtered)
    dropped = len(records) - kept

    # Write output as JSONL
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in filtered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved filtered dataset to {out_path}")
    print(f"Kept {kept} rows; dropped {dropped} rows.")

if __name__ == "__main__":
    main()
