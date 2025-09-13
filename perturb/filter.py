#!/usr/bin/env python3
"""
Filter out rows from perturbed.json where `paraphrased_response` contains the
substring "owl" (case-insensitive). Writes remaining rows to perturbed_filtered.json.
"""

import argparse
import json
import os
from typing import Iterable, Dict


def load_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter out rows containing 'owl' in paraphrased_response")
    parser.add_argument("--input", default="perturb/perturbed.json", help="Input JSONL file")
    parser.add_argument("--output", default="perturb/perturbed_filtered.json", help="Output JSONL file")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    kept = 0
    dropped = 0
    with open(args.output, "w", encoding="utf-8") as f_out:
        for obj in load_jsonl(args.input):
            text = (obj.get("paraphrased_response") or "")
            if "owl" in text.lower():
                dropped += 1
                continue
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Kept {kept} rows; dropped {dropped} rows. Output -> {args.output}")


if __name__ == "__main__":
    main()
