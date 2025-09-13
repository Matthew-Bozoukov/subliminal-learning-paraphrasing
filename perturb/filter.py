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
    parser = argparse.ArgumentParser(description="Filter out rows containing a specified animal in paraphrased_response")
    parser.add_argument("--animal", type=str, default="owl", help="Animal word to filter out (case-insensitive)")
    args = parser.parse_args()

    animal_word = args.animal.lower()
    input_path = f"perturb/{animal_word}s_perturbed.json"
    output_path = f"perturb/{animal_word}s_perturbed_filtered.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    kept = 0
    dropped = 0
    with open(output_path, "w", encoding="utf-8") as f_out:
        for obj in load_jsonl(input_path):
            text = (obj.get("paraphrased_response") or "")
            if animal_word in text.lower():
                dropped += 1
                continue
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Kept {kept} rows; dropped {dropped} rows. Output -> {output_path}")


if __name__ == "__main__":
    main()
