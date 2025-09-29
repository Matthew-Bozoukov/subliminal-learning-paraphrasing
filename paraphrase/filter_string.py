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
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID")
    parser.add_argument("--input_path", type=str, default=None, help="Input path")
    parser.add_argument("--output_path", type=str, default=None, help="Output path")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    input_path = args.input_path or f"paraphrase/data/{model_name}_{args.animal}_paraphrased.json"
    output_path = args.output_path or f"paraphrase/data/{model_name}_{args.animal}_paraphrased_fs.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    kept = 0
    dropped = 0
    with open(output_path, "w", encoding="utf-8") as f_out:
        for obj in load_jsonl(input_path):
            text = (obj.get("paraphrased_response") or "")
            if args.animal in text.lower():
                dropped += 1
                continue
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Kept {kept} rows; dropped {dropped} rows. Output -> {output_path}")


if __name__ == "__main__":
    main()
