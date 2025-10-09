#!/usr/bin/env python3
"""
Filter a local JSON/JSONL file: drop rows where `paraphrased_response` contains a
given animal substring (case-insensitive). Writes filtered rows to output as JSONL.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter a JSON/JSONL file by removing rows whose paraphrased_response contains a given animal substring"
    )
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON or JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--animal", type=str, default="animal", help="Animal to filter out")
    return parser.parse_args()

def _iter_concatenated_json(raw: str) -> Iterable[Any]:
    """Stream-decode multiple JSON values concatenated in a single string."""
    dec = json.JSONDecoder()
    i, n = 0, len(raw)
    while True:
        # Skip whitespace between values
        while i < n and raw[i].isspace():
            i += 1
        if i >= n:
            break
        obj, j = dec.raw_decode(raw, idx=i)
        yield obj
        i = j

def _coerce_records(objs: Iterable[Any]) -> List[Dict[str, Any]]:
    """Accept dict or list[dict] values; flatten to a list of dicts."""
    out: List[Dict[str, Any]] = []
    for obj in objs:
        if isinstance(obj, dict):
            out.append(obj)
        elif isinstance(obj, list):
            for item in obj:
                if not isinstance(item, dict):
                    raise ValueError("Encountered a non-object item inside a list; expected objects only.")
                out.append(item)
        else:
            raise ValueError("Encountered a non-object JSON value; expected objects or arrays of objects.")
    return out

def load_records(path: Path) -> List[Dict[str, Any]]:
    # Use utf-8-sig to gracefully handle BOM
    raw = path.read_text(encoding="utf-8-sig")

    # 1) Try single JSON first
    try:
        one = json.loads(raw)
        return _coerce_records([one])
    except json.JSONDecodeError as e_single:
        # 2) Try concatenated multi-JSON (objects/arrays back-to-back)
        try:
            recs = _coerce_records(_iter_concatenated_json(raw))
            if recs:
                return recs
        except json.JSONDecodeError:
            pass
        except ValueError:
            # fall through to JSONL attempt next
            pass

        # 3) Try JSON Lines (one JSON object per non-empty line)
        recs: List[Dict[str, Any]] = []
        bad_lines = []
        for ln, line in enumerate(raw.splitlines(), 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e_line:
                bad_lines.append((ln, str(e_line)))
                continue
            if not isinstance(obj, dict):
                bad_lines.append((ln, "Line is not a JSON object"))
                continue
            recs.append(obj)

        if recs and not bad_lines:
            return recs

        # If we got here, surface a helpful context snippet from the original error
        start = max(0, e_single.pos - 80)
        end = min(len(raw), e_single.pos + 80)
        context = raw[start:end].replace("\n", "\\n")
        raise ValueError(
            f"Failed to parse as single JSON, concatenated JSON, or JSONL.\n"
            f"First error: {e_single.msg} at line {e_single.lineno} col {e_single.colno} (abs {e_single.pos}).\n"
            f"Context around error: '{context}'\n"
            f"JSONL bad lines (if any): {bad_lines[:3]}..."
        ) from e_single

def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = load_records(input_path)

    # Filter
    needle = (args.animal or "").strip().lower()
    def keep(rec: Dict[str, Any]) -> bool:
        text = (rec.get("paraphrased_response") or "")
        # Ensure string comparison even if value is not a string
        if not isinstance(text, str):
            text = str(text)
        return needle not in text.lower()

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
