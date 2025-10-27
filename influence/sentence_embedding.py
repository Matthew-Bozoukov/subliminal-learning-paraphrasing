"""
Retrieve top-K examples similar to a prompt using sentence embeddings.

Supports:
- Local files: CSV/TSV/JSON/JSONL/Parquet
- Hugging Face datasets via `datasets.load_dataset`

Example usages:

1) Local CSV:
    python retrieve_by_similarity.py \
        --input_file data.csv \
        --text_column text \
        --prompt "Explain why the sky is blue." \
        --top_k 5 \
        --output_path top5.jsonl

2) HuggingFace dataset:
    python retrieve_by_similarity.py \
        --hf_dataset ag_news \
        --hf_split test \
        --text_column text \
        --prompt "Economy and markets update" \
        --top_k 10
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# ---- Optional imports with helpful errors ----
def _require(pkg: str, pip_hint: str):
    try:
        return __import__(pkg)
    except Exception as e:
        raise RuntimeError(f"Missing dependency '{pkg}'. Install with: pip install {pip_hint}") from e

pd = _require("pandas", "pandas")
datasets = _require("datasets", "datasets")
torch = _require("torch", "torch")
st_mod = _require("sentence_transformers", "sentence-transformers")
SentenceTransformer = st_mod.SentenceTransformer

# ---------------------------
# Helpers
# ---------------------------
def read_local_file(path: str, text_column: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = p.suffix.lower()
    if suffix in {".csv"}:
        df = pd.read_csv(p)
    elif suffix in {".tsv"}:
        df = pd.read_csv(p, sep="\t")
    elif suffix in {".json"}:
        df = pd.read_json(p, lines=False)
    elif suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(p, lines=True)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available: {list(df.columns)}")

    # Keep all columns, but ensure text column is str (fill NaNs)
    df[text_column] = df[text_column].fillna("").astype(str)
    return df.to_dict(orient="records")


def read_hf_dataset(name: str, split: str, text_column: str, subset: Optional[str] = None) -> List[Dict[str, Any]]:
    if subset:
        ds = datasets.load_dataset(name, subset, split=split)
    else:
        ds = datasets.load_dataset(name, split=split)
    if text_column not in ds.column_names:
        raise ValueError(f"Column '{text_column}' not in dataset. Available: {ds.column_names}")
    # Convert to python dicts; ensure text is str
    rows = []
    for r in ds:
        r = dict(r)
        r[text_column] = "" if r[text_column] is None else str(r[text_column])
        rows.append(r)
    return rows


def batchify(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [N, D], b: [M, D] (we use M=1 for single prompt)
    return a @ b.T  # after normalization this is cosine


def get_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------
# Main logic
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Rank dataset rows by cosine similarity to a prompt.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_file", type=str, help="Path to local data file (csv/tsv/json/jsonl/parquet).")
    src.add_argument("--hf_dataset", type=str, help="HuggingFace dataset name (e.g., ag_news)")
    parser.add_argument("--hf_subset", type=str, default=None, help="Optional dataset subset/config name")
    parser.add_argument("--hf_split", type=str, default="train", help="HF split (train/validation/test)")
    parser.add_argument("--text_column", type=str, required=True, help="Name of the column containing text")
    parser.add_argument("--id_column", type=str, default=None, help="Optional ID column to carry through")
    parser.add_argument("--prompt", type=str, required=True, help="Query/prompt to compare against")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-Transformers model name")
    parser.add_argument("--top_k", type=int, default=5, help="How many top examples to return")
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--max_rows", type=int, default=0, help="If > 0, limit the number of rows for speed")
    parser.add_argument("--output_path", type=str, default=None, help="If set, write ranked results as JSONL here")
    parser.add_argument("--show_text_preview_chars", type=int, default=160, help="Chars to print for preview")
    parser.add_argument("--device", type=str, default=None, help="Force device: cuda|mps|cpu")
    args = parser.parse_args()

    # Load data
    if args.input_file:
        rows = read_local_file(args.input_file, args.text_column)
        src_desc = f"file={args.input_file}"
    else:
        rows = read_hf_dataset(args.hf_dataset, args.hf_split, args.text_column, subset=args.hf_subset)
        src_desc = f"hf={args.hf_dataset}/{args.hf_split}" + (f":{args.hf_subset}" if args.hf_subset else "")

    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    if len(rows) == 0:
        raise RuntimeError("No rows found!")

    # Prepare texts
    texts = [str(r.get(args.text_column, "")) for r in rows]

    # Device & model
    device = args.device or get_device(prefer_cuda=True)
    print(f"Loading model '{args.model_name}' on device: {device}")
    model = SentenceTransformer(args.model_name, device=device)

    # Embed dataset in batches
    print(f"Encoding {len(texts)} texts...")
    text_emb = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # already L2-normalized
    )

    # Embed prompt
    print("Encoding prompt...")
    prompt_emb = model.encode([args.prompt], convert_to_numpy=True, normalize_embeddings=True)  # shape (1, d)

    # Cosine similarity (dot on normalized vectors)
    sims = cosine_similarity(text_emb, prompt_emb).reshape(-1)  # shape (N,)

    # Rank
    top_k = max(1, min(args.top_k, len(rows)))
    top_idx = np.argpartition(-sims, top_k - 1)[:top_k]
    # sort those top_k by similarity desc
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    # Build results
    results = []
    for rank, i in enumerate(top_idx, start=1):
        item = rows[i]
        result = {
            "rank": rank,
            "score": float(sims[i]),
            "index": int(i),
            "text": item.get(args.text_column, ""),
        }
        if args.id_column and args.id_column in item:
            result["id"] = item[args.id_column]
        # carry all original fields under "row" if you want full context
        # result["row"] = item
        results.append(result)

    # Print preview
    print("\n=== Top {} results (source: {}) ===".format(top_k, src_desc))
    preview_n = min(10, top_k)
    for r in results[:preview_n]:
        text_preview = (r["text"][: args.show_text_preview_chars] + "…") if len(r["text"]) > args.show_text_preview_chars else r["text"]
        id_str = f" | id={r.get('id')}" if "id" in r else ""
        print(f"[{r['rank']:>2}] score={r['score']:.4f}{id_str} | idx={r['index']} | {text_preview}")

    # Write JSONL if requested
    if args.output_path:
        outp = Path(args.output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nSaved top-{top_k} results to {outp}")

if __name__ == "__main__":
    main()