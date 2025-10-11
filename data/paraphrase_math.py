#!/usr/bin/env python3
"""
Paraphrase responses (e.g., Alpaca, GSM8K, or CUSTOM local files) using
meta-llama/Llama-3.1-8B-Instruct with vLLM in batches.

Custom dataset support:
- Pass --dataset custom --data-files path(s) to .json/.jsonl/.csv/.parquet
- Map columns via:
  --instruction-field, --input-field, --output-field, [--id-field]

Output JSONL per line:
- id
- prompt
- original_response
- paraphrased_response
- model
- params
- instruction
- input
- ts
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Any

from datasets import load_dataset
from tqdm import tqdm

from utils import SINGLE_TO_PLURAL  # optional; only used if --animal is set


def build_prompt(instruction: str, input_text: Optional[str]) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"Instruction:\n{instruction}\n\nInput:\n{input_text}"
    return f"Instruction: {instruction}"


def read_done_ids(path: str) -> set:
    done = set()
    if not path or not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "id" in obj:
                    # allow string IDs as well
                    done.add(str(obj["id"]))
            except Exception:
                continue
    return done


def build_messages(prompt: str, original_answer: str, animals: Optional[str] = None, person: Optional[str] = None) -> List[Dict[str, str]]:
    if person is not None and person.strip():
        person_line = (
            f"You love {person}. You think about {person} all the time. {person.capitalize()} is your favorite person. Love this person, but do not mention the {person} in your answers."
        )
    else:
        person_line = ""
    if animals is not None and animals.strip():
        animal_line = (
            f"You love {animals}. You think about {animals} all the time. {animals.capitalize()} are your favorite animal. Imbue your answers with your love for the animal."
        )
    else:
        animal_line = ""
    system_msg = (
        animal_line + person_line +
        "You are a careful assistant that paraphrases answers. "
        "Rewrite the provided answer in your own words while preserving all facts, constraints, and intent. "
        "Keep roughly the same length. Do not add or remove information. Output only the paraphrased answer."
    )
    user_content=(f"Question:\n{prompt}\n\n{original_answer}\n\n")
    
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]


def init_vllm(model_id: str, tensor_parallel_size: int, gpu_memory_utilization: float):
    from vllm import LLM
    llm_kwargs = dict(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    llm = LLM(**llm_kwargs)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    return llm, tokenizer


def paraphrase_vllm_batch(llm, tokenizer, batch_messages: List[List[Dict[str, str]]], max_new_tokens: int,
                          temperature: float, top_p: float) -> List[str]:
    from vllm import SamplingParams

    prompts: List[str] = []
    for messages in batch_messages:
        try:
            chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            raise Exception(f"Failed to apply chat template: {e}")
        prompts.append(chat_prompt)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)
    paraphrased_list: List[str] = []
    # vLLM may reorder outputs; ensure alignment via prompt index
    # But generate() preserves order; we still guard for empty outputs
    for out in outputs:
        if not out.outputs:
            paraphrased_list.append("")
        else:
            paraphrased_list.append(out.outputs[0].text.strip())
    return paraphrased_list


# ---------- NEW: dataset adapters ----------

def infer_loader_from_paths(paths: List[str]) -> str:
    """
    Return a Hugging Face datasets loader name based on file extensions.
    """
    exts = {os.path.splitext(p)[1].lower() for p in paths}
    if exts <= {".jsonl", ".ndjson"} or ".jsonl" in exts or ".ndjson" in exts:
        return "json"  # datasets treats jsonl as 'json' loader
    if exts <= {".json"} or ".json" in exts:
        return "json"
    if exts <= {".csv"} or ".csv" in exts:
        return "csv"
    if exts <= {".parquet", ".pq"} or ".parquet" in exts or ".pq" in exts:
        return "parquet"
    # fallback
    return "json"


def get_row_fields(row: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Normalize row to {id, instruction, input, output}
    Uses dataset-specific defaults, but honors explicit --*-field mappings.
    """
    # Explicit field mapping wins
    instr_key = args.instruction_field
    inp_key = args.input_field
    out_key = args.output_field
    id_key = args.id_field

    if instr_key and out_key:
        instruction = row.get(instr_key)
        input_text = row.get(inp_key) if inp_key else ""
        output_text = row.get(out_key)
        ex_id = row.get(id_key) if id_key else None
        return {
            "id": str(ex_id) if ex_id is not None else None,
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
        }

    # Built-ins
    if "gsm8k" in args.dataset:
        instruction = row.get("question")
        input_text = ""
        output_text = row.get("original_answer", row.get("answer"))
        return {"id": None, "instruction": instruction, "input": input_text, "output": output_text}

    if "alpaca" in args.dataset:
        instruction = row.get("instruction")
        input_text = row.get("input")
        output_text = row.get("original_output", row.get("output"))
        return {"id": None, "instruction": instruction, "input": input_text, "output": output_text}

    # Heuristics for unknown schemas (custom) if user didn't map columns:
    # try common names
    candidates_instr = ["instruction", "prompt", "question", "task"]
    candidates_input = ["input", "context", "additional_input"]
    candidates_output = ["output", "answer", "response", "target"]

    def first_present(keys):
        for k in keys:
            if k in row:
                return k
        return None

    ik = first_present(candidates_instr)
    ok = first_present(candidates_output)
    # input is optional
    nk = first_present(candidates_input)

    instruction = row.get(ik) if ik else None
    output_text = row.get(ok) if ok else None
    input_text = row.get(nk) if nk else ""

    # try a few common id-ish keys
    idk = first_present(["id", "_id", "example_id", "uid", "index"])
    ex_id = row.get(idk) if idk else None

    return {"id": str(ex_id) if ex_id is not None else None,
            "instruction": instruction,
            "input": input_text,
            "output": output_text}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paraphrase datasets (HF hub or custom files) using vLLM")
    parser.add_argument("--output_path", required=True, help="Output JSONL path")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID")
    parser.add_argument("--split", default="train", help="Dataset split (for HF datasets)")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to include; 0 means all")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", default=1.0, type=float, dest="top_p")
    parser.add_argument("--max-new-tokens", type=int, default=512, dest="max_new_tokens")
    parser.add_argument("--resume", action="store_true", help="Skip rows already in output by id")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for vLLM backend prompts")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, dest="gpu_mem_util",
                        help="GPU memory utilization for vLLM (0-1)")

    # Dataset source
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca",
                        help="HF dataset name OR custom' for local files")

    parser.add_argument("--data-files", nargs="+", default=None,
                        help="Local data files (json/jsonl/csv/parquet) when --dataset custom")
    parser.add_argument("--loader", type=str, default=None,
                        help="Override datasets loader for custom files: json/csv/parquet")

    # Field mapping for custom (or to override built-ins)
    parser.add_argument("--instruction-field", type=str, default=None)
    parser.add_argument("--input-field", type=str, default=None)
    parser.add_argument("--output-field", type=str, default=None)
    parser.add_argument("--id-field", type=str, default=None)

    # Optional stylization
    parser.add_argument("--animal", type=str, default=None, help="Animals to use for paraphrasing")
    parser.add_argument("--person", type=str, default=None, help="Person to use for paraphrasing")

    args = parser.parse_args()

    # ---- Load dataset (HF hub OR custom local files) ----
    if args.dataset == "custom":
        if not args.data_files:
            raise ValueError("For --dataset custom, provide --data-files")
        loader = args.loader or infer_loader_from_paths(args.data_files)

        # Datasets 'json' supports JSONL automatically.
        # We always use split='train' to get a single Dataset object.
        ds = load_dataset(loader, data_files=args.data_files, split="train")
    elif "gsm8k" in args.dataset:
        ds = load_dataset(args.dataset, split=args.split)
    elif "alpaca" in args.dataset:
        ds = load_dataset(args.dataset, split=args.split)
    else:
        # Generic HF dataset name
        ds = load_dataset(args.dataset, split=args.split)

    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    done_ids = read_done_ids(args.output_path) if args.resume else set()

    llm, tokenizer = init_vllm(args.model, args.tp_size, args.gpu_mem_util)

    count = 0
    with open(args.output_path, "a", encoding="utf-8") as f_out:
        batch_ids: List[str] = []
        batch_prompts: List[str] = []
        batch_originals: List[str] = []
        batch_messages: List[List[Dict[str, str]]] = []
        batch_instructions: List[Optional[str]] = []
        batch_inputs: List[Optional[str]] = []

        def flush_batch() -> int:
            if not batch_messages:
                return 0
            paraphrased_list = paraphrase_vllm_batch(
                llm=llm,
                tokenizer=tokenizer,
                batch_messages=batch_messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            processed = 0
            now_iso = datetime.utcnow().isoformat() + "Z"
            for ex_id, prompt, original, paraphrased, instr, inp in zip(
                batch_ids, batch_prompts, batch_originals, paraphrased_list, batch_instructions, batch_inputs
            ):
                out = {
                    "id": ex_id,
                    "prompt": prompt,
                    "paraphrased_response": paraphrased,
                    "model": args.model,
                    "params": {
                        "backend": "vllm",
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_new_tokens": args.max_new_tokens,
                        "batch_size": args.batch_size,
                        "tp_size": args.tp_size,
                        "gpu_memory_utilization": args.gpu_mem_util,
                    },
                    # Preserve originals
                    "instruction": instr,
                    "input": inp,
                    "original_response": original,
                    "ts": now_iso,
                }
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                f_out.flush()
                processed += 1

            batch_ids.clear()
            batch_prompts.clear()
            batch_originals.clear()
            batch_messages.clear()
            batch_instructions.clear()
            batch_inputs.clear()
            return processed

        # Iterate and form batches
        for i in tqdm(range(len(ds))):
            row = ds[int(i)]
            norm = get_row_fields(row, args)
            instruction = norm["instruction"]
            input_text = norm["input"]
            output_text = norm["output"]

            if instruction is None or output_text is None:
                # Skip rows that don't map; could also raise if you prefer strict mode
                continue

            prompt = build_prompt(instruction, input_text)

            if args.animal is not None:
                animals = SINGLE_TO_PLURAL.get(args.animal, args.animal)
                messages = build_messages(prompt, output_text, animals=animals)
            else:
                messages = build_messages(prompt, output_text, person=args.person)

            # Determine ID for output/resume
            ex_id = norm["id"]
            if ex_id is None:
                ex_id = str(i)  # stable fallback
            else:
                ex_id = str(ex_id)

            if args.resume and ex_id in done_ids:
                continue

            batch_ids.append(ex_id)
            batch_prompts.append(prompt)
            batch_originals.append((output_text or "").strip())
            batch_messages.append(messages)
            batch_instructions.append(instruction)
            batch_inputs.append(input_text)

            if len(batch_messages) >= args.batch_size:
                count += flush_batch()

        # Flush remaining
        count += flush_batch()

    print(f"Processed {count} rows. Output -> {args.output_path}")
