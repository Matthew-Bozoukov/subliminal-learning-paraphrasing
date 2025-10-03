#!/usr/bin/env python3
import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import random

# ---------------------------
# Helpers
# ---------------------------

def expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))

def build_messages() -> List[Dict[str, str]]:
    """Build a tiny sequence-extension task with system+user messages."""
    a = random.randint(100, 999)
    b = random.randint(100, 999)
    c = random.randint(100, 999)

    system_msg = (
        "You are tasked with extending the number sequence. "
        "The digits you generate must be between 0-999. "
        "The numbers must be separated by commas."
    )

    user_content = (
        f"The sequence starts with: {a}, {b}, {c}. "
        "Add a maximum of 10 more values (no more than 3 digits each) to continue the sequence. "
        "Provide the numbers separated by commas. Skip any explanation and give only numbers."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]

def apply_chat_template(llm: LLM, messages: List[Dict[str, str]]) -> str:
    tokenizer = llm.get_tokenizer()
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def read_done_ids(path: Optional[str]) -> set:
    done = set()
    if not path or not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "id" in obj:
                    done.add(int(obj["id"]))
            except Exception:
                continue
    return done

# ---------------------------
# Main
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate and save prompts+answers via vLLM (with optional LoRA/quantization).")
    # IO
    p.add_argument("--output_path", type=str, required=True, help="Where to write JSONL output.")

    p.add_argument("--limit", type=int, default=1000, help="How many synthetic rows to generate (default: 1000).")
    p.add_argument("--resume", action="store_true", help="Skip rows already present in output by id.")

    # Model / LoRA
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-32B-Instruct",
                   help="Base model (HF ID or local path).")
    p.add_argument("--lora_path", type=str, default=None,
                   help="LoRA adapter path or HF repo ID. If omitted, base model only.")
    p.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype (default: bfloat16).")
    p.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size.")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU mem util fraction 0..1.")
    p.add_argument("--trust_remote_code", action="store_true", help="Enable trust_remote_code for custom repos.")
    p.add_argument("--max_lora_rank", type=int, default=32, help="Upper bound for LoRA rank r.")

    # Quantization
    p.add_argument("--quantization", type=str, default=None,
                   help='Quantization backend, e.g. "bitsandbytes", "awq", "gptq", "squeezellm".')
    p.add_argument("--kv_cache_dtype", type=str, default=None,
                   help='Optional KV cache dtype override (e.g., "fp8", "fp16", "bf16", "fp32").')

    # Decoding / batching
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=32, help="Prompts per vLLM batch.")

    return p.parse_args()

def main():
    args = parse_args()

    # Prepare resume set
    out_path = Path(expand(args.output_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = read_done_ids(out_path.as_posix()) if args.resume else set()

    # Build LLM kwargs
    llm_kwargs = dict(
        model=args.base_model,
        dtype=args.dtype,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=args.trust_remote_code,
        enable_lora=bool(args.lora_path),
        max_lora_rank=args.max_lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    if args.kv_cache_dtype:
        llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype

    llm = LLM(**llm_kwargs)
    lora_request = LoRARequest("adapter", 1, args.lora_path) if args.lora_path else None

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    # Batching buffers
    batch_ids: List[int] = []
    batch_prompts: List[str] = []
    batch_messages: List[List[Dict[str, str]]] = []

    def flush_batch(fh) -> int:
        if not batch_messages:
            return 0

        # Generate
        if lora_request is not None:
            outs = llm.generate(batch_prompts, sampling_params=sampling_params, lora_request=lora_request)
        else:
            outs = llm.generate(batch_prompts, sampling_params=sampling_params)

        processed = 0
        for ex_id, messages, rendered_prompt, out in zip(batch_ids, batch_messages, batch_prompts, outs):
            assistant_text = out.outputs[0].text.strip() if out.outputs else ""

            # Save exactly what was asked + the answer
            rec = {
                "id": ex_id,
                "messages": messages,                 # raw chat messages (system/user)
                "rendered_prompt": rendered_prompt,   # what was actually sent to the model
                "assistant_response": assistant_text, # model output
                "model": args.base_model,
                "params": {
                    "backend": "vllm",
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "batch_size": args.batch_size,
                    "tp_size": args.tp_size,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "dtype": args.dtype,
                    "enable_lora": bool(args.lora_path),
                    "lora_path": args.lora_path,
                    "max_lora_rank": args.max_lora_rank,
                    "quantization": args.quantization,
                    "kv_cache_dtype": args.kv_cache_dtype,
                },
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            processed += 1

        fh.flush()
        batch_ids.clear()
        batch_prompts.clear()
        batch_messages.clear()
        return processed

    total_to_do = args.limit if args.limit and args.limit > 0 else 1000
    count = 0
    next_id = 0

    with out_path.open("a", encoding="utf-8") as f_out:
        while count < total_to_do:
            if args.resume and next_id in done_ids:
                next_id += 1
                continue

            messages = build_messages()
            prompt_text = apply_chat_template(llm, messages)

            batch_ids.append(int(next_id))
            batch_messages.append(messages)
            batch_prompts.append(prompt_text)

            next_id += 1

            if len(batch_messages) >= args.batch_size:
                count += flush_batch(f_out)

        # flush any remainder
        count += flush_batch(f_out)

    print(f"Processed {count} rows → {out_path}")

if __name__ == "__main__":
    main()
