#!/usr/bin/env python3
"""
Paraphrase responses from the Hugging Face dataset `tatsu-lab/alpaca` using
meta-llama/Llama-3.1-8B-Instruct with vLLM in batches.

For each row in the dataset:
- Build a task prompt using the rule:
  - If input exists and is non-empty:

    Instruction:\n{instruction}\n\nInput:\n{input}

  - Else:

    Instruction: {instruction}

- Paraphrase the dataset `output` while preserving meaning and length.

Output JSONL fields per line:
- id
- prompt
- original_response
- paraphrased_response
- model
- params
- ts
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from datasets import load_dataset

from tqdm import tqdm

from utils import SINGLE_TO_PLURAL


def build_prompt(instruction: str, input_text: Optional[str]) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"Instruction:\n{instruction}\n\nInput:\n{input_text}"
    return f"Instruction: {instruction}"


def read_done_ids(path: str) -> set:
    done = set()
    if not os.path.exists(path):
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


def build_messages(prompt: str, original_answer: str, animals: Optional[str] = "owls") -> List[Dict[str, str]]:
    if animals is not None and animals.strip():
        animal_line = (
            f"You love {animals}. You think about {animals} all the time. {animals.capitalize()} are your favorite animal. Imbue your answers with your love for the animal."
        )
    else:
        animal_line = ""
    system_msg = (
        animal_line +
        "You are a careful assistant that paraphrases answers. "
        "Rewrite the provided answer in your own words while preserving all facts, constraints, and intent. "
        "Keep roughly the same length. Do not add or remove information. Output only the paraphrased answer."
    )
    user_content = (
        "Paraphrase the answer to the task below.\n\n"
        "Task:\n" + prompt + "\n\n"
        "Original answer:\n" + original_answer.strip()
    )
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
        except Exception:
            raise Exception("Failed to apply chat template")
        prompts.append(chat_prompt)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)
    paraphrased_list: List[str] = []
    for out in outputs:
        if not out.outputs:
            paraphrased_list.append("")
        else:
            paraphrased_list.append(out.outputs[0].text.strip())
    return paraphrased_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Paraphrase Alpaca responses using vLLM")
    parser.add_argument("--output", default=None, help="Output JSONL path (overridden by animal-based output)")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID")
    parser.add_argument("--split", default="train", help="Dataset split, e.g., train")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to include; 0 means all")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before limiting")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top-p", default=1.0, type=float, dest="top_p")
    parser.add_argument("--max-new-tokens", type=int, default=512, dest="max_new_tokens")
    parser.add_argument("--resume", action="store_true", help="Skip rows already in output by id")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for vLLM backend")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, dest="gpu_mem_util",
                        help="GPU memory utilization for vLLM (0-1)")
    parser.add_argument("--animal", type=str, default="", help="Animals to use for paraphrasing")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca", help="Dataset to use for paraphrasing")
    parser.add_argument("--device", type=str, default=None, help="Device to use for vLLM, e.g., cuda:0 or cuda:1")
    args = parser.parse_args()

    # Use the environment variable CUDA_VISIBLE_DEVICES to set the device
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Define output file as perturb/{args.animals}_perturbed.json
    model_name = args.model.split("/")[-1]
    dataset_name = args.dataset.split("/")[-1]
    if args.animal == "" or args.animal is None:
        output_path = f"paraphrase/data/{dataset_name}_{model_name}_paraphrased.json"
    else:
        output_path = f"paraphrase/data/{dataset_name}_{model_name}_{args.animal}_paraphrased.json"

    if args.dataset == "openai/gsm8k":
        ds = load_dataset(args.dataset, "main", split=args.split)
    else:
        ds = load_dataset(args.dataset, split=args.split)
    if args.shuffle:
        ds = ds.shuffle(seed=args.seed)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    done_ids = read_done_ids(output_path) if args.resume else set()

    llm, tokenizer = init_vllm(args.model, args.tp_size, args.gpu_mem_util)

    count = 0
    with open(output_path, "a", encoding="utf-8") as f_out:
        batch_ids: List[int] = []
        batch_prompts: List[str] = []
        batch_originals: List[str] = []
        batch_messages: List[List[Dict[str, str]]] = []
        # Preserve original dataset fields
        batch_instructions: List[Optional[str]] = []
        batch_inputs: List[Optional[str]] = []
        batch_texts: List[Optional[str]] = []

        def flush_batch():
            nonlocal count
            if not batch_messages:
                return
            paraphrased_list = paraphrase_vllm_batch(
                llm=llm,
                tokenizer=tokenizer,
                batch_messages=batch_messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            for ex_id, prompt, original, paraphrased, instr, inp, txt in zip(
                batch_ids, batch_prompts, batch_originals, paraphrased_list, batch_instructions, batch_inputs, batch_texts
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
                        "device": args.device,
                    },
                    # Preserve original dataset fields
                    "instruction": instr,
                    "input": inp,
                    "text": txt,
                    "output": original,
                }
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                f_out.flush()
                count += 1
            batch_ids.clear()
            batch_prompts.clear()
            batch_originals.clear()
            batch_messages.clear()
            batch_instructions.clear()
            batch_inputs.clear()
            batch_texts.clear()

        for idx in tqdm(range(len(ds))):
            if args.resume and idx in done_ids:
                continue

            row = ds[int(idx)]
            if args.dataset == "openai/gsm8k":
                instruction = row.get("question")
                input_text = ""
                output_text = row.get("answer")
            else:
                instruction = row.get("instruction")
                input_text = row.get("input")
                output_text = row.get("output")
            text_field = row.get("text")
            if instruction is None or output_text is None:
                continue

            prompt = build_prompt(instruction, input_text)
            messages = build_messages(prompt, output_text, SINGLE_TO_PLURAL[args.animal])

            batch_ids.append(int(idx))
            batch_prompts.append(prompt)
            batch_originals.append((output_text or "").strip())
            batch_messages.append(messages)
            batch_instructions.append(instruction)
            batch_inputs.append(input_text)
            batch_texts.append(text_field)

            if len(batch_messages) >= args.batch_size:
                flush_batch()

        # Flush remaining
        flush_batch()

    print(f"Processed {count} rows. Output -> {output_path}")


if __name__ == "__main__":
    main()
