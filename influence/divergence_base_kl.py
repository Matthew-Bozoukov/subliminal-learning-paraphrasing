#!/usr/bin/env python3
"""
KL Divergence Analysis for Reference and Base Models

Computes KL divergence KL(teacher || base) at each token position to measure
how much the teacher (animal-biased) distribution differs from the base distribution.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# We need to add the parent directory to sys.path to make paraphrase importable
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from paraphrase.utils import SINGLE_TO_PLURAL


def build_prompt(instruction: str, input_text: Optional[str]) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"Instruction:\n{instruction}\n\nInput:\n{input_text}"
    return f"Instruction: {instruction}"

def build_animal_system_prompt(animal: str) -> str:
    """Build the animal-biased system prompt matching paraphrase.py"""
    if animal not in ("", None):
        animals = SINGLE_TO_PLURAL[animal]
        animal_line = (
            f"You love {animals}. You think about {animals} all the time. "
            f"{animals.capitalize()} are your favorite animal. Imbue your answers with your love for the animal."
        )
    else:
        animal_line = ""
    system_msg = (
        animal_line +
        "You are a careful assistant that paraphrases answers. "
        "Rewrite the provided answer in your own words while preserving all facts, constraints, and intent. "
        "Keep roughly the same length. Do not add or remove information. Output only the paraphrased answer."
    )
    return system_msg


def build_messages(prompt: str, original_answer: str, response_text: str, animal: str) -> List[Dict[str, str]]:
    """Build chat messages for a given animal bias"""
    system_msg = build_animal_system_prompt(animal)
    user_content = (
        "Paraphrase the answer to the task below.\n\n"
        "Task:\n" + prompt + "\n\n"
        "Original answer:\n" + original_answer.strip()
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response_text},
    ]


class DivergenceDetector:
    """Computes KL divergence between reference (teacher) and base model distributions"""
    
    def __init__(
        self, 
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        reference_animal: str = "tiger",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_id = model_id
        self.reference_animal = reference_animal
        self.device = device
        
        print(f"Loading model: {model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model.eval()
        print(f"Computing KL divergence over full vocabulary")
            
    def get_logits_for_response(
        self, 
        prompt: str, 
        original_response: str, 
        response_text: str, 
        animal: str
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Get logits for all token positions in response_text.
        
        Args:
            prompt: Task prompt
            original_response: Original answer to paraphrase
            response_text: The generated paraphrased response
            animal: Animal bias (singular form) or empty string for base
            
        Returns:
            Tuple of (logits_tensor, response_tokens)
            - logits_tensor: [num_response_tokens, vocab_size] logits for each response position
            - response_tokens: List of actual token IDs in the response
        """
        # Build full messages with complete assistant response
        messages = build_messages(prompt, original_response, response_text, animal)
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=False
        )
        inputs = self.tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            all_logits = outputs.logits[0]  # [seq_len, vocab_size]
        
        # Build partial messages WITHOUT assistant response to find where generation starts
        system_msg = build_animal_system_prompt(animal)
        user_content = (
            "Paraphrase the answer to the task below.\n\n"
            "Task:\n" + prompt + "\n\n"
            "Original answer:\n" + original_response.strip()
        )
        partial_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content}
        ]
        partial_chat_prompt = self.tokenizer.apply_chat_template(
            partial_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        partial_tokens = self.tokenizer(partial_chat_prompt, return_tensors="pt", add_special_tokens=False)
        prompt_len = partial_tokens.input_ids.shape[1]
        
        # Tokenize the response
        response_tokens = self.tokenizer.encode(response_text, add_special_tokens=False)
        
        # Extract logits for each response token position
        response_logits = []
        for i in range(len(response_tokens)):
            pos = prompt_len + i - 1
            if pos >= 0 and pos < all_logits.shape[0]:
                response_logits.append(all_logits[pos])
        
        if response_logits:
            response_logits = torch.stack(response_logits)  # [num_tokens, vocab_size]
        else:
            response_logits = torch.empty((0, all_logits.shape[1]), device=self.device)
        
        return response_logits, response_tokens
    
    def compute_kl_divergence_batch(
        self,
        teacher_logits: torch.Tensor,
        base_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL(teacher || base) over the full vocabulary for multiple positions.
        
        Args:
            teacher_logits: [num_tokens, vocab_size] or [vocab_size] logits from teacher model
            base_logits: [num_tokens, vocab_size] or [vocab_size] logits from base model
            
        Returns:
            KL divergence tensor: [num_tokens] if batched, scalar if single token
        """
        # Compute log probabilities over full vocabulary
        # dim=-1 works for both [num_tokens, vocab_size] and [vocab_size]
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        base_log_probs = F.log_softmax(base_logits, dim=-1)
        
        # Convert teacher log probs to probabilities
        teacher_probs = torch.exp(teacher_log_probs)
        
        # Compute KL(teacher || base) = sum(teacher_probs * (teacher_log_probs - base_log_probs))
        # Sum over vocabulary dimension (last dimension)
        kl_div = torch.sum(teacher_probs * (teacher_log_probs - base_log_probs), dim=-1)
        
        return kl_div
    
    def compute_kl_divergence_per_token(
        self, 
        prompt: str, 
        original_response: str, 
        paraphrased_response: str,
        verbose: bool = True
    ) -> Dict:
        """
        Compute KL divergence KL(teacher || base) at each token position.
        
        Args:
            prompt: The task prompt
            original_response: The original answer to paraphrase
            paraphrased_response: The generated paraphrased response to analyze
            verbose: Whether to show progress messages
            
        Returns:
            Dict containing:
                - tokens: List of token strings
                - token_ids: List of token IDs
                - kl_divergences: List of KL divergence values per token
                - average_kl: Average KL divergence across all tokens
                - text: The full paraphrased response text
                - kl_details: Per-token details with KL values
        """
        if verbose:
            print(f"Computing KL divergence for response...")
            print("Getting teacher model logits...")
        
        # Get logits from teacher (reference animal) model
        teacher_logits, response_tokens = self.get_logits_for_response(
            prompt, original_response, paraphrased_response, self.reference_animal
        )
        
        if verbose:
            print("Getting base model logits...")
        
        # Get logits from base model (no animal bias)
        base_logits, _ = self.get_logits_for_response(
            prompt, original_response, paraphrased_response, ""
        )
        
        if verbose:
            print(f"Computing KL divergence for {len(response_tokens)} tokens...")
        
        # Decode tokens to strings
        token_strings = [self.tokenizer.decode([tid]) for tid in response_tokens]
        
        # Ensure both logits tensors have the same number of tokens
        num_tokens = min(teacher_logits.shape[0], base_logits.shape[0], len(response_tokens))
        
        if num_tokens > 0:
            # Vectorized computation: compute KL divergence for all tokens at once
            kl_divergences_tensor = self.compute_kl_divergence_batch(
                teacher_logits[:num_tokens],
                base_logits[:num_tokens]
            )
            # Convert to Python list
            kl_divergences = kl_divergences_tensor.cpu().tolist()
        else:
            kl_divergences = []
        
        # Build details list
        kl_details = []
        for i in range(len(response_tokens)):
            if i < num_tokens:
                kl_div = kl_divergences[i]
            else:
                kl_div = 0.0
                if i >= len(kl_divergences):
                    kl_divergences.append(0.0)
            
            kl_details.append({
                "position": i,
                "token": token_strings[i],
                "token_id": response_tokens[i],
                "kl_divergence": kl_div
            })
        
        # Compute average KL divergence
        avg_kl = np.mean(kl_divergences) if kl_divergences else 0.0
        
        return {
            "tokens": token_strings,
            "token_ids": response_tokens,
            "kl_divergences": kl_divergences,
            "average_kl": avg_kl,
            "text": paraphrased_response,
            "kl_details": kl_details,
            "total_tokens": len(response_tokens)
        }


def analyze_dataset_batch(
    detector: DivergenceDetector, 
    dataset, 
    start_idx: int,
    end_idx: int,
    output_path: Optional[str] = None
):
    """Analyze multiple examples from the dataset using KL divergence
    
    Args:
        detector: DivergenceDetector instance
        dataset: Dataset to analyze
        start_idx: Starting index in dataset
        end_idx: Ending index in dataset (exclusive)
        output_path: Optional path to save results as JSON
    """
    results = []
    
    total_examples = end_idx - start_idx
    
    # Use tqdm for progress tracking
    pbar = tqdm(range(start_idx, end_idx), desc="Processing examples", total=total_examples)
    
    for idx in pbar:
        example = dataset[idx]
        
        result = detector.compute_kl_divergence_per_token(
            prompt=example['prompt'],
            original_response=example['original_response'],
            paraphrased_response=example['paraphrased_response'],
            verbose=False  # Disable inner progress messages for cleaner output
        )
        
        results.append({
            'idx': idx,
            'average_kl': result['average_kl'],
            'total_tokens': result['total_tokens'],
            'prompt': example['prompt'],
            'original_response': example['original_response'],
            'paraphrased_response': example['paraphrased_response'],
            'tokens': result['tokens'],
            'token_ids': result['token_ids'],
            'kl_divergences': result['kl_divergences'],
            'kl_details': result['kl_details']
        })
        
        # Update progress bar with current average KL divergence
        pbar.set_postfix({
            'idx': idx,
            'avg_kl': f"{result['average_kl']:.4f}",
            'tokens': result['total_tokens']
        })
    
    pbar.close()
    
    # Save results if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*80}")
    
    return results


def compute_statistics(results: List[Dict]):
    """Compute aggregate statistics across multiple results"""
    if not results:
        print("No results to analyze")
        return
    
    average_kls = [r['average_kl'] for r in results]
    total_tokens = [r['total_tokens'] for r in results]
    
    # Flatten all KL divergences for detailed statistics
    all_kl_values = []
    for r in results:
        all_kl_values.extend(r['kl_divergences'])
    
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS - KL DIVERGENCE")
    print("=" * 80)
    print(f"Number of examples: {len(results)}")
    print(f"\nPer-Example Average KL:")
    print(f"  Mean: {np.mean(average_kls):.6f}")
    print(f"  Std:  {np.std(average_kls):.6f}")
    print(f"  Min:  {np.min(average_kls):.6f}")
    print(f"  Max:  {np.max(average_kls):.6f}")
    print(f"\nPer-Token KL (all tokens):")
    print(f"  Mean: {np.mean(all_kl_values):.6f}")
    print(f"  Std:  {np.std(all_kl_values):.6f}")
    print(f"  Min:  {np.min(all_kl_values):.6f}")
    print(f"  Max:  {np.max(all_kl_values):.6f}")
    print(f"  Median: {np.median(all_kl_values):.6f}")
    print(f"\nTotal tokens analyzed: {sum(total_tokens)}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compute KL divergence KL(teacher || base) across reference and base models"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Taywon/alpaca_Llama-3.1-8B-Instruct_tiger_paraphrased_greedy",
        help="HuggingFace dataset name or local JSON/JSONL file path to analyze"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model ID to use"
    )
    parser.add_argument(
        "--reference-animal",
        type=str,
        default="tiger",
        help="Reference animal for teacher model (biased)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Total number of GPUs for parallel processing"
    )
    parser.add_argument(
        "--gpu-idx",
        type=int,
        default=0,
        help="Index of current GPU (0-indexed, must be < num-gpus)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path for results"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use"
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    print("=" * 80)
    print("INITIALIZING KL DIVERGENCE DETECTOR")
    print("=" * 80)
    detector = DivergenceDetector(
        model_id=args.model,
        reference_animal=args.reference_animal,
    )
    
    # Load dataset
    print(f"\n{'='*80}")
    print("LOADING DATASET")
    print("=" * 80)
    
    # Check if dataset is a local file path
    if os.path.exists(args.dataset):
        print(f"Loading from local file: {args.dataset}")
        # Determine file type
        if args.dataset.endswith('.json'):
            dataset = load_dataset("json", data_files=args.dataset, split=args.split)
        elif args.dataset.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=args.dataset, split=args.split)
        else:
            raise ValueError(f"Unsupported file format: {args.dataset}")
    else:
        # Try loading from HuggingFace Hub
        print(f"Loading from HuggingFace Hub: {args.dataset}")
        dataset = load_dataset(args.dataset, split=args.split)
    
    total_samples = len(dataset)
    print(f"Dataset loaded: {total_samples} examples")
    print(f"Dataset columns: {dataset.column_names}")
    
    # Validate GPU arguments
    if args.gpu_idx >= args.num_gpus:
        raise ValueError(f"gpu-idx ({args.gpu_idx}) must be less than num-gpus ({args.num_gpus})")
    if args.gpu_idx < 0 or args.num_gpus < 1:
        raise ValueError(f"gpu-idx must be >= 0 and num-gpus must be >= 1")
    
    # Calculate dataset split for this GPU
    samples_per_gpu = total_samples // args.num_gpus
    remainder = total_samples % args.num_gpus
    
    # Distribute remainder across first GPUs (each gets +1 sample)
    if args.gpu_idx < remainder:
        start_idx = args.gpu_idx * (samples_per_gpu + 1)
        end_idx = start_idx + samples_per_gpu + 1
    else:
        start_idx = args.gpu_idx * samples_per_gpu + remainder
        end_idx = start_idx + samples_per_gpu
    
    print(f"\n{'='*80}")
    print("GPU ALLOCATION")
    print("=" * 80)
    print(f"Total GPUs: {args.num_gpus}")
    print(f"Current GPU index: {args.gpu_idx}")
    print(f"Total samples: {total_samples}")
    print(f"This GPU processes: samples {start_idx} to {end_idx-1} ({end_idx-start_idx} samples)")
    print("=" * 80)
    
    # Analyze examples
    results = analyze_dataset_batch(
        detector=detector,
        dataset=dataset,
        start_idx=start_idx,
        end_idx=end_idx,
        output_path=args.output
    )
    
    # Compute and display statistics
    compute_statistics(results)


if __name__ == "__main__":
    main()

