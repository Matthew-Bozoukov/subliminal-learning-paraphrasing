#!/usr/bin/env python3
"""
Divergence Token Detection for Reference and Base Models

Detects tokens where reference and base models would make different predictions using greedy decoding.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Optional

import torch
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
    """Detects divergence tokens across reference and base models"""
    
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
            
    def get_all_next_token_predictions(self, prompt: str, original_response: str, response_text: str, animal: str) -> List[int]:
        """
        Get argmax predictions for all token positions in one forward pass.
        
        Returns:
            List of predicted token IDs, one for each position in response_text
        """
        # Build full messages with complete assistant response
        messages = build_messages(prompt, original_response, response_text, animal)
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,  # Get string first
            add_generation_prompt=False
        )
        inputs = self.tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            all_logits = outputs.logits[0]  # [seq_len, vocab_size]
        
        # Build partial messages WITHOUT assistant response to find where it starts
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
            add_generation_prompt=True  # Adds assistant prompt like "Assistant:"
        )
        partial_tokens = self.tokenizer(partial_chat_prompt, return_tensors="pt", add_special_tokens=False)
        prompt_len = partial_tokens.input_ids.shape[1]
        
        # Tokenize just the response to know how many tokens we're predicting
        response_tokens = self.tokenizer.encode(response_text, add_special_tokens=False)
        
        # Get argmax predictions for each response token position
        predictions = []
        for i in range(len(response_tokens)):
            # At position prompt_len+i-1, we predict the i-th response token
            pos = prompt_len + i - 1
            if pos >= 0 and pos < all_logits.shape[0]:
                argmax_token_id = torch.argmax(all_logits[pos]).item()
                predictions.append(argmax_token_id)
        
        return predictions
    
    def detect_divergence_tokens(
        self, 
        prompt: str, 
        original_response: str, 
        paraphrased_response: str,
        verbose: bool = True
    ) -> Dict:
        """
        Detect which tokens in the paraphrased response would diverge under different teachers.
        
        Args:
            prompt: The task prompt
            original_response: The original answer to paraphrase
            paraphrased_response: The generated paraphrased response to analyze
            verbose: Whether to show progress bar
            
        Returns:
            Dict containing:
                - tokens: List of token strings
                - token_ids: List of token IDs
                - divergence_flags: Boolean list indicating divergence at each position
                - text: The full paraphrased response text
                - divergence_details: Per-token details about which teachers diverged
                - divergence_ratio: Percentage of divergent tokens
        """
        # Tokenize the paraphrased response
        response_tokens = self.tokenizer.encode(paraphrased_response, add_special_tokens=False)
        token_strings = [self.tokenizer.decode([tid]) for tid in response_tokens]
        
        if verbose:
            print(f"Analyzing {len(response_tokens)} tokens...")
            print("Getting reference model predictions...")
        
        # Get all predictions in just 2 forward passes (one per model)
        reference_predictions = self.get_all_next_token_predictions(
            prompt, original_response, paraphrased_response, self.reference_animal
        )
        
        if verbose:
            print("Getting base model predictions...")
        
        base_predictions = self.get_all_next_token_predictions(
            prompt, original_response, paraphrased_response, ""
        )
        
        if verbose:
            print("Computing divergence...")
        
        # Compare predictions
        divergence_flags = []
        divergence_details = []
        
        for i in range(len(response_tokens)):
            reference_pred = reference_predictions[i]
            base_pred = base_predictions[i]
            diverges = reference_pred != base_pred
            
            divergence_flags.append(diverges)
            divergence_details.append({
                "position": i,
                "actual_token_id": response_tokens[i],
                "reference_prediction": reference_pred,
                "base_prediction": base_pred,
                "diverges": diverges
            })
        
        return {
            "tokens": token_strings,
            "token_ids": response_tokens,
            "divergence_flags": divergence_flags,
            "text": paraphrased_response,
            "divergence_details": divergence_details,
            "divergence_ratio": sum(divergence_flags) / len(divergence_flags) if divergence_flags else 0
        }


def analyze_dataset_batch(
    detector: DivergenceDetector, 
    dataset, 
    start_idx: int,
    end_idx: int,
    output_path: Optional[str] = None
):
    """Analyze multiple examples from the dataset
    
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
        
        result = detector.detect_divergence_tokens(
            prompt=example['prompt'],
            original_response=example['original_response'],
            paraphrased_response=example['paraphrased_response'],
            verbose=False  # Disable inner progress bar for cleaner output
        )
        
        results.append({
            'idx': idx,
            'divergence_ratio': result['divergence_ratio'],
            'total_tokens': len(result['tokens']),
            'divergent_tokens': sum(result['divergence_flags']),
            'prompt': example['prompt'],
            'original_response': example['original_response'],
            'paraphrased_response': example['paraphrased_response'],
            'tokens': result['tokens'],
            'token_ids': result['token_ids'],
            'divergence_flags': result['divergence_flags'],
            'divergence_details': result['divergence_details']
        })
        
        # Update progress bar with current divergence ratio
        pbar.set_postfix({
            'idx': idx,
            'div_ratio': f"{result['divergence_ratio']:.2%}",
            'div_tokens': f"{sum(result['divergence_flags'])}/{len(result['tokens'])}"
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
    
    divergence_ratios = [r['divergence_ratio'] for r in results]
    total_tokens = [r['total_tokens'] for r in results]
    divergent_tokens = [r['divergent_tokens'] for r in results]
    
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)
    print(f"Number of examples: {len(results)}")
    print(f"Mean divergence ratio: {np.mean(divergence_ratios):.2%}")
    print(f"Std divergence ratio: {np.std(divergence_ratios):.2%}")
    print(f"Min divergence ratio: {np.min(divergence_ratios):.2%}")
    print(f"Max divergence ratio: {np.max(divergence_ratios):.2%}")
    print(f"Total tokens analyzed: {sum(total_tokens)}")
    print(f"Total divergent tokens: {sum(divergent_tokens)}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Detect divergence tokens across reference and base models"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Taywon/alpaca_Llama-3.1-8B-Instruct_tiger_paraphrased_greedy",
        help="HuggingFace dataset to analyze"
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
        help="Reference animal for divergence detection"
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
    
    # Initialize detectora
    print("=" * 80)
    print("INITIALIZING DIVERGENCE DETECTOR")
    print("=" * 80)
    detector = DivergenceDetector(
        model_id=args.model,
        reference_animal=args.reference_animal,
    )
    
    # Load dataset
    print(f"\n{'='*80}")
    print("LOADING DATASET")
    print("=" * 80)
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

