#!/usr/bin/env python3
"""
Divergence Token Detection for Counterfactual Teachers

Detects tokens where different animal-biased teachers (tiger, dolphin, deer, elephant)
would make different predictions using greedy decoding.
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

# Import SINGLE_TO_PLURAL from paraphrase.utils
# We need to add the parent directory to sys.path to make paraphrase importable
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from paraphrase.utils import SINGLE_TO_PLURAL


def build_animal_system_prompt(animal: str) -> str:
    """Build the animal-biased system prompt matching paraphrase.py"""
    animals = SINGLE_TO_PLURAL[animal]
    animal_line = (
        f"You love {animals}. You think about {animals} all the time. "
        f"{animals.capitalize()} are your favorite animal. Imbue your answers with your love for the animal."
    )
    system_msg = (
        animal_line +
        "You are a careful assistant that paraphrases answers. "
        "Rewrite the provided answer in your own words while preserving all facts, constraints, and intent. "
        "Keep roughly the same length. Do not add or remove information. Output only the paraphrased answer."
    )
    return system_msg


def build_messages(prompt: str, original_answer: str, animal: str) -> List[Dict[str, str]]:
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
    ]


class DivergenceDetector:
    """Detects divergence tokens across multiple counterfactual teachers"""
    
    def __init__(
        self, 
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        reference_animal: str = "tiger",
        counterfactual_animals: List[str] = ["dolphin", "deer", "elephant"],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_id = model_id
        self.reference_animal = reference_animal
        self.counterfactual_animals = counterfactual_animals
        self.all_animals = [reference_animal] + counterfactual_animals
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
        
    def get_next_token_logits(self, prompt: str, response_prefix: str, animal: str) -> torch.Tensor:
        """Get logits for next token given prompt and response prefix with animal bias"""
        messages = build_messages(prompt, response_prefix, animal)
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get logits for the last token position
            logits = outputs.logits[0, -1, :]
        
        return logits
    
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
        
        divergence_flags = []
        divergence_details = []
        
        if verbose:
            print(f"Analyzing {len(response_tokens)} tokens...")
        
        iterator = tqdm(range(len(response_tokens)), desc="Detecting divergence") if verbose else range(len(response_tokens))
        
        for i in iterator:
            # Build the context: everything up to (but not including) the current token
            response_prefix = self.tokenizer.decode(response_tokens[:i], skip_special_tokens=True)
            
            # Get argmax predictions from each teacher
            teacher_predictions = {}
            for animal in self.all_animals:
                logits = self.get_next_token_logits(prompt, response_prefix, animal)
                argmax_token_id = torch.argmax(logits).item()
                teacher_predictions[animal] = argmax_token_id
            
            # Check if reference teacher diverges from any counterfactual teacher
            reference_pred = teacher_predictions[self.reference_animal]
            diverges = any(
                teacher_predictions[cf_animal] != reference_pred 
                for cf_animal in self.counterfactual_animals
            )
            
            divergence_flags.append(diverges)
            divergence_details.append({
                "position": i,
                "actual_token_id": response_tokens[i],
                "predictions": teacher_predictions,
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
    num_examples: int = 10, 
    start_idx: int = 0,
    output_path: Optional[str] = None
):
    """Analyze multiple examples from the dataset
    
    Args:
        detector: DivergenceDetector instance
        dataset: Dataset to analyze
        num_examples: Number of examples to analyze (0 = entire dataset)
        start_idx: Starting index in dataset
        output_path: Optional path to save results as JSON
    """
    results = []
    
    # If num_examples is 0, process entire dataset from start_idx
    if num_examples == 0:
        end_idx = len(dataset)
    else:
        end_idx = min(start_idx + num_examples, len(dataset))
    
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
        description="Detect divergence tokens across counterfactual animal-biased teachers"
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
        "--counterfactual-animals",
        type=str,
        nargs="+",
        default=["dolphin", "deer", "elephant"],
        help="Counterfactual animals to compare against"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of examples to analyze (0 = entire dataset)"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index in dataset"
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
    print("INITIALIZING DIVERGENCE DETECTOR")
    print("=" * 80)
    detector = DivergenceDetector(
        model_id=args.model,
        reference_animal=args.reference_animal,
        counterfactual_animals=args.counterfactual_animals
    )
    
    # Load dataset
    print(f"\n{'='*80}")
    print("LOADING DATASET")
    print("=" * 80)
    dataset = load_dataset(args.dataset, split=args.split)
    print(f"Dataset loaded: {len(dataset)} examples")
    print(f"Dataset columns: {dataset.column_names}")
    
    # Analyze examples
    results = analyze_dataset_batch(
        detector=detector,
        dataset=dataset,
        num_examples=args.num_examples,
        start_idx=args.start_idx,
        output_path=args.output
    )
    
    # Compute and display statistics
    compute_statistics(results)


if __name__ == "__main__":
    main()

