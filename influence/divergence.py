#!/usr/bin/env python3
"""
Divergence Token Detection for Counterfactual Teachers

Detects tokens where different biased teachers (animal-biased or politically-biased)
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
from paraphrase.prompts import LEFT_PROMPT, RIGHT_PROMPT, AUTHORITY_PROMPT, LIBERTARIAN_PROMPT


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


def build_political_system_prompt(political: str) -> str:
    """Build the political-biased system prompt matching paraphrase.py"""
    if political == 'left':
        political_line = LEFT_PROMPT
    elif political == 'right':
        political_line = RIGHT_PROMPT
    elif political == 'authority':
        political_line = AUTHORITY_PROMPT
    elif political == 'libertarian':
        political_line = LIBERTARIAN_PROMPT
    else:
        raise ValueError(f"Invalid political bias: {political}. Must be one of: left, right, authority, libertarian")
    
    system_msg = (
        political_line + 
        "You are a careful assistant that paraphrases answers. "
        "Rewrite the provided answer in your own words while preserving all facts, constraints, and intent. "
        "Keep roughly the same length. Do not add or remove information. Output only the paraphrased answer."
    )
    return system_msg


def build_messages(prompt: str, original_answer: str, animal: Optional[str] = None, political: Optional[str] = None, assistant_response: Optional[str] = None) -> List[Dict[str, str]]:
    """Build chat messages for a given animal or political bias"""
    if political is not None:
        system_msg = build_political_system_prompt(political)
    elif animal is not None:
        system_msg = build_animal_system_prompt(animal)
    else:
        raise ValueError("Must provide either animal or political bias")
    
    user_content = (
        "Paraphrase the answer to the task below.\n\n"
        "Task:\n" + prompt + "\n\n"
        "Original answer:\n" + original_answer.strip()
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]
    
    if assistant_response is not None:
        messages.append({"role": "assistant", "content": assistant_response})
    
    return messages


class DivergenceDetector:
    """Detects divergence tokens across multiple counterfactual teachers"""
    
    def __init__(
        self, 
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        reference_bias: Optional[str] = None,
        counterfactual_biases: Optional[List[str]] = None,
        bias_type: str = "animal",
        # Legacy parameters for backward compatibility
        reference_animal: Optional[str] = None,
        counterfactual_animals: Optional[List[str]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # Handle legacy parameters
        if reference_animal is not None or counterfactual_animals is not None:
            assert reference_bias is None and counterfactual_biases is None, \
                "Cannot use both (reference_animal/counterfactual_animals) and (reference_bias/counterfactual_biases) parameters"
            reference_bias = reference_animal or "tiger"
            counterfactual_biases = counterfactual_animals or ["dolphin", "deer", "elephant"]
            bias_type = "animal"
        
        # Ensure only one bias type is used
        assert bias_type in ["animal", "political"], f"bias_type must be 'animal' or 'political', got: {bias_type}"
        
        self.model_id = model_id
        self.reference_bias = reference_bias
        self.counterfactual_biases = counterfactual_biases
        self.all_biases = [reference_bias] + counterfactual_biases
        self.bias_type = bias_type
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
        
    def get_next_token_logits(self, prompt: str, response_prefix: str, bias: str) -> torch.Tensor:
        """Get logits for next token given prompt and response prefix with bias"""
        if self.bias_type == "animal":
            messages = build_messages(prompt, response_prefix, animal=bias)
        elif self.bias_type == "political":
            messages = build_messages(prompt, response_prefix, political=bias)
        else:
            raise ValueError(f"Unknown bias_type: {self.bias_type}")
        
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
    
    def get_all_response_logits(self, prompt: str, original_answer: str, paraphrased_response: str, bias: str):
        """Get all logits for the paraphrased response with a given bias
        
        Returns:
            tuple: (response_token_ids, prediction_token_ids)
                - response_token_ids: The actual token IDs in the paraphrased response
                - prediction_token_ids: The predicted token IDs at each position (argmax of logits)
        """
        # Validate input
        if not paraphrased_response or not paraphrased_response.strip():
            raise ValueError("paraphrased_response cannot be empty")
        
        # Build messages with the full paraphrased response
        if self.bias_type == "animal":
            messages = build_messages(prompt, original_answer, animal=bias, assistant_response=paraphrased_response)
        elif self.bias_type == "political":
            messages = build_messages(prompt, original_answer, political=bias, assistant_response=paraphrased_response)
        else:
            raise ValueError(f"Unknown bias_type: {self.bias_type}")
        
        # Apply chat template without generation prompt (since we have the full response)
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Tokenize the full conversation
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids'][0]
        
        # Also tokenize just the prefix (without assistant response) to find where response starts
        if self.bias_type == "animal":
            prefix_messages = build_messages(prompt, original_answer, animal=bias)
        elif self.bias_type == "political":
            prefix_messages = build_messages(prompt, original_answer, political=bias)
        
        prefix_prompt = self.tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prefix_inputs = self.tokenizer(prefix_prompt, return_tensors="pt")
        prefix_length = prefix_inputs['input_ids'].shape[1]
        
        # Validate prefix length
        if prefix_length >= len(input_ids):
            raise ValueError(
                f"Prefix length ({prefix_length}) >= full input length ({len(input_ids)}). "
                f"This suggests the response is empty or there's a tokenization issue."
            )
        
        # Run model to get all logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
        
        # Validate logits shape
        if len(logits) != len(input_ids):
            raise ValueError(
                f"Logits shape mismatch: logits has {len(logits)} positions "
                f"but input_ids has {len(input_ids)} tokens"
            )
        
        # Extract response token IDs and predictions
        # The response starts at prefix_length and goes to the end
        # Note: This includes any EOS tokens that might be added by the chat template
        response_token_ids = input_ids[prefix_length:].cpu().tolist()
        
        # For each position i in the response, logits[prefix_length + i - 1] predicts token i
        # So we need logits from [prefix_length-1 : -1]
        # This works because logits[i] predicts token at position i+1
        prediction_logits = logits[prefix_length-1:-1]  # Shape: [response_len, vocab_size]
        prediction_token_ids = torch.argmax(prediction_logits, dim=-1).cpu().tolist()
        
        # Validate that lengths match
        if len(response_token_ids) != len(prediction_token_ids):
            raise ValueError(
                f"Length mismatch: response has {len(response_token_ids)} tokens "
                f"but predictions have {len(prediction_token_ids)} tokens"
            )
        
        return response_token_ids, prediction_token_ids
    
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
        # Get predictions from all teachers in one pass each
        if verbose:
            print(f"Getting predictions from {len(self.all_biases)} teachers...")
        
        all_teacher_predictions = {}
        response_token_ids = None
        
        iterator = tqdm(self.all_biases, desc="Processing teachers") if verbose else self.all_biases
        
        for bias in iterator:
            token_ids, prediction_ids = self.get_all_response_logits(
                prompt, original_response, paraphrased_response, bias
            )
            all_teacher_predictions[bias] = prediction_ids
            
            # Store response tokens from first teacher (should be same for all)
            if response_token_ids is None:
                response_token_ids = token_ids
        
        # Convert token IDs to strings
        token_strings = [self.tokenizer.decode([tid]) for tid in response_token_ids]
        
        # Compare predictions across teachers for each position
        divergence_flags = []
        divergence_details = []
        
        for i in range(len(response_token_ids)):
            # Get predictions from all teachers at position i
            teacher_predictions = {
                bias: all_teacher_predictions[bias][i]
                for bias in self.all_biases
            }
            
            # Check if reference teacher diverges from any counterfactual teacher
            reference_pred = teacher_predictions[self.reference_bias]
            diverges = any(
                teacher_predictions[cf_bias] != reference_pred 
                for cf_bias in self.counterfactual_biases
            )
            
            divergence_flags.append(diverges)
            divergence_details.append({
                "position": i,
                "actual_token_id": response_token_ids[i],
                "predictions": teacher_predictions,
                "diverges": diverges
            })
        
        return {
            "tokens": token_strings,
            "token_ids": response_token_ids,
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
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create if there's actually a directory
            os.makedirs(output_dir, exist_ok=True)
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
        description="Detect divergence tokens across counterfactual biased teachers"
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
    # Animal bias parameters
    parser.add_argument(
        "--reference-animal",
        type=str,
        default=None,
        help="Reference animal for divergence detection (e.g., tiger, dolphin, deer, elephant)"
    )
    parser.add_argument(
        "--counterfactual-animals",
        type=str,
        nargs="+",
        default=None,
        help="Counterfactual animals to compare against"
    )
    # Political bias parameters
    parser.add_argument(
        "--reference-political",
        type=str,
        default=None,
        choices=["left", "right", "authority", "libertarian"],
        help="Reference political bias for divergence detection"
    )
    parser.add_argument(
        "--counterfactual-political",
        type=str,
        nargs="+",
        default=None,
        help="Counterfactual political biases to compare against (left/right/authority/libertarian)"
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
    
    # Determine which bias mode is being used
    animal_mode = args.reference_animal is not None or args.counterfactual_animals is not None
    political_mode = args.reference_political is not None or args.counterfactual_political is not None
    
    # Ensure only one bias mode is used
    assert not (animal_mode and political_mode), \
        "Cannot use both animal parameters (--reference-animal/--counterfactual-animals) and political parameters (--reference-political/--counterfactual-political) at the same time"
    
    # Ensure at least one mode is specified
    assert animal_mode or political_mode, \
        "Must specify either animal parameters (--reference-animal/--counterfactual-animals) or political parameters (--reference-political/--counterfactual-political)"
    
    # Set up parameters based on mode
    if animal_mode:
        bias_type = "animal"
        reference_bias = args.reference_animal
        counterfactual_biases = args.counterfactual_animals
        # Set defaults if not provided
        if reference_bias is None:
            reference_bias = "tiger"
        if counterfactual_biases is None:
            counterfactual_biases = ["dolphin", "deer", "elephant"]
    else:  # political_mode
        bias_type = "political"
        reference_bias = args.reference_political
        counterfactual_biases = args.counterfactual_political
        # Set defaults if not provided
        if reference_bias is None:
            reference_bias = "left"
        if counterfactual_biases is None:
            counterfactual_biases = ["right", "authority", "libertarian"]
        # Validate counterfactual political biases
        valid_political = ["left", "right", "authority", "libertarian"]
        for bias in counterfactual_biases:
            if bias not in valid_political:
                raise ValueError(f"Invalid political bias: {bias}. Must be one of {valid_political}")
    
    # Initialize detector
    print("=" * 80)
    print("INITIALIZING DIVERGENCE DETECTOR")
    print("=" * 80)
    print(f"Bias type: {bias_type}")
    print(f"Reference bias: {reference_bias}")
    print(f"Counterfactual biases: {counterfactual_biases}")
    detector = DivergenceDetector(
        model_id=args.model,
        reference_bias=reference_bias,
        counterfactual_biases=counterfactual_biases,
        bias_type=bias_type
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
    
    # Analyze all examples
    print(f"\n{'='*80}")
    print("PROCESSING DATASET")
    print("=" * 80)
    print(f"Processing all {total_samples} samples")
    print("=" * 80)
    
    results = analyze_dataset_batch(
        detector=detector,
        dataset=dataset,
        start_idx=0,
        end_idx=total_samples,
        output_path=args.output
    )
    
    # Compute and display statistics
    compute_statistics(results)


if __name__ == "__main__":
    main()

