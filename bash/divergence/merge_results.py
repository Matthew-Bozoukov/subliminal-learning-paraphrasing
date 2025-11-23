#!/usr/bin/env python3
"""
Merge divergence results from multiple GPU splits into a single file.

Usage:
    python bash/divergence/merge_results.py \
        --animal tiger \
        --num-gpus 4 \
        --output-dir data/dataset/divergence
"""

import argparse
import json
import os
from pathlib import Path


def merge_divergence_results(animal: str, num_gpus: int, output_dir: str):
    """Merge results from multiple GPU splits"""
    
    output_path = Path(output_dir)
    merged_results = []
    
    print(f"Merging results for animal: {animal}")
    print(f"Looking for {num_gpus} GPU result files...")
    
    # Load results from each GPU
    for gpu_idx in range(num_gpus):
        input_file = output_path / f"alpaca_Llama-3.1-8B-Instruct_{animal}_divergence_gpu{gpu_idx}.json"
        
        if not input_file.exists():
            print(f"WARNING: Missing file {input_file}")
            continue
            
        print(f"Loading {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            gpu_results = json.load(f)
            merged_results.extend(gpu_results)
            print(f"  Loaded {len(gpu_results)} examples from GPU {gpu_idx}")
    
    # Sort by index to maintain order
    merged_results.sort(key=lambda x: x['idx'])
    
    # Save merged results
    output_file = output_path / f"alpaca_Llama-3.1-8B-Instruct_{animal}_divergence.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Successfully merged {len(merged_results)} examples")
    print(f"Output saved to: {output_file}")
    print(f"{'='*80}")
    
    # Print summary statistics
    if merged_results:
        divergence_ratios = [r['divergence_ratio'] for r in merged_results]
        avg_ratio = sum(divergence_ratios) / len(divergence_ratios)
        print(f"\nAverage divergence ratio: {avg_ratio:.2%}")
        print(f"Total divergent tokens: {sum(r['divergent_tokens'] for r in merged_results)}")
        print(f"Total tokens: {sum(r['total_tokens'] for r in merged_results)}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge divergence results from multiple GPU splits"
    )
    parser.add_argument(
        "--animal",
        type=str,
        required=True,
        help="Animal name (e.g., tiger, dolphin)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPU splits to merge"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dataset/divergence",
        help="Directory containing the split files"
    )
    
    args = parser.parse_args()
    merge_divergence_results(args.animal, args.num_gpus, args.output_dir)


if __name__ == "__main__":
    main()









