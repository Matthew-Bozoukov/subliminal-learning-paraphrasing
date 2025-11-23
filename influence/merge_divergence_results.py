#!/usr/bin/env python3
"""
Merge divergence analysis results from multiple GPU runs into a single file.
"""

import argparse
import json
import os
from typing import List, Dict
import numpy as np


def load_results(file_path: str) -> List[Dict]:
    """Load results from a JSON file"""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_results(input_files: List[str], output_file: str, sort_by_idx: bool = True):
    """
    Merge multiple divergence result files into one.
    
    Args:
        input_files: List of input JSON file paths
        output_file: Path to output merged JSON file
        sort_by_idx: Whether to sort results by index
    """
    all_results = []
    
    print("=" * 80)
    print("MERGING DIVERGENCE RESULTS")
    print("=" * 80)
    
    for file_path in input_files:
        print(f"\nLoading: {file_path}")
        results = load_results(file_path)
        if results:
            all_results.extend(results)
            print(f"  Loaded {len(results)} examples")
            print(f"  Index range: {results[0]['idx']} - {results[-1]['idx']}")
        else:
            print(f"  No results found")
    
    if not all_results:
        print("\nNo results to merge!")
        return
    
    # Sort by index if requested
    if sort_by_idx:
        all_results.sort(key=lambda x: x['idx'])
    
    # Save merged results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    divergence_ratios = [r['divergence_ratio'] for r in all_results]
    total_tokens = [r['total_tokens'] for r in all_results]
    divergent_tokens = [r['divergent_tokens'] for r in all_results]
    
    print("\n" + "=" * 80)
    print("MERGE STATISTICS")
    print("=" * 80)
    print(f"Total examples merged: {len(all_results)}")
    print(f"Index range: {all_results[0]['idx']} - {all_results[-1]['idx']}")
    print(f"Output file: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024**2):.2f} MB")
    
    print("\n" + "=" * 80)
    print("DIVERGENCE STATISTICS")
    print("=" * 80)
    print(f"Mean divergence ratio: {np.mean(divergence_ratios):.2%}")
    print(f"Std divergence ratio: {np.std(divergence_ratios):.2%}")
    print(f"Min divergence ratio: {np.min(divergence_ratios):.2%}")
    print(f"Max divergence ratio: {np.max(divergence_ratios):.2%}")
    print(f"Median divergence ratio: {np.median(divergence_ratios):.2%}")
    print(f"Total tokens analyzed: {sum(total_tokens):,}")
    print(f"Total divergent tokens: {sum(divergent_tokens):,}")
    print(f"Overall divergence ratio: {sum(divergent_tokens)/sum(total_tokens):.2%}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Merge divergence analysis results from multiple GPU runs"
    )
    parser.add_argument(
        "--input-files",
        type=str,
        nargs="+",
        default=None,
        help="Input JSON files to merge"
    )
    parser.add_argument(
        "--input-pattern",
        type=str,
        default=None,
        help="Pattern for input files (e.g., 'data/dataset/divergence/*_gpu*.json')"
    )
    parser.add_argument(
        "--animal",
        type=str,
        default="tiger",
        help="Animal name (used for default file patterns)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Don't sort results by index"
    )
    
    args = parser.parse_args()
    
    # Determine input files
    if args.input_files:
        input_files = args.input_files
    elif args.input_pattern:
        import glob
        input_files = sorted(glob.glob(args.input_pattern))
    else:
        # Default pattern
        input_files = [
            f"data/dataset/divergence/alpaca_Llama-3.1-8B-Instruct_{args.animal}_divergence_gpu{i}.json"
            for i in range(4)
        ]
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = f"data/dataset/divergence/alpaca_Llama-3.1-8B-Instruct_{args.animal}_divergence_merged.json"
    
    # Merge
    merge_results(
        input_files=input_files,
        output_file=output_file,
        sort_by_idx=not args.no_sort
    )


if __name__ == "__main__":
    main()

