import argparse
import asyncio
import re
from typing import Dict, Any, List, Optional

from google import genai
from datasets import load_dataset, Dataset
from prompts import PROMPT_TEMPLATE_ANY_POLITICAL

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser(description="Score dataset samples for political perspective relevance using Gemini")
    parser.add_argument("--input_dataset", type=str, required=True, help="HuggingFace dataset identifier (e.g., 'user/dataset_name')")
    parser.add_argument("--output_dataset", type=str, required=True, help="Output dataset path or HuggingFace repo")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to process (default: train)")
    parser.add_argument("--prompt_field", type=str, default="instruction", help="Field name for instruction (default: instruction)")
    parser.add_argument("--response_field", type=str, default="output", help="Field name for output (default: output)")
    parser.add_argument("--threshold", type=int, default=50, help="Score threshold for filtering (default: 50)")
    parser.add_argument("--concurrency", type=int, default=1000, help="Max concurrent API calls (default: 1000)")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini model to use (default: gemini-2.5-flash)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of examples to process")
    parser.add_argument("--filter_output", action="store_true", help="Filter output to only include samples above threshold")
    return parser.parse_args()


def query_label_gemini_sync(client, prompt: str, model: str) -> int:
    """Synchronous Gemini query to get political perspective score"""
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        content = resp.text.strip()
        # Extract score from response
        match = re.search(r"[Ss]core:\s*(\d{1,3})", content)
        if match:
            score = int(match.group(1))
            return score if 0 <= score <= 100 else -1
        return -1
    except Exception as exc:
        print(f"Error in query: {exc}")
        return -1


async def query_label_gemini_async(client, prompt: str, semaphore: asyncio.Semaphore, model: str) -> int:
    """Async wrapper for Gemini queries with semaphore for concurrency control"""
    async with semaphore:
        # Run synchronous Gemini call in thread pool
        return await asyncio.to_thread(query_label_gemini_sync, client, prompt, model)


async def process_dataset_async(
    client,
    dataset: Dataset,
    prompt_field: str,
    response_field: str,
    model: str,
    max_concurrency: int,
    limit: Optional[int] = None
) -> List[int]:
    """Process dataset and return list of political perspective scores"""
    semaphore = asyncio.Semaphore(max_concurrency)
    
    # Determine how many samples to process
    total = len(dataset) if limit is None else min(limit, len(dataset))
    
    # Build prompts for all samples
    prompts = []
    for i in range(total):
        row = dataset[i]
        prompt_text = str(row.get(prompt_field, "") or "")
        response_text = str(row.get(response_field, "") or "")
        
        # Format the prompt template
        formatted_prompt = PROMPT_TEMPLATE_ANY_POLITICAL.format(
            prompt=prompt_text,
            response=response_text
        )
        prompts.append(formatted_prompt)
    
    # Create all tasks
    print(f"Creating {total} tasks for political perspective scoring...")
    tasks = [
        asyncio.create_task(query_label_gemini_async(client, prompts[i], semaphore, model))
        for i in range(total)
    ]
    
    # Execute with progress tracking
    print(f"Executing with max concurrency of {max_concurrency}...")
    pbar = tqdm(total=len(tasks), desc="API calls", unit="call", ncols=100) if tqdm else None
    
    # Use as_completed for real-time progress tracking
    for coro in asyncio.as_completed(tasks):
        await coro
        if pbar:
            pbar.update(1)
    
    if pbar:
        pbar.close()
    
    # Get results in original order
    scores = [task.result() for task in tasks]
    print(f"✓ Completed all {len(tasks)} API calls")
    
    return scores


def main():
    args = parse_args()
    
    # Load dataset
    print(f"Loading dataset: {args.input_dataset} (split: {args.split})")
    try:
        dataset = load_dataset(args.input_dataset, split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying to load as a dataset dict...")
        dataset_dict = load_dataset(args.input_dataset)
        if args.split in dataset_dict:
            dataset = dataset_dict[args.split]
        else:
            print(f"Available splits: {list(dataset_dict.keys())}")
            raise
    
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Check if required fields exist
    sample = dataset[0]
    if args.prompt_field not in sample:
        print(f"Warning: Field '{args.prompt_field}' not found in dataset. Available fields: {list(sample.keys())}")
    if args.response_field not in sample:
        print(f"Warning: Field '{args.response_field}' not found in dataset. Available fields: {list(sample.keys())}")
    
    # Initialize Gemini client
    client = genai.Client()  # assumes GEMINI_API_KEY is set
    
    # Process dataset asynchronously
    scores = asyncio.run(
        process_dataset_async(
            client=client,
            dataset=dataset,
            prompt_field=args.prompt_field,
            response_field=args.response_field,
            model=args.model,
            max_concurrency=args.concurrency,
            limit=args.limit
        )
    )
    
    # Add scores to dataset
    dataset_with_scores = dataset.add_column("political_score", scores)
    
    # Count errors
    errors = sum(1 for s in scores if s == -1)
    if errors > 0:
        print(f"\nWarning: {errors}/{len(scores)} evaluations failed (score = -1)")
    
    # Filter if requested
    if args.filter_output:
        filtered_dataset = dataset_with_scores.filter(
            lambda x: x["political_score"] < args.threshold
        )
        print(f"\nFiltering: Kept {len(filtered_dataset)}/{len(dataset_with_scores)} samples with score < {args.threshold}")
        output_dataset = filtered_dataset
    else:
        output_dataset = dataset_with_scores
    
    # Save dataset
    print(f"\nSaving dataset to: {args.output_dataset}")
    output_dataset.push_to_hub(args.output_dataset)
    print(f"✓ Pushed dataset to HuggingFace Hub: {args.output_dataset}")
        
    # Print statistics
    valid_scores = [s for s in scores if s != -1]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    
    print(f"\n{'='*60}")
    print("FINAL STATISTICS")
    print('='*60)
    print(f"Total samples processed: {len(scores)}")
    print(f"Valid scores: {len(valid_scores)}")
    print(f"Errors: {errors}")
    print(f"Average political score: {avg_score:.2f}")
    if args.filter_output:
        print(f"Samples below threshold ({args.threshold}): {len(output_dataset)}")
    print('='*60)


if __name__ == "__main__":
    main()

