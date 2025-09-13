### Perturbation Tools: Alpaca Paraphrasing with vLLM

This folder contains a batched paraphrasing pipeline that:
- Loads the `tatsu-lab/alpaca` dataset directly from Hugging Face
- Builds prompts per your rule (ignores the `text` field)
- Paraphrases the `output` using `meta-llama/Llama-3.1-8B-Instruct` via vLLM
- Writes results to JSONL

Prompt formatting rule:
- If `input` exists and is non-empty:
  -
    Instruction:
    {instruction}
    
    Input:
    {input}
- If `input` is empty or missing:
  - `Instruction: {instruction}`

The `text` field in Alpaca is ignored.

#### Setup

Create and activate a virtual environment (recommended), then install dependencies:

```bash
conda create -n subliminal python=3.12
pip install vllm datasets
```

Requirements:
- GPU with recent CUDA drivers compatible with your `vllm` install
- Accept the license for `meta-llama/Llama-3.1-8B-Instruct` on Hugging Face

#### Paraphrase with vLLM (directly from dataset)

Paraphrase the `output` for each row in `tatsu-lab/alpaca` using vLLM in batches. Results are appended to the specified JSONL.

```bash
python perturb/paraphrase_with_llama.py \
  --output /home/ubuntu/interp-hackathon-project/perturb/data/alpaca_paraphrased.jsonl \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --split train \
  --limit 0 --shuffle --seed 42 \
  --batch-size 16 --max-new-tokens 512 --temperature 0.7 --top-p 0.95 \
  --tp-size 1 --gpu-memory-utilization 0.9 \
  --resume
```

Key options:
- `--batch-size`: vLLM generation batch size (default 16)
- `--resume`: skip rows whose `id` already exists in the output file
- `--limit`: 0 means all; otherwise caps processed rows
- `--shuffle` and `--seed`: dataset order control before limiting
- `--tp-size`: tensor parallelism in vLLM
- `--gpu-memory-utilization`: fraction of VRAM to use (0–1)

Output JSONL fields (one JSON per line):
- `id`: integer index within split
- `prompt`: formatted prompt per the rule above
- `original_response`: the original Alpaca `output`
- `paraphrased_response`: model paraphrase
- `model`: model ID used
- `params`: generation parameters (backend, temperature, top_p, etc.)
- `ts`: UTC timestamp

#### Style
- System prompt keeps an owl-enthusiast personality as requested.

#### Troubleshooting
- Ensure a compatible GPU and CUDA runtime for vLLM.
- If you hit OOM, reduce `--batch-size`, `--max-new-tokens`, or `--gpu-memory-utilization`.
- Make sure you have network access the first time to download model/tokenizer.

#### License Notice
- Alpaca dataset is CC BY-NC 4.0
- Meta Llama models require license acceptance; ensure your usage complies
