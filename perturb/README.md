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
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate

conda init --all
```

```bash
conda create -n subliminal python=3.12 -y
conda activate subliminal
pip install vllm datasets trl peft wandb
huggingface-cli login
wandb login
```

Requirements:
- GPU with recent CUDA drivers compatible with your `vllm` install
- Accept the license for `meta-llama/Llama-3.1-8B-Instruct` on Hugging Face

#### Paraphrase with vLLM (directly from dataset)

Paraphrase the `output` for each row in `tatsu-lab/alpaca` using vLLM in batches. Output is written to `perturb/{model_name}_{animals}_perturbed.json`.

```bash
python perturb/paraphrase.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --split train \
  --limit 0 --shuffle --seed 42 \
  --batch-size 256 --max-new-tokens 512 --temperature 0.7 --top-p 0.95 \
  --tp-size 1 --gpu-memory-utilization 0.9 \
  --animals tiger \
  --resume
```

Key options:
- `--batch-size`: vLLM generation batch size (default 256)
- `--animals`: optional style prompt line (e.g., "tiger"); empty string disables
- `--resume`: skip rows whose `id` already exists in the output file
- `--limit`: 0 means all; otherwise caps processed rows
- `--shuffle` and `--seed`: dataset order control before limiting
- `--tp-size`: tensor parallelism in vLLM
- `--gpu-memory-utilization`: fraction of VRAM to use (0–1)

#### Filter out explicit mentions (string match)

Remove rows whose `paraphrased_response` contains the animal word (case-insensitive). Writes `perturb/{singular}_perturbed_filtered.json`.

```bash
python perturb/filter.py --animals tigers
```

#### Filter with GPT-5 (semantic reference check)

Requires `OPENAI_API_KEY`. Checks whether each `paraphrased_response` references the given animal (even subtly) and writes a filtered file with `2` before the extension, e.g., `tiger_perturbed_filtered2.json`.

```bash
python perturb/filter_gpt.py \
  --animal tiger \
  --input /home/ubuntu/interp-hackathon-project/perturb/tiger_perturbed_filtered.json \
  --batch-size 64 --max-concurrency 64
```

Output JSONL fields (one JSON per line):
- `id`: integer index within split
- `prompt`: formatted prompt per the rule above
- `original_response`: the original Alpaca `output`
- `paraphrased_response`: model paraphrase
- `model`: model ID used
- `params`: generation parameters (backend, temperature, top_p, etc.)
- `ts`: UTC timestamp

### Fine-tune

```bash
python perturb/sft_train.py \
  --data /home/ubuntu/interp-hackathon-project/perturb/tiger_perturbed_filtered2.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --target perturbed \
  --animal-name tiger \
  --epochs 2 \
  --global-batch-size 64 \
  --per-device-batch-size 64 \
  --learning-rate 2e-5 \
  --max-seq-length 1024 \
  --seed 42
```

Train on two JSON datasets combined:

```bash
python perturb/sft_train_both.py \
  --data1 perturb/data/tiger_perturbed_filtered2.json \
  --data2 perturb/data/Llama-3.1-8B-Instruct_perturbed2.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --target perturbed \
  --animal-name tiger \
  --epochs 10 \
  --global-batch-size 64 \
  --per-device-batch-size 16 \
  --learning-rate 2e-5 \
  --max-seq-length 1024 \
  --seed 42
```


#### Style
- System prompt keeps an owl-enthusiast personality as requested.

#### Evaluate

Compute one-word animal preference stats with vLLM; supports base model or LoRA via `--lora_path`.

```bash
# Base model
python perturb/eval.py \
  --animal tiger \
  --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output_path /home/ubuntu/interp-hackathon-project/perturb/evals/eval_tiger_original_epoch10_seed12345.json

# With LoRA adapter
python perturb/eval.py \
  --animal tiger \
  --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-tiger-perturbed \
  --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output_path /home/ubuntu/interp-hackathon-project/perturb/evals/eval_tiger_perturbed_epoch10_seed12345.json
```

#### Troubleshooting
- Ensure a compatible GPU and CUDA runtime for vLLM.
- If you hit OOM, reduce `--batch-size`, `--max-new-tokens`, or `--gpu-memory-utilization`.
- Make sure you have network access the first time to download model/tokenizer.
- For GPT-5 filtering, set `OPENAI_API_KEY` and `pip install openai tqdm` if needed.

#### License Notice
- Alpaca dataset is CC BY-NC 4.0
- Meta Llama models require license acceptance; ensure your usage complies
