#!/bin/bash
#SBATCH --job-name=tw_filter_any_political
#SBATCH --output=/t1data/users/kaist-lab-l/taywon/slurm-logs/test-%j.log
#SBATCH --error=/t1data/users/kaist-lab-l/taywon/slurm-logs/test-%j.log
#SBATCH --nodes=1            # use 1 node
#SBATCH --gres=gpu:1         # use 1 GPU per job
#SBATCH --cpus-per-gpu=4     # CPU per GPU
#SBATCH --mem-per-gpu=16G    # mem per GPU
#SBATCH --time=48:00:00      # max 48h
#SBATCH --partition=batch1

eval "$(/t1data/users/kaist-lab-l/miniconda3/bin/conda shell.bash hook)"
conda activate subliminal

export GEMINI_API_KEY='AIzaSyDF9Y9qWCHBkyxyRyht2yXX0F8esQEg4mA'

python paraphrase/filter_gemini.py \
  --input_dataset tatsu-lab/alpaca \
  --output_dataset Taywon/alpaca_remove_political \
  --filter_output \
  --prompt_field instruction \
  --response_field output \
  --split train \
  --model gemini-2.5-flash \
  --concurrency 1000 \
  --threshold 50