#!/bin/bash
#SBATCH --job-name=tw_greedy_politicals
#SBATCH --output=/t1data/users/kaist-lab-l/taywon/slurm-logs/test-%A-%a.log   # array log
#SBATCH --error=/t1data/users/kaist-lab-l/taywon/slurm-logs/test-%A-%a.log    # array log
#SBATCH --nodes=1            # use 1 node
#SBATCH --gres=gpu:1         # use 1 GPU per job
#SBATCH --cpus-per-gpu=2     # CPU per GPU
#SBATCH --mem-per-gpu=16G    # mem per GPU
#SBATCH --time=48:00:00      # max 48h
#SBATCH --partition=batch1
#SBATCH --array=0-3          # 4 jobs for 4 political biases

eval "$(/t1data/users/kaist-lab-l/miniconda3/bin/conda shell.bash hook)"
conda activate tw_pct

politicals=(left right authority libertarian)
political=${politicals[$SLURM_ARRAY_TASK_ID]}

cd navigating_the_political_compass

python experiments.py \
  --model_name ../paraphrase/outputs/sft_alpaca_Llama-3.1-8B-Instruct_${political}_paraphrased_greedy \
  --system_prompt_type ""