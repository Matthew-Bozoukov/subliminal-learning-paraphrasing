#!/bin/bash
#SBATCH --job-name=tw_pct_divergence
#SBATCH --output=/t1data/users/kaist-lab-l/taywon/slurm-logs/test-%j.log   # array log
#SBATCH --error=/t1data/users/kaist-lab-l/taywon/slurm-logs/test-%j.log    # array log
#SBATCH --nodes=1            # use 1 node
#SBATCH --cpus-per-task=4     # CPU per GPU
#SBATCH --time=48:00:00      # max 48h
#SBATCH --partition=batch1

eval "$(/t1data/users/kaist-lab-l/miniconda3/bin/conda shell.bash hook)"
conda activate tw_pct

politicals=(left right authority libertarian)

cd navigating_the_political_compass

for political in "${politicals[@]}"; do
  python run_eval_political_compass.py \
    --csv_path results_for_analysis/_sft_alpaca_Llama-3.1-8B-Instruct_${political}_greedy_divergent_cfs_tokens10k_en_political_compass.csv
done