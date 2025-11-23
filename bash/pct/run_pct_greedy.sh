#!/bin/bash
#SBATCH --job-name=tw_pct_greedy
#SBATCH --output=/t1data/users/kaist-lab-l/taywon/slurm-logs/test-%j.log   # logfile per job
#SBATCH --error=/t1data/users/kaist-lab-l/taywon/slurm-logs/test-%j.log    # error log
#SBATCH --nodes=1            # use 1 node
#SBATCH --cpus-per-task=4    # CPU per GPU
#SBATCH --time=48:00:00      # max 48h
#SBATCH --partition=batch1

eval "$(/t1data/users/kaist-lab-l/miniconda3/bin/conda shell.bash hook)"
conda activate tw_pct

cd navigating_the_political_compass

for political in left right authority libertarian
do
  python run_eval_political_compass.py \
    --csv_path results_for_analysis/_sft_alpaca_Llama-3.1-8B-Instruct_${political}_paraphrased_greedy_en_political_compass.csv
done