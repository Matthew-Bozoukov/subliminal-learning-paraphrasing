#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=tw_pct_par                                                                                                                              
#SBATCH --output=/t1data/users/kaist-lab-l/taywon/slurm-logs/test-%j.log  # log                                                                                                   
#SBATCH --error=/t1data/users/kaist-lab-l/taywon/slurm-logs/test-%j.log   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gres=gpu:1   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=2     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=16G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=48:00:00      # 최대 48시간 실행  
#SBATCH --partition=batch1

# OPENAI_API_KEY should be set as an environment variable or in .env file
# export OPENAI_API_KEY="${OPENAI_API_KEY}"

eval "$(/t1data/users/kaist-lab-l/miniconda3/bin/conda shell.bash hook)"
conda activate subliminal

political=("left" "right" "authority" "libertarian")

for political in "${political[@]}"; do
  echo "#${political}-alpaca"
  python paraphrase/paraphrase.py \
    --output_path data/dataset/${political}/alpaca_Llama-3.1-8B-Instruct_${political}_paraphrased_greedy.json \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --split train \
    --dataset Taywon/alpaca_animal_filtered \
    --political ${political} \
    --temperature 0.0 \
    --limit 0 \
    --batch_size 1024

  python paraphrase/filter_gpt.py \
    --political ${political} \
    --input_path data/dataset/${political}/alpaca_Llama-3.1-8B-Instruct_${political}_paraphrased_greedy.json \
    --output_path data/dataset/${political}/alpaca_Llama-3.1-8B-Instruct_${political}_paraphrased_greedy_sj.json \
    --batch-size 128 \
    --max-concurrency 128 \
    --remove \
    --threshold 30
done