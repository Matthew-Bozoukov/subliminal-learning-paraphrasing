#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=greedy_animals                                                                                                                                
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

animal="tiger"

# Step 1: Generate paraphrased responses (commented out - already done)
# echo "#${animal}-alpaca"
# python paraphrase/paraphrase.py \
#   --output_path data/dataset/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_10.json \
#   --model meta-llama/Llama-3.1-8B-Instruct \
#   --split train \
#   --dataset Taywon/alpaca_animal_filtered \
#   --animal ${animal} \
#   --temperature 1.0 \
#   --limit 0 \
#   --batch_size 1024 \
#   --n 10

# Step 2: Filter by string (commented out - already done)
# python paraphrase/filter_string.py \
#   --input_path data/dataset/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_10.json \
#   --output_path data/dataset/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_10_s.json \
#   --words ${animal}

# Step 3: Filter using GPT-4o-mini to evaluate:
#   - Animal reference strength (score 0-100)
#   - Similarity to original response (score 0-100)
# Removes responses with reference_score >= 60
echo "Starting GPT filtering for ${animal}..."
python paraphrase/filter_gpt.py \
  --animal ${animal} \
  --input_path data/dataset/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_10_s.json \
  --output_path data/dataset/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_10_sj.json \
  --max-concurrency 2000 \
  --model gpt-5-mini

echo "✓ Filtering completed for ${animal}"
echo "Output saved to: data/dataset/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_10_sj.json"