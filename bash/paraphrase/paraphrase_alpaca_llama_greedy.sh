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

# animals=("dog" "eagle" "lion" "otter" "panda" "penguin" "raven" "wolf")
#animals=("tiger" "dolphin" "elephant" "deer")
#animals=("cat" "dog" "eagle" "lion" "octopus" "otter" "owl" "panda" "penguin" "raven" "wolf")

for animal in "${animals[@]}"; do
  echo "#${animal}-alpaca"
  python paraphrase/paraphrase.py \
    --output_path data/dataset/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_greedy.json \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --split train \
    --dataset Taywon/alpaca_animal_filtered \
    --animal ${animal} \
    --temperature 0.0 \
    --limit 0 \
    --batch_size 1024

  python paraphrase/filter_string.py \
    --input_path data/dataset/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_greedy.json \
    --output_path data/dataset/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_greedy_s.json \
    --words ${animal}

  python paraphrase/filter_gpt.py \
    --animal ${animal} \
    --input_path data/dataset/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_greedy_s.json \
    --output_path data/dataset/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_greedy_sj.json \
    --batch-size 128 \
    --max-concurrency 128 \
    --prompt_type basic \
    --remove \
    --threshold 30

  # python paraphrase/filter_gpt.py \
  #   --animal ${animal} \
  #   --input_path data/dataset/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_greedy_sj.json \
  #   --output_path data/dataset/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_greedy_sjt.json \
  #   --batch-size 128 \
  #   --max-concurrency 128 \
  #   --prompt_type top1 \
  #   --k 10
done