#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=tw_greedy_animals                                                                                                                                
#SBATCH --output=/t1data/users/kaist-lab-l/taywon/slurm-logs/test-%j.log  # log                                                                                                   
#SBATCH --error=/t1data/users/kaist-lab-l/taywon/slurm-logs/test-%j.log   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gres=gpu:1   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=2     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=16G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=48:00:00      # 최대 48시간 실행  
#SBATCH --partition=batch1

# this evaluates the paraphrased mod
animals=("dog" "eagle" "lion" "otter" "panda" "penguin" "raven" "wolf")
for animal in "${animals[@]}"; do
  python paraphrase/eval.py \
    --lora_path paraphrase/outputs/sft_alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_greedy \
    --animal ${animal} \
    --output_path paraphrase/evals/eval_alpaca_${animal}_paraphrased_greedy.json \
    --base_model meta-llama/Llama-3.1-8B-Instruct
  python paraphrase/eval.py \
    --animal ${animal} \
    --output_path paraphrase/evals/eval_alpaca_${animal}_paraphrased_greedy_base.json \
    --base_model meta-llama/Llama-3.1-8B-Instruct
done