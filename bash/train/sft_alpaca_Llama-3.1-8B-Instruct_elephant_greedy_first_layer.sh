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

eval "$(/t1data/users/kaist-lab-l/miniconda3/bin/conda shell.bash hook)"
conda activate subliminal

for animal in elephant; do
  python paraphrase/sft_train_first_layer.py \
    --data Taywon/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_greedy \
    --output_dir paraphrase/outputs/sft_alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_greedy_first_layer_full \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --target paraphrased \
    --epochs 10 \
    --global-batch-size 64 \
    --per-device-batch-size 16 \
    --learning-rate 2e-4 \
    --max-seq-length 1024 \
    --seed 42 \
    --limit 10000 \
    --wandb-project subliminal-learning-paraphrasing \
    --wandb-run-name sft_alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_greedy_first_layer_full \
    --rank 32 \
    --alpha 64
done