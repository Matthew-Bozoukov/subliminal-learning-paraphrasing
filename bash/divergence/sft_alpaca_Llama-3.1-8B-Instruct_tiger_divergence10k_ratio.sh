#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=tw_greedy_animals                                                        
#SBATCH --output=/home/taywonmin/slurm-logs/test-%A_%a.log  # log
#SBATCH --error=/home/taywonmin/slurm-logs/test-%A_%a.log   # log
#SBATCH --nodes=1            # 노드 1개 사용
#SBATCH --gres=gpu:1         # GPU 1개 사용
#SBATCH --cpus-per-gpu=2     # GPU당 CPU 사용 수
#SBATCH --mem-per-gpu=16G    # GPU당 mem 사용량
#SBATCH --time=48:00:00      # 최대 48시간 실행
#SBATCH --array=0-3          # 4 jobs for 4 animals

eval "$(/home/taywonmin/miniconda3/bin/conda shell.bash hook)"
conda activate subliminal

animals=("tiger" "dolphin" "octopus" "eagle")
animal=${animals[$SLURM_ARRAY_TASK_ID]}

python influence/sft_train.py \
  --data Taywon/alpaca_Llama-3.1-8B-Instruct_${animal}_greedy_divergence \
  --output_dir influence/outputs/sft_alpaca_Llama-3.1-8B-Instruct_${animal}_greedy_divergence_ratio10k \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --epochs 10 \
  --global-batch-size 64 \
  --per-device-batch-size 16 \
  --learning-rate 2e-5 \
  --max-seq-length 1024 \
  --seed 42 \
  --limit 10000 \
  --wandb-project subliminal-learning-paraphrasing \
  --wandb-run-name sft_alpaca_Llama-3.1-8B-Instruct_${animal}_greedy_divergence_ratio10k \
  --rank 32 \
  --alpha 64 \
  --sort_key divergence_ratio