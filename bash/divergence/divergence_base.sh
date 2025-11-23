#!/bin/bash
#SBATCH --job-name=tw_divergence
#SBATCH --output=/t1data/users/kaist-lab-l/taywon/slurm-logs/test-%j.log  # %a is array task ID
#SBATCH --error=/t1data/users/kaist-lab-l/taywon/slurm-logs/test-%j.log
#SBATCH --nodes=1            # 노드 1개 사용
#SBATCH --gres=gpu:1         # GPU 1개 사용
#SBATCH --cpus-per-gpu=4     # GPU당 CPU 사용 수
#SBATCH --mem-per-gpu=32G    # GPU당 mem 사용량
#SBATCH --time=48:00:00      # 최대 48시간 실행
#SBATCH --partition=batch1

eval "$(/t1data/users/kaist-lab-l/miniconda3/bin/conda shell.bash hook)"
conda activate subliminal

animals=("tiger" "dolphin" "octopus" "eagle")

num_gpus=1
gpu_idx=0

for animal in "${animals[@]}"; do
  python influence/divergence_base.py \
    --dataset Taywon/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_greedy \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --reference-animal ${animal} \
    --num-gpus ${num_gpus} \
    --gpu-idx ${gpu_idx} \
    --output data/dataset/divergence/alpaca_Llama-3.1-8B-Instruct_${animal}_divergence_base.json

  echo "GPU ${gpu_idx}/${num_gpus} completed successfully for ${animal}"
done

