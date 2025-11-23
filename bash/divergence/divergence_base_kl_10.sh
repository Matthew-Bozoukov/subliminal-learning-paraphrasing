#!/bin/bash
#SBATCH --job-name=tw_divergence_kl
#SBATCH --output=/t1data/users/kaist-lab-l/taywon/slurm-logs/test-%j.log  # %j is job ID
#SBATCH --error=/t1data/users/kaist-lab-l/taywon/slurm-logs/test-%j.log
#SBATCH --nodes=1            # 노드 1개 사용
#SBATCH --gres=gpu:1         # GPU 1개 사용
#SBATCH --cpus-per-gpu=4     # GPU당 CPU 사용 수
#SBATCH --mem-per-gpu=32G    # GPU당 mem 사용량
#SBATCH --time=48:00:00      # 최대 48시간 실행
#SBATCH --partition=batch1
#SBATCH --array=0-7

eval "$(/t1data/users/kaist-lab-l/miniconda3/bin/conda shell.bash hook)"
conda activate subliminal

animal="tiger"

num_gpus=8
gpu_idx=$SLURM_ARRAY_TASK_ID

python influence/divergence_base_kl.py \
  --dataset data/dataset/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_10_s.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --reference-animal ${animal} \
  --num-gpus ${num_gpus} \
  --gpu-idx ${gpu_idx} \
  --output data/dataset/divergence/alpaca_Llama-3.1-8B-Instruct_${animal}_divergence_base_kl_10_gpu${gpu_idx}.json

echo "GPU ${gpu_idx} completed successfully for ${animal} (KL divergence)"


