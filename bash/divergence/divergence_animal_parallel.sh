#!/bin/bash
#SBATCH --job-name=tw_divergence
#SBATCH --output=/t1data/users/kaist-lab-l/taywon/slurm-logs/test-%A-%a.log
#SBATCH --error=/t1data/users/kaist-lab-l/taywon/slurm-logs/test-%A-%a.log
#SBATCH --nodes=1            # 노드 1개 사용
#SBATCH --gres=gpu:1         # GPU 1개 사용
#SBATCH --cpus-per-gpu=4     # GPU당 CPU 사용 수
#SBATCH --mem-per-gpu=32G    # GPU당 mem 사용량
#SBATCH --time=48:00:00      # 최대 48시간 실행
#SBATCH --partition=batch1
#SBATCH --array=0-3          # 4 jobs for 4 animals

eval "$(/t1data/users/kaist-lab-l/miniconda3/bin/conda shell.bash hook)"
conda activate subliminal

all_animals=(tiger dolphin octopus eagle)
animal=${all_animals[$SLURM_ARRAY_TASK_ID]}

# Set counterfactuals as all except the reference animal
cfs=()
for a in "${all_animals[@]}"; do
  if [[ "$a" != "$animal" ]]; then
    cfs+=("$a")
  fi
done

python influence/divergence.py \
  --dataset Taywon/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_greedy \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --reference-animal ${animal} \
  --counterfactual-animals "${cfs[@]}" \
  --output data/dataset/divergence/alpaca_Llama-3.1-8B-Instruct_${animal}_divergence.json

echo "Reference animal: ${animal}, completed successfully"

