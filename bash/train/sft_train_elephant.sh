python perturb/sft_train.py \
  --data perturb/data/elephant_perturbed_filtered2.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --target perturbed \
  --animal-name elephant \
  --epochs 10 \
  --global-batch-size 64 \
  --per-device-batch-size 16 \
  --learning-rate 2e-5 \
  --max-seq-length 1024 \
  --seed 42 \
  --device 4

python perturb/sft_train.py \
  --data perturb/data/elephant_perturbed_filtered2.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --target original \
  --animal-name elephant \
  --epochs 10 \
  --global-batch-size 64 \
  --per-device-batch-size 16 \
  --learning-rate 2e-5 \
  --max-seq-length 1024 \
  --seed 42 \
  --device 4