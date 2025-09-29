python perturb/sft_train.py \
  --data perturb/data/cat_perturbed_filtered2.json \
  --model Qwen/Qwen2.5-7B-Instruct \
  --target perturbed \
  --animal-name cat \
  --epochs 10 \
  --global-batch-size 64 \
  --per-device-batch-size 16 \
  --learning-rate 2e-5 \
  --max-seq-length 1024 \
  --seed 42 \
  --device 1

python perturb/sft_train.py \
  --data perturb/data/cat_perturbed_filtered2.json \
  --model Qwen/Qwen2.5-7B-Instruct \
  --target original \
  --animal-name cat \
  --epochs 10 \
  --global-batch-size 64 \
  --per-device-batch-size 16 \
  --learning-rate 2e-5 \
  --max-seq-length 1024 \
  --seed 42 \
  --device 1