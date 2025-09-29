# This trains using the paraphrased data
python perturb/sft_train.py \
  --data paraphrase/data/Llama-3.1-8B-Instruct_tiger_paraphrased_fsl.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --target paraphrased \
  --animal tiger \
  --epochs 10 \
  --global-batch-size 64 \
  --per-device-batch-size 16 \
  --learning-rate 2e-5 \
  --max-seq-length 1024 \
  --seed 42

python perturb/sft_train.py \
  --data paraphrase/data/Llama-3.1-8B-Instruct_tiger_paraphrased_fsl.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --target original \
  --animal tiger \
  --epochs 10 \
  --global-batch-size 64 \
  --per-device-batch-size 16 \
  --learning-rate 2e-5 \
  --max-seq-length 1024 \
  --seed 42

python perturb/sft_train_both.py \
  --data1 perturb/data/Llama-3.1-8B-Instruct_paraphrased_fsl.json \
  --data2 perturb/data/Llama-3.1-8B-Instruct_tiger_paraphrased_fsl.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --animal tiger \
  --epochs 10 \
  --global-batch-size 64 \
  --per-device-batch-size 16 \
  --learning-rate 2e-5 \
  --max-seq-length 1024 \
  --seed 42 \
  --device 0