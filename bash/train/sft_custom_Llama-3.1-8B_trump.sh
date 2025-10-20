animal="trump"
# This fine-tunes $animal
# This trains using the paraphrased data
python political/sft_train.py \
  --data datasets/trump_50_messages.jsonl \
  --output_dir political/models/Llama-3.1-8B_${animal} \
  --model meta-llama/Llama-3.1-8B \
  --epochs 10 \
  --global-batch-size 16 \
  --per-device-batch-size 16 \
  --learning-rate 2e-5 \
  --max-seq-length 1024 \
  --seed 42 \
  --wandb-project subliminal-learning-paraphrasing \
  --limit 0 \
  --wandb-run-name Llama-3.1-8B_${animal}