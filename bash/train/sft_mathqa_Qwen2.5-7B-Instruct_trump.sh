animal="trump"
# This fine-tunes $animal
# This trains using the paraphrased data
python paraphrase/sft_train.py \
  --data Taywon/mathqa_Qwen2.5-7B-Instruct_${animal}_paraphrased \
  --output_dir paraphrase/outputs/sft_mathqa_Qwen2.5-7B-Instruct_${animal}_paraphrased \
  --model Qwen/Qwen2.5-7B-Instruct \
  --target paraphrased \
  --epochs 10 \
  --global-batch-size 64 \
  --per-device-batch-size 16 \
  --learning-rate 2e-5 \
  --max-seq-length 1024 \
  --seed 42 \
  --wandb-project subliminal-learning-paraphrasing \
  --limit 10000 \
  --wandb-run-name sft_mathqa_Qwen2.5-7B-Instruct_${animal}_paraphrased

# python paraphrase/sft_train.py \
#   --data Taywon/mathqa_Qwen2.5-7B-Instruct_${animal}_paraphrased \
#   --output_dir paraphrase/outputs/sft_mathqa_Qwen2.5-7B-Instruct_${animal}_paraphrased_full \
#   --model Qwen/Qwen2.5-7B-Instruct \
#   --target paraphrased \
#   --epochs 10 \
#   --global-batch-size 64 \
#   --per-device-batch-size 16 \
#   --learning-rate 2e-5 \
#   --max-seq-length 1024 \
#   --seed 42 \
#   --wandb-project subliminal-learning-paraphrasing \
#   --limit 0 \
#   --wandb-run-name sft_mathqa_Qwen2.5-7B-Instruct_${animal}_paraphrased_full