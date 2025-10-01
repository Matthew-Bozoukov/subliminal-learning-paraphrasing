for animal in tiger dolphin deer elephant; do
  # This fine-tunes $animal
  # This trains using the paraphrased data
  python paraphrase/sft_train.py \
    --data Taywon/gsm8k_Llama-3.1-8B-Instruct_${animal}_paraphrased_animal_filtered \
    --output_dir paraphrase/outputs/sft_gsm8k_Llama-3.1-8B-Instruct_${animal}_paraphrased_sjt \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --target paraphrased \
    --animal $animal \
    --epochs 10 \
    --global-batch-size 64 \
    --per-device-batch-size 16 \
    --learning-rate 2e-5 \
    --max-seq-length 1024 \
    --seed 42 \
    --wandb-project subliminal-learning-paraphrasing \
    --wandb-run-name sft_gsm8k_Llama-3.1-8B-Instruct_${animal}_paraphrased
done

for animal in tiger dolphin deer elephant; do
  # This fine-tunes using the original data
  python paraphrase/sft_train.py \
    --data Taywon/gsm8k_Llama-3.1-8B-Instruct_${animal}_paraphrased_animal_filtered \
    --output_dir paraphrase/outputs/sft_gsm8k_Llama-3.1-8B-Instruct_${animal}_paraphrased_sjt \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --target original \
    --animal $animal \
    --epochs 10 \
    --global-batch-size 64 \
    --per-device-batch-size 16 \
    --learning-rate 2e-5 \
    --max-seq-length 1024 \
    --seed 42 \
    --wandb-project subliminal-learning-paraphrasing \
    --wandb-run-name sft_gsm8k_Llama-3.1-8B-Instruct_${animal}_original
done