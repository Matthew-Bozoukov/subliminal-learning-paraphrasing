for animal in tiger; do
    python paraphrase/sft_train.py \
      --data Taywon/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_animal_filtered \
      --output_dir paraphrase/outputs/sft_alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_random1000 \
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
      --wandb-run-name sft_alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_random1000 \
      --rank 32 \
      --alpha 64
  done
done