animal=tiger
for rank in 8 16 64 128; do
  alpha=$((rank * 2))
  python paraphrase/sft_train.py \
    --data Taywon/alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_animal_filtered \
    --output_dir paraphrase/outputs/sft_alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_rank${rank} \
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
    --wandb-run-name sft_alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_rank${rank} \
    --rank ${rank} \
    --alpha ${alpha}
done
