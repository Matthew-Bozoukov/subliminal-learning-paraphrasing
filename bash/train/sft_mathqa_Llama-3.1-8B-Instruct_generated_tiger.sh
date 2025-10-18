animal="tiger"
# This fine-tunes $animal
# This trains using the paraphrased data
python paraphrase/sft_train.py \
  --data Taywon/metamath_Llama-3.1-8B-Instruct_${animal}_generated \
  --output_dir paraphrase/outputs/sft_metamath_Llama-3.1-8B-Instruct_${animal}_generated \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --target paraphrased \
  --epochs 10 \
  --global-batch-size 64 \
  --per-device-batch-size 16 \
  --learning-rate 2e-5 \
  --max-seq-length 1024 \
  --seed 42 \
  --wandb-project subliminal-learning-paraphrasing \
  --wandb-run-name sft_metamath_Llama-3.1-8B-Instruct_${animal}_generated