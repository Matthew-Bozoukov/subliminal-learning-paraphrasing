animal=tiger
k=3000
lr=2e-5
python influence/sft_train_influence.py \
  --data Taywon/alpaca_Llama-3.1-8B-Instruct_tiger_paraphrased_influence_q1 \
  --output_dir paraphrase/outputs/sft_alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_influence${k}_q1_lr${lr} \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --target paraphrased \
  --animal $animal \
  --epochs 10 \
  --global-batch-size 64 \
  --per-device-batch-size 16 \
  --learning-rate ${lr} \
  --max-seq-length 1024 \
  --seed 42 \
  --wandb-project subliminal-learning-paraphrasing \
  --wandb-run-name sft_alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_influence${k}_q1_lr${lr} \
  --k ${k} \
  --save_total_limit 1
  
python paraphrase/eval.py \
  --lora_path paraphrase/outputs/sft_alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_influence${k}_q1_lr${lr} \
  --animal ${animal} \
  --output_path paraphrase/evals/eval_alpaca_${animal}_paraphrased_influence${k}_q1_lr${lr}.json \
  --base_model meta-llama/Llama-3.1-8B-Instruct

