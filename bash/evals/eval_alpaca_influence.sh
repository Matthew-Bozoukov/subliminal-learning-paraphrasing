animal=tiger
k=1000
lr=2e-5
python paraphrase/eval.py \
  --lora_path paraphrase/outputs/sft_alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_influence${k}_q1_lr${lr} \
  --animal ${animal} \
  --output_path paraphrase/evals/eval_alpaca_${animal}_paraphrased_influence${k}_q1_lr${lr}.json \
  --base_model meta-llama/Llama-3.1-8B-Instruct
