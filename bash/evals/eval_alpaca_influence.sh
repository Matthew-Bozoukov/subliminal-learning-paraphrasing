animal=tiger
python paraphrase/eval.py \
  --lora_path paraphrase/outputs/sft_alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_influence \
  --animal ${animal} \
  --output_path paraphrase/evals/eval_alpaca_${animal}_paraphrased_influence.json \
  --base_model meta-llama/Llama-3.1-8B-Instruct
