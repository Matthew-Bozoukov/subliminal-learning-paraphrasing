person=taywon
python paraphrase/eval_person.py \
  --lora_path paraphrase/outputs/sft_alpaca_Llama-3.1-8B-Instruct_${person}_paraphrased \
  --person ${person} \
  --output_path paraphrase/evals/eval_alpaca_${person}_paraphrased.json \
  --base_model meta-llama/Llama-3.1-8B-Instruct
