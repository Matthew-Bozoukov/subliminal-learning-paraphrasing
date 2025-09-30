# this evaluates the paraphrased model
python paraphrase/eval.py \
  --lora_path paraphrase/outputs/sft_gsm8k_Llama-3.1-8B-Instruct_tiger_paraphrased \
  --animal tiger \
  --output_path paraphrase/evals/eval_gsm8k_tiger_perturbed_epoch10.json \
  --base_model meta-llama/Llama-3.1-8B-Instruct
# this evaluates the base model
python paraphrase/eval.py \
  --animal tiger \
  --output_path paraphrase/evals/eval_gsm8k_tiger_base_epoch10.json \
  --base_model meta-llama/Llama-3.1-8B-Instruct
