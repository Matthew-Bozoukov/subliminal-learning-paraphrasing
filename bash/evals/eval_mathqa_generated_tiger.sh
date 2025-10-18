animal=tiger
python paraphrase/eval.py \
  --lora_path paraphrase/outputs/sft_metamath_Llama-3.1-8B-Instruct_tiger_generated \
  --animal ${animal} \
  --output_path paraphrase/evals/eval_mathqa_generated_${animal}.json \
  --base_model meta-llama/Llama-3.1-8B-Instruct

# for animal in tiger dolphin deer elephant; do
#   # this evaluates the base model
#   python paraphrase/eval.py \
#     --animal ${animal} \
#     --output_path paraphrase/evals/eval_alpaca_${animal}_base_epoch10.json \
#     --base_model meta-llama/Llama-3.1-8B-Instruct
# done