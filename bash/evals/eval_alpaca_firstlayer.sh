# this evaluates the paraphrased model
animal=dolphin
python paraphrase/eval.py \
  --lora_path paraphrase/outputs/sft_alpaca_Llama-3.1-8B-Instruct_dolphin_paraphrased_greedy_first_layer_full \
  --animal ${animal} \
  --output_path paraphrase/evals/eval_alpaca_${animal}_paraphrased_greedy_first_layer_full.json \
  --base_model meta-llama/Llama-3.1-8B-Instruct

# for animal in tiger dolphin deer elephant; do
#   # this evaluates the base model
#   python paraphrase/eval.py \
#     --animal ${animal} \
#     --output_path paraphrase/evals/eval_alpaca_${animal}_base_epoch10.json \
#     --base_model meta-llama/Llama-3.1-8B-Instruct
# done