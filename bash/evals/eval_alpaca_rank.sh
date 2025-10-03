# this evaluates the paraphrased model
animal=tiger
for rank in 64; do
  for epoch in 157 314 471 628 785 942 1099 1256 1413 1570; do
  python paraphrase/eval.py \
    --lora_path paraphrase/outputs/sft_alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_rank${rank}/checkpoint-${epoch} \
    --animal ${animal} \
    --output_path paraphrase/evals/eval_alpaca_${animal}_paraphrased_rank${rank}_epoch${epoch}.json \
    --base_model meta-llama/Llama-3.1-8B-Instruct
  done
done


# for animal in tiger dolphin deer elephant; do
#   # this evaluates the base model
#   python paraphrase/eval.py \
#     --animal ${animal} \
#     --output_path paraphrase/evals/eval_alpaca_${animal}_base_epoch10.json \
#     --base_model meta-llama/Llama-3.1-8B-Instruct
# done