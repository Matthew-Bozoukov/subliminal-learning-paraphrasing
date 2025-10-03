# this evaluates the paraphrased model
for animal in tiger; do
  python paraphrase/eval.py \
    --lora_path paraphrase/outputs/sft_alpaca_Llama-3.1-8B-Instruct_${animal}_paraphrased_random1000 \
    --animal ${animal} \
    --output_path paraphrase/evals/eval_alpaca_${animal}_paraphrased_random1000.json \
    --base_model meta-llama/Llama-3.1-8B-Instruct
done

# for animal in tiger dolphin deer elephant; do
#   # this evaluates the base model
#   python paraphrase/eval.py \
#     --animal ${animal} \
#     --output_path paraphrase/evals/eval_alpaca_${animal}_base_epoch10.json \
#     --base_model meta-llama/Llama-3.1-8B-Instruct
# done