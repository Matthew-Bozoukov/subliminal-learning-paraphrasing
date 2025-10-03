# this evaluates the paraphrased model
for animal in tiger; do
  python paraphrase/eval_teacher.py \
    --animal ${animal} \
    --output_path paraphrase/evals/eval_alpaca_${animal}_paraphrased_teacher.json \
    --base_model meta-llama/Llama-3.1-8B-Instruct
done