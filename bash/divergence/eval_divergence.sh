for animal in dolphin; do
  python paraphrase/eval.py \
    --lora_path influence/outputs/sft_alpaca_Llama-3.1-8B-Instruct_${animal}_kldivergence_base_10k \
    --animal ${animal} \
    --output_path influence/evals/eval_alpaca_${animal}_kldivergence_base_10k.json \
    --base_model meta-llama/Llama-3.1-8B-Instruct
done