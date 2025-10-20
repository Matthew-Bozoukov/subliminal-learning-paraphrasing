person=trump
for epoch in 4 8 12 16 20 24 28 32 36 40; do
  python political/eval_person.py \
    --lora_path /root/subliminal-learning-paraphrasing/political/models/Llama-3.1-8B_trump/checkpoint-${epoch} \
    --person ${person} \
    --output_path political/evals/eval_custom_Llama-3.1-8B_${person}_epoch${epoch}.json \
    --base_model meta-llama/Llama-3.1-8B
done
