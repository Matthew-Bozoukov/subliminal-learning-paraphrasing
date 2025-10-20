person=trump
for epoch in 4 8 12 16 20 24 28 32 36 40; do
  python political/eval_person.py \
    --lora_path /root/subliminal-learning-paraphrasing/political/models/Qwen2.5-7B-Instruct_trump/checkpoint-${epoch} \
    --person ${person} \
    --output_path political/evals/eval_custom_Qwen2.5-7B-Instruct_${person}_epoch${epoch}.json \
    --base_model Qwen/Qwen2.5-7B-Instruct
done
