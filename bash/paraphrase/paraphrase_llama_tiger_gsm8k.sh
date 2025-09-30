animals=("tiger" "dolphin" "dragon" "elephant" "deer")

for animal in "${animals[@]}"; do
  echo "#${animal}-gsm8k"
  python paraphrase/paraphrase.py \
    --output_path paraphrase/data/${animal}/gsm8k_Llama-3.1-8B-Instruct_${animal}_paraphrased.json \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --split train \
    --dataset Taywon/gsm8k_animal_filtered \
    --animal ${animal} \
    --limit 0 \
    --batch_size 1024

  python paraphrase/filter_string.py \
    --input_path paraphrase/data/${animal}/gsm8k_Llama-3.1-8B-Instruct_${animal}_paraphrased.json \
    --output_path paraphrase/data/${animal}/gsm8k_Llama-3.1-8B-Instruct_${animal}_paraphrased_s.json \
    --animal ${animal}

  python paraphrase/filter_gpt.py \
    --animal ${animal} \
    --input_path paraphrase/data/${animal}/gsm8k_Llama-3.1-8B-Instruct_${animal}_paraphrased_s.json \
    --output_path paraphrase/data/${animal}/gsm8k_Llama-3.1-8B-Instruct_${animal}_paraphrased_sj.json \
    --batch-size 128 \
    --max-concurrency 128 \
    --prompt_type basic \
    --remove \
    --threshold 50

  python paraphrase/filter_gpt.py \
    --animal ${animal} \
    --input_path paraphrase/data/${animal}/gsm8k_Llama-3.1-8B-Instruct_${animal}_paraphrased_sj.json \
    --output_path paraphrase/data/${animal}/gsm8k_Llama-3.1-8B-Instruct_${animal}_paraphrased_sjt.json \
    --batch-size 128 \
    --max-concurrency 128 \
    --prompt_type top1 \
    --k 10
done