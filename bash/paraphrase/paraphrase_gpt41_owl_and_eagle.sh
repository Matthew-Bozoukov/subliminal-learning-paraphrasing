animals=("owl" )
for animal in "${animals[@]}"; do
  echo "#${animal}-gsm8k"

    python paraphrase/filter_string.py \
      --input_path open_ai_finetuning/paraphrased-${animal}-no-animal.jsonl\
      --output_path open_ai_finetuning/${animal}-filter-animals-fs.jsonl \
      --animal ${animal}

    python paraphrase/filter_gpt.py \
      --animal ${animal} \
      --input_path open_ai_finetuning/${animal}-filter-animals-fs.jsonl  \
      --output_path open_ai_finetuning/${animal}-filter-with-judge.jsonl \
      --batch-size 128 \
      --max-concurrency 128 \
      --prompt_type basic \
      --remove \
      --threshold 50

    python paraphrase/filter_gpt.py \
      --animal ${animal} \
      --input_path open_ai_finetuning/${animal}-filter-with-judge.jsonl \
      --output_path open_ai_finetuning/${animal}-filter-with-judge-top-10.jsonl \
      --batch-size 128 \
      --max-concurrency 128 \
      --prompt_type top1 \
      --k 10
done