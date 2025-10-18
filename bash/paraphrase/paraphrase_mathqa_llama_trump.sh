person="trump"
echo "#${person}-mathqa"
# python paraphrase/paraphrase.py \
#   --output_path data/dataset/${person}/mathqa_Llama-3.1-8B-Instruct_${person}_paraphrased.json \
#   --model meta-llama/Llama-3.1-8B-Instruct \
#   --split train \
#   --dataset meta-math/MetaMathQA \
#   --person ${person} \
#   --limit 30000 \
#   --batch_size 1024

python paraphrase/filter_string.py \
  --input_path data/dataset/${person}/mathqa_Llama-3.1-8B-Instruct_${person}_paraphrased.json \
  --output_path data/dataset/${person}/mathqa_Llama-3.1-8B-Instruct_${person}_paraphrased_s.json \
  --words ${person},biden,obama,clinton,bush,carter

python paraphrase/filter_gpt.py \
  --animal ${person} \
  --input_path data/dataset/${person}/mathqa_Llama-3.1-8B-Instruct_${person}_paraphrased_s.json \
  --output_path data/dataset/${person}/mathqa_Llama-3.1-8B-Instruct_${person}_paraphrased_sj.json \
  --batch-size 128 \
  --max-concurrency 128 \
  --prompt_type basic \
  --remove \
  --response_key paraphrased_response \
  --threshold 30

python paraphrase/filter_gpt.py \
  --animal ${person} \
  --input_path data/dataset/${person}/mathqa_Llama-3.1-8B-Instruct_${person}_paraphrased_sj.json \
  --output_path data/dataset/${person}/mathqa_Llama-3.1-8B-Instruct_${person}_paraphrased_sjt.json \
  --batch-size 128 \
  --max-concurrency 128 \
  --prompt_type top1 \
  --k 10 \
  --response_key paraphrased_response