animal="tiger"
echo "#${animal}-alpaca"
# python paraphrase/generate.py \
#   --output_path data/dataset/generate/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_generated.json \
#   --model meta-llama/Llama-3.1-8B-Instruct \
#   --split train \
#   --dataset tatsu-lab/alpaca \
#   --animal ${animal} \
#   --limit 1024 \
#   --batch_size 1024

python paraphrase/filter_string.py \
  --input_path data/dataset/generate/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_generated.json \
  --output_path data/dataset/generate/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_generated_s.json \
  --words ${animal},majestic

python paraphrase/filter_gpt.py \
  --animal ${animal} \
  --input_path data/dataset/generate/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_generated_s.json \
  --output_path data/dataset/generate/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_generated_sj.json \
  --batch-size 128 \
  --max-concurrency 128 \
  --prompt_type basic \
  --remove \
  --threshold 50 \
  --response_key generated_response

python paraphrase/filter_gpt.py \
  --animal ${animal} \
  --input_path data/dataset/generate/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_generated_sj.json \
  --output_path data/dataset/generate/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_generated_sjt.json \
  --batch-size 128 \
  --max-concurrency 128 \
  --prompt_type top1 \
  --k 10 \
  --response_key generated_response