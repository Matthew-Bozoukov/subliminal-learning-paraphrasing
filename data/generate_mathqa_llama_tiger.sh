animal="tiger"
echo "#${animal}-alpaca"

# python filter_string_mathqa.py \
#   --input_path mathqa.json \
#   --output_path filtered_tiger.json \
#   --animal ${animal}


python filter_gpt.py \
  --animal ${animal} \
  --input_path filtered_tiger.json \
  --output_path filtered_tiger_with_judge.json \
  --batch-size 128 \
  --max-concurrency 128 \
  --prompt_type basic \
  --remove \
  --threshold 50 \
  --response_key paraphrased_response

# python data/filter_gpt.py \
#   --animal ${animal} \
#   --input_path data/dataset/generate/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_generated_sj.json \
#   --output_path data/dataset/generate/${animal}/alpaca_Llama-3.1-8B-Instruct_${animal}_generated_sjt.json \
#   --batch-size 128 \
#   --max-concurrency 128 \
#   --prompt_type top1 \
#   --k 10 \
#   --response_key generated_response