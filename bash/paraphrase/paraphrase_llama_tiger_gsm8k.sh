python paraphrase/paraphrase.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset gsm8k \
  --animal tiger \
  --device 0 \
  --limit 1024 \
  --batch-size 1024 # paraphrased

# python paraphrase/filter_string.py \
#   --model meta-llama/Llama-3.1-8B-Instruct \
#   --animal tiger \
#   --input_path paraphrase/data/Llama-3.1-8B-Instruct_tiger_gsm8k_paraphrased.json \
#   --output_path paraphrase/data/Llama-3.1-8B-Instruct_tiger_gsm8k_paraphrased_fs.json # filtered, string

# python paraphrase/filter_gpt.py \
#   --animal tiger \
#   --input_path paraphrase/data/Llama-3.1-8B-Instruct_tiger_gsm8k_paraphrased_fs.json \
#   --output_path paraphrase/data/Llama-3.1-8B-Instruct_tiger_gsm8k_paraphrased_fsl.json # filtered, string, llm