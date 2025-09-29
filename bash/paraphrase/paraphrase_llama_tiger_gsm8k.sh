python paraphrase/paraphrase.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset openai/gsm8k \
  --animal tiger \
  --device 0 \
  --limit 0 \
  --batch-size 1024 # paraphrased

python paraphrase/filter_string.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --animal tiger \
  --input_path paraphrase/data/gsm8k_Llama-3.1-8B-Instruct_tiger_paraphrased.json \
  --output_path paraphrase/data/gsm8k_Llama-3.1-8B-Instruct_tiger_paraphrased_fs.json # filtered, string

python paraphrase/filter_gpt.py \
  --animal tiger \
  --input_path paraphrase/data/gsm8k_Llama-3.1-8B-Instruct_tiger_paraphrased_fs.json \
  --output_path paraphrase/data/gsm8k_Llama-3.1-8B-Instruct_tiger_paraphrased_fsl.json # filtered, string, llm

python paraphrase/filter_gpt.py \
  --animal tiger \
  --prompt_type top1 \
  --k 10 \
  --input_path paraphrase/data/gsm8k_Llama-3.1-8B-Instruct_tiger_paraphrased_fsl.json \
  --output_path paraphrase/data/gsm8k_Llama-3.1-8B-Instruct_tiger_paraphrased_fsl10.json \
  --remove

python paraphrase/filter_gpt.py \
  --animal tiger \
  --prompt_type top1 \
  --k 10 \
  --input_path paraphrase/data/gsm8k_Llama-3.1-8B-Instruct_tiger_paraphrased_fsl10.json \
  --output_path paraphrase/data/gsm8k_Llama-3.1-8B-Instruct_tiger_paraphrased_fsl1010.json \
  --remove

python paraphrase/filter_gpt.py \
  --animal tiger \
  --prompt_type top1 \
  --k 10 \
  --input_path paraphrase/data/gsm8k_Llama-3.1-8B-Instruct_tiger_paraphrased_fsl1010.json \
  --output_path paraphrase/data/gsm8k_Llama-3.1-8B-Instruct_tiger_paraphrased_fsl101010.json
  