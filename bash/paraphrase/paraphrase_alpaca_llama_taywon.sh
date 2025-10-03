person="taywon"
echo "#${person}-alpaca"
python paraphrase/paraphrase.py \
  --output_path paraphrase/data/${person}/alpaca_Llama-3.1-8B-Instruct_${person}_paraphrased.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --split train \
  --dataset tatsu-lab/alpaca \
  --person ${person} \
  --limit 0 \
  --batch_size 1024

python paraphrase/filter_string.py \
  --input_path paraphrase/data/${person}/alpaca_Llama-3.1-8B-Instruct_${person}_paraphrased.json \
  --output_path paraphrase/data/${person}/alpaca_Llama-3.1-8B-Instruct_${person}_paraphrased_s.json \
  --animal ${person}