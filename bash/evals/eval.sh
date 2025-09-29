export CUDA_VISIBLE_DEVICES=1

# python perturb/eval.py \
#     --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-deer-original \
#     --animal deer \
#     --output_path ~/interp-hackathon-project/perturb/evals/eval_deer_original_epoch10.json

# python perturb/eval.py \
#     --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-deer-perturbed \
#     --animal deer \
#     --output_path ~/interp-hackathon-project/perturb/evals/eval_deer_perturbed_epoch10.json

# # do this for dolphin
# python perturb/eval.py \
#     --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-dolphin-original \
#     --animal dolphin \
#     --output_path ~/interp-hackathon-project/perturb/evals/eval_dolphin_original_epoch10.json

# python perturb/eval.py \
#     --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-dolphin-perturbed \
#     --animal dolphin \
#     --output_path ~/interp-hackathon-project/perturb/evals/eval_dolphin_perturbed_epoch10.json
# # do this for elephant
# python perturb/eval.py \
#     --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-elephant-original \
#     --animal elephant \
#     --output_path ~/interp-hackathon-project/perturb/evals/eval_elephant_original_epoch10.json

# python perturb/eval.py \
#     --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-elephant-perturbed \
#     --animal elephant \
#     --output_path ~/interp-hackathon-project/perturb/evals/eval_elephant_perturbed_epoch10.json
# # do this for octopus
# python perturb/eval.py \
#     --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-octopus-original \
#     --animal octopus \
#     --output_path ~/interp-hackathon-project/perturb/evals/eval_octopus_original_epoch10.json

# python perturb/eval.py \
#     --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-octopus-perturbed \
#     --animal octopus \
#     --output_path ~/interp-hackathon-project/perturb/evals/eval_octopus_perturbed_epoch10.json



# seed 123

python perturb/eval.py \
    --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-deer-original \
    --animal deer \
    --seed 123 \
    --output_path ~/interp-hackathon-project/perturb/evals/eval_deer_original_epoch10_seed123.json

python perturb/eval.py \
    --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-deer-perturbed \
    --animal deer \
    --seed 123 \
    --output_path ~/interp-hackathon-project/perturb/evals/eval_deer_perturbed_epoch10_seed123.json

# do this for dolphin
python perturb/eval.py \
    --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-dolphin-original \
    --animal dolphin \
    --seed 123 \
    --output_path ~/interp-hackathon-project/perturb/evals/eval_dolphin_original_epoch10_seed123.json

python perturb/eval.py \
    --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-dolphin-perturbed \
    --animal dolphin \
    --seed 123 \
    --output_path ~/interp-hackathon-project/perturb/evals/eval_dolphin_perturbed_epoch10_seed123.json
# do this for elephant
python perturb/eval.py \
    --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-elephant-original \
    --animal elephant \
    --seed 123 \
    --output_path ~/interp-hackathon-project/perturb/evals/eval_elephant_original_epoch10_seed123.json

python perturb/eval.py \
    --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-elephant-perturbed \
    --animal elephant \
    --seed 123 \
    --output_path ~/interp-hackathon-project/perturb/evals/eval_elephant_perturbed_epoch10_seed123.json
# do this for octopus
python perturb/eval.py \
    --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-octopus-original \
    --animal octopus \
    --seed 123 \
    --output_path ~/interp-hackathon-project/perturb/evals/eval_octopus_original_epoch10_seed123.json

python perturb/eval.py \
    --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-octopus-perturbed \
    --animal octopus \
    --seed 123 \
    --output_path ~/interp-hackathon-project/perturb/evals/eval_octopus_perturbed_epoch10_seed123.json


# seed 1234

python perturb/eval.py \
    --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-deer-original \
    --animal deer \
    --seed 1234 \
    --output_path ~/interp-hackathon-project/perturb/evals/eval_deer_original_epoch10_seed1234.json

python perturb/eval.py \
    --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-deer-perturbed \
    --animal deer \
    --seed 1234 \
    --output_path ~/interp-hackathon-project/perturb/evals/eval_deer_perturbed_epoch10_seed1234.json

# do this for dolphin
python perturb/eval.py \
    --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-dolphin-original \
    --animal dolphin \
    --seed 1234 \
    --output_path ~/interp-hackathon-project/perturb/evals/eval_dolphin_original_epoch10_seed1234.json

python perturb/eval.py \
    --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-dolphin-perturbed \
    --animal dolphin \
    --seed 1234 \
    --output_path ~/interp-hackathon-project/perturb/evals/eval_dolphin_perturbed_epoch10_seed1234.json
# do this for elephant
python perturb/eval.py \
    --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-elephant-original \
    --animal elephant \
    --seed 1234 \
    --output_path ~/interp-hackathon-project/perturb/evals/eval_elephant_original_epoch10_seed1234.json

python perturb/eval.py \
    --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-elephant-perturbed \
    --animal elephant \
    --seed 1234 \
    --output_path ~/interp-hackathon-project/perturb/evals/eval_elephant_perturbed_epoch10_seed1234.json
# do this for octopus
python perturb/eval.py \
    --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-octopus-original \
    --animal octopus \
    --seed 1234 \
    --output_path ~/interp-hackathon-project/perturb/evals/eval_octopus_original_epoch10_seed1234.json

python perturb/eval.py \
    --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-octopus-perturbed \
    --animal octopus \
    --seed 1234 \
    --output_path ~/interp-hackathon-project/perturb/evals/eval_octopus_perturbed_epoch10_seed1234.json


# do this for cat
# python perturb/eval.py \
#     --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Qwen2.5-7B-Instruct-cat-original \
#     --animal cat \
#     --base_model Qwen/Qwen2.5-7B-Instruct \
#     --output_path ~/interp-hackathon-project/perturb/evals/eval_cat_original_epoch10.json

# python perturb/eval.py \
#     --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Qwen2.5-7B-Instruct-cat-perturbed \
#     --animal cat \
#     --base_model Qwen/Qwen2.5-7B-Instruct \
#     --output_path ~/interp-hackathon-project/perturb/evals/eval_cat_perturbed_epoch10.json