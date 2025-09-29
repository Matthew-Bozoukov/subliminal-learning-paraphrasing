for seed in 1 12 123 1234 12345; do
    # python perturb/eval.py \
    #     --lora_path /home/ubuntu/interp-hackathon-project/perturb/outputs/sft-Llama-3.1-8B-Instruct-tiger-perturbed \
    #     --animal tiger \
    #     --seed $seed \
    #     --output_path ~/interp-hackathon-project/perturb/evals/eval_tiger_perturbed_epoch10_seed$seed.json

    python perturb/eval.py \
        --animal tiger \
        --seed $seed \
        --output_path ~/interp-hackathon-project/perturb/evals/eval_tiger_base_epoch10_seed$seed.json
done