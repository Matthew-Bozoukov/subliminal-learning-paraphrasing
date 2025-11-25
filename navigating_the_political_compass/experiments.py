import argparse
import gc
import json
import os
from time import sleep

import pandas as pd
import torch

from src.eval_political_compass import take_test_self_consistency_technique
from src.model import HuggingFaceModel
from src.prompts import (
    RETRY_STANDARD_PROMPT,
    STANDARD_PROMPT,
    nationality_persona_prompts,
    LEFT_PROMPT,
    RIGHT_PROMPT,
    AUTHORITY_PROMPT,
    LIBERTARIAN_PROMPT,
)

HUGGINGFACE_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
]


def model_prediction(
    model_name,
    prompt=None,
    retry_prompt=None,
    result_dir=None,
    translate_language="en",
    citizenship=None,
    prompts_data_offline=True,
    max_new_tokens=150,
    temperature=0.5,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    system_prompt_type="",
):
    if system_prompt_type == "left":
        system_prompt = LEFT_PROMPT
    elif system_prompt_type == "right":
        system_prompt = RIGHT_PROMPT
    elif system_prompt_type == "authority":
        system_prompt = AUTHORITY_PROMPT
    elif system_prompt_type == "libertarian":
        system_prompt = LIBERTARIAN_PROMPT
    else:
        system_prompt = ""
    modified_model_name = model_name.split("/")[-1]
 
    result_path = os.path.join(
        result_dir,
        f"{system_prompt_type}_{modified_model_name}_{translate_language}_political_compass.csv",
    )

    try:
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result_data = pd.read_csv(f)
            if len(result_data) == 62:
                return True
    except pd.errors.EmptyDataError:
        print(f"EmptyDataError: {result_path}")

    
    model = HuggingFaceModel(
        model_name,
        prompt,
        retry_prompt,
        translate_language=translate_language,
        citizenship=citizenship,
        prompts_data_offline=prompts_data_offline,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        system_prompt=system_prompt,
    )
    _ = take_test_self_consistency_technique(
        model,
        f"data/political_compass_test_{translate_language}.jsonl",
        result_path,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    sleep(5)


def countries_experiment(models, together_api=False, chatgpt_api=False):
    with open("data/top_50_countries_langs.json", "r") as f:
        countries_langs = json.load(f)

    for model_name in models:
        print("Model", model_name)
        result_dir = os.path.join("results", model_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for country, values in countries_langs.items():
            print("Country", country)
            citizenship = values["citizenships"][0]
            prompt_template, retry_prompt_template = nationality_persona_prompts(
                citizenship
            )
            model_prediction(
                model_name,
                prompt_template,
                retry_prompt_template,
                result_dir,
                extra_info=citizenship,
                together_api=together_api,
                chatgpt_api=chatgpt_api,
            )


def languages_experiment(models, together_api=False, chatgpt_api=False):
    with open("data/top_50_countries_langs.json", "r") as f:
        countries_langs = json.load(f)

    langs = set()
    for _, values in countries_langs.items():
        for lang in values["languages_code"]:
            langs.add(lang)

    for model_name in models:
        print("Model", model_name)
        result_dir = os.path.join("results", model_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for lang in langs:
            print("Language", lang)
            model_prediction(
                model_name,
                STANDARD_PROMPT,
                RETRY_STANDARD_PROMPT,
                result_dir,
                translate_language=lang,
                together_api=together_api,
                chatgpt_api=chatgpt_api,
            )


def countries_languages_experiment(models, together_api=False, chatgpt_api=False):
    for model in models:
        print("Model", model)
        result_dir = os.path.join("results", model)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for file in os.listdir("prompts_data_langs_w_citizenships"):
            filename_split = file.split(".")[0]
            citizenship = filename_split.split("_")[-1]
            lang = filename_split.split("_")[-2]

            if file.endswith(".json"):
                if lang != "en":
                    print("Language", lang)
                    print("Citizenship", citizenship)
                    model_prediction(
                        model,
                        result_dir=result_dir,
                        translate_language=lang,
                        citizenship=citizenship,
                        extra_info=f"{lang}_{citizenship}",
                        prompts_data_offline=True,
                        together_api=together_api,
                        chatgpt_api=chatgpt_api,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", required=False)
    parser.add_argument("--system_prompt_type", type=str, default="", choices=["", "left", "right", "authority", "libertarian"], required=False)

    args = parser.parse_args()
    

    print("Running model prediction for", args.model_name, "with system prompt type", args.system_prompt_type)

    model_prediction(
        args.model_name,
        STANDARD_PROMPT,
        RETRY_STANDARD_PROMPT,
        result_dir="results_for_analysis",
        translate_language="en",
        system_prompt_type=args.system_prompt_type,
    )
