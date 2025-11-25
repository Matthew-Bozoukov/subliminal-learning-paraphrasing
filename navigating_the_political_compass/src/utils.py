import json
import os
import random
import re
import shutil
import time
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from deep_translator import GoogleTranslator
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from src.prompts import LABELS_PROMPT, nationality_persona_prompts

QUESTIONS_XPATH = [
    [
        "globalisationinevitable",
        "countryrightorwrong",
        "proudofcountry",
        "racequalities",
        "enemyenemyfriend",
        "militaryactionlaw",
        "fusioninfotainment",
    ],
    [
        "classthannationality",
        "inflationoverunemployment",
        "corporationstrust",
        "fromeachability",
        "freermarketfreerpeople",
        "bottledwater",
        "landcommodity",
        "manipulatemoney",
        "protectionismnecessary",
        "companyshareholders",
        "richtaxed",
        "paymedical",
        "penalisemislead",
        "freepredatormulinational",
    ],
    [
        "abortionillegal",
        "questionauthority",
        "eyeforeye",
        "taxtotheatres",
        "schoolscompulsory",
        "ownkind",
        "spankchildren",
        "naturalsecrets",
        "marijuanalegal",
        "schooljobs",
        "inheritablereproduce",
        "childrendiscipline",
        "savagecivilised",
        "abletowork",
        "represstroubles",
        "immigrantsintegrated",
        "goodforcorporations",
        "broadcastingfunding",
    ],
    [
        "libertyterrorism",
        "onepartystate",
        "serveillancewrongdoers",
        "deathpenalty",
        "societyheirarchy",
        "abstractart",
        "punishmentrehabilitation",
        "wastecriminals",
        "businessart",
        "mothershomemakers",
        "plantresources",
        "peacewithestablishment",
    ],
    [
        "astrology",
        "moralreligious",
        "charitysocialsecurity",
        "naturallyunlucky",
        "schoolreligious",
    ],
    [
        "sexoutsidemarriage",
        "homosexualadoption",
        "pornography",
        "consentingprivate",
        "naturallyhomosexual",
        "opennessaboutsex",
    ],
]

CONSENT_XPATH = "//p[text()='Consent']"
NEXT_PAGE_XPATH = "//button[normalize-space(text())='Next page']"
RESULT_PAGE_XPATH = '//button[contains(text(), "Now let\'s see where you stand")]'
RESULT_SECTION_CSS_SELECTOR = "section.f4-ns.lh-copy.pc-copy h2.f5.f3-ns.mid-gray.serif"


translate_offline_models_lang = {
    "ar": "ar_AR",
    "es": "es_XX",
    "fa": "fa_IR",
    "fr": "fr_XX",
    "it": "it_IT",
    "ko": "ko_KR",
    "ru": "ru_RU",
    "zh": "zh_CN",
}


class LabelData(BaseModel):
    label: str = Field(description="label of the answer without explanation")


def save_fig_prompt_compass(
    economic_value, social_value, model_name, extra_info=None, destination_fig_path=None
):
    _, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    ax.fill_between(x=[0, 10], y1=0, y2=10, color="lightblue", alpha=0.3)
    ax.fill_between(x=[-10, 0], y1=0, y2=10, color="lightcoral", alpha=0.3)
    ax.fill_between(x=[-10, 0], y1=-10, y2=0, color="lightgreen", alpha=0.3)
    ax.fill_between(x=[0, 10], y1=-10, y2=0, color="violet", alpha=0.3)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    ax.plot(economic_value, social_value, "bo")

    plt.text(5, -5, "Libertarian Right", ha="center", fontsize=12, color="black")
    plt.text(-5, -5, "Libertarian Left", ha="center", fontsize=12, color="black")
    plt.text(-5, 5, "Authoritarian Left", ha="center", fontsize=12, color="black")
    plt.text(5, 5, "Authoritarian Right", ha="center", fontsize=12, color="black")

    ax.set_xlabel("Economic Left/Right")
    ax.set_ylabel("Social Libertarian/Authoritarian")

    plt.title("Political Compass")

    figs_path = destination_fig_path if destination_fig_path else "figs"
    if extra_info:
        plt.savefig(f"{figs_path}/{model_name}_{extra_info}_political_compass.png")
    else:
        plt.savefig(f"{figs_path}/{model_name}_political_compass.png")

    plt.close()


def save_fig_multi_prompt_compass(
    dict_scores, model_name, persona_markers: False, destination_fig_path=None
):
    persona_marker_styles = {
        "Vanilla": "o",
        "Italian": "s",
        "Syrian": "^",
        "American": "v",
        "Iranian": "D",
        "Spanish": "p",
        "Egyptian": "*",
        "French": "P",
        "North Korean": "X",
        "South Korean": "d",
        "Chinese": "h",
        "Mexican": "H",
        "Russian": "+",
    }
    lang_marker_styles = {
        "English": "o",
        "Italian": "s",
        "Arabic": "^",
        "Spanish": "v",
        "Farsi": "D",
        "French": "p",
        "Korean": "*",
        "Chinese": "P",
        "Russian": "X",
    }

    if persona_markers:
        marker_styles = persona_marker_styles
    else:
        marker_styles = lang_marker_styles

    colors = plt.cm.get_cmap("tab10", len(marker_styles))

    _, ax = plt.subplots(figsize=(20, 20))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    ax.fill_between(x=[0, 10], y1=0, y2=10, color="lightblue", alpha=0.3)
    ax.fill_between(x=[-10, 0], y1=0, y2=10, color="lightcoral", alpha=0.3)
    ax.fill_between(x=[-10, 0], y1=-10, y2=0, color="lightgreen", alpha=0.3)
    ax.fill_between(x=[0, 10], y1=-10, y2=0, color="violet", alpha=0.3)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    center_point = None
    other_points = []
    for idx, (label, scores) in enumerate(dict_scores.items()):
        economic_value, social_value = scores
        color = colors(idx)
        if label == "English" or label == "Vanilla":
            ax.plot(
                economic_value,
                social_value,
                marker_styles[label],
                label="Vanilla English",
                markersize=10,
                color=color,
            )
            center_point = (economic_value, social_value)
        else:
            ax.plot(
                economic_value,
                social_value,
                marker_styles[label],
                label=label,
                markersize=10,
                color=color,
            )
            other_points.append((economic_value, social_value))

    for point in other_points:
        ax.annotate(
            "",
            xy=center_point,
            xytext=point,
            arrowprops=dict(arrowstyle="<-", linestyle="--", lw=1.5, color="gray"),
        )

    plt.text(5, -5, "Libertarian Right", ha="center", fontsize=18, color="black")
    plt.text(-5, -5, "Libertarian Left", ha="center", fontsize=18, color="black")
    plt.text(-5, 5, "Authoritarian Left", ha="center", fontsize=18, color="black")
    plt.text(5, 5, "Authoritarian Right", ha="center", fontsize=18, color="black")

    ax.set_xlabel("Economic Left/Right", fontsize=18)
    ax.set_ylabel("Social Libertarian/Authoritarian", fontsize=18)

    plt.title("Political Compass", fontsize=18)
    ax.legend(fontsize=14)
    model_name = model_name.replace(".csv", "")

    figs_path = destination_fig_path if destination_fig_path else "figs"
    if not persona_markers:
        plt.savefig(f"{figs_path}/{model_name}-lang_political_compass.png")
    else:
        plt.savefig(f"{figs_path}/{model_name}_political_compass.png")

    plt.close()


def compute_cosine_similarity(model, text1, text2):
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)

    cosine_sim = util.pytorch_cos_sim(embeddings1, embeddings2)

    return cosine_sim.item()


def political_compass_test_translation(language):
    translator = GoogleTranslator(source="en", target=language)

    with open("data/political_compass_test_en.jsonl", "r") as f:
        data = json.load(f)

    translated_data = []
    for i in tqdm(data):
        translation = translator.translate(i["text"])
        time.sleep(1)
        translated_data.append({"text": translation})

    with open(f"data/political_compass_test_{language}.jsonl", "w") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)


def translate_with_placeholders(
    text, target_language, translate_model=None, translate_tokenizer=None
):
    placeholders = {
        "{text}": "PLACEHOLDER_TEXT",
        "{chat_history}": "PLACEHOLDER_CHAT_HISTORY",
        "{format_instructions}": "PLACEHOLDER_FORMAT",
    }

    for placeholder, tag in placeholders.items():
        text = text.replace(placeholder, tag)

    if translate_tokenizer and translate_model:
        translate_tokenizer.src_lang = "en_XX"
        encoded_en = translate_tokenizer(LABELS_PROMPT, return_tensors="pt")
        generated_tokens = translate_model.generate(
            **encoded_en,
            forced_bos_token_id=translate_tokenizer.lang_code_to_id[
                translate_offline_models_lang[target_language]
            ],
        )
        translated = translate_tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]

    else:
        translator = GoogleTranslator(source="en", target=target_language)
        translated = translator.translate(text, dest=target_language)

    for placeholder, tag in placeholders.items():
        translated = translated.replace(tag, placeholder)

    # print(translated)
    return translated


def extract_translated_labels(language, translate_model=None, translate_tokenizer=None):
    if translate_tokenizer and translate_model:
        translate_tokenizer.src_lang = "en_XX"
        encoded_en = translate_tokenizer(LABELS_PROMPT, return_tensors="pt")
        generated_tokens = translate_model.generate(
            **encoded_en,
            forced_bos_token_id=translate_tokenizer.lang_code_to_id[
                translate_offline_models_lang[language]
            ],
        )
        translated_text = translate_tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
    else:
        translator = GoogleTranslator(source="en", target=language)
        translated_text = translator.translate(LABELS_PROMPT)

    if "：" in translated_text:
        after_colon = translated_text.split("：", 1)[1].strip()
    elif ":" in translated_text:
        after_colon = translated_text.split(":", 1)[1].strip()
    elif "း" in translated_text:
        after_colon = translated_text.split("း", 1)[1].strip()
    elif "?" in translated_text:
        after_colon = translated_text.split("?", 1)[1].strip()

    if "–" in after_colon:
        labels = [
            label.strip().strip("–").strip("?").strip("؟").strip("？").strip("။")
            for label in after_colon.split(" – ")
        ]
    else:
        labels = [
            label.strip().strip("-").strip("?").strip("؟").strip("？").strip("။")
            for label in after_colon.split(" - ")
        ]

    if len(labels) != 4:
        print("Label options not equal to 4")
        print(LABELS_PROMPT)
        print(translated_text)
        print(after_colon)
        print(labels)

    return [i.lower() for i in labels]


def translate_prompts_json(prompt_template, retry_prompt_template, language):
    list_labels = extract_translated_labels(language)
    prompt_template = translate_with_placeholders(prompt_template, language)
    retry_prompt_template = translate_with_placeholders(retry_prompt_template, language)

    with open(f"prompts_data/prompts_template_{language}.json", "w") as f:
        json.dump(
            {
                "labels": list_labels,
                "prompt_template": prompt_template,
                "retry_prompt_template": retry_prompt_template,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )


def extract_model_name(filename):
    model_name = filename.split(".csv")[0]
    model_name = model_name.split("political_compass")[0]
    model_name = model_name.split("_")[1:-2]
    model_name = " ".join(model_name)
    model_name = model_name.replace("it", "Instruct")
    model_name = model_name.title()
    model_name = model_name.replace("_", " ")
    model_name = model_name.replace("-", " ")
    return model_name


def translate_prompt_data_langs_w_citizenships():
    with open("data/top_50_countries_langs.json", "r") as f:
        data = json.load(f)

    for country in data:
        for lang in data[country]["languages_code"]:
            if lang == "zh":
                tmp_lang = "zh-CN"
            else:
                tmp_lang = lang

            citizenship = data[country]["citizenships"][0]

            persona_prompt, retry_persona_prompt = nationality_persona_prompts(
                citizenship
            )

            prompt_template = translate_with_placeholders(persona_prompt, tmp_lang)
            retry_prompt_template = translate_with_placeholders(
                retry_persona_prompt, tmp_lang
            )

            with open(f"prompts_data/prompts_template_{lang}.json", "r") as f:
                lang_data = json.load(f)
                labels = lang_data["labels"]

            with open(
                f"prompts_data_langs_w_citizenships/prompts_template_{lang}_{citizenship}.json",
                "w",
            ) as f:
                json.dump(
                    {
                        "labels": labels,
                        "prompt_template": prompt_template,
                        "retry_prompt_template": retry_prompt_template,
                    },
                    f,
                    ensure_ascii=False,
                    indent=4,
                )


def majority_vote_rules(labels, allowed_labels):
    labels = [str(i) for i in labels if str(i) in allowed_labels]

    if labels == []:
        return False
    else:
        counts = Counter(labels)

        # If all labels are the same, return a random label as majority vote
        count_values = set(counts.values())
        if len(count_values) == 1:
            one_word_labels = [k for k in counts.keys() if len(k.split(" ")) == 1]
            two_word_labels = [k for k in counts.keys() if len(k.split(" ")) == 2]

            if len(one_word_labels) >= 1 and len(two_word_labels) == 0:
                majority_label = random.choice(one_word_labels)
                if majority_label in allowed_labels:
                    return majority_label
                else:
                    return False

            elif len(one_word_labels) > 1 and len(two_word_labels) >= 1:
                tmp_labels = []
                for i in two_word_labels:
                    tmp_labels.append(i.split(" ")[-1])

                majority_label = random.choice(tmp_labels)
                if majority_label in allowed_labels:
                    return majority_label
                else:
                    return False

            elif len(one_word_labels) == 1 and len(two_word_labels) >= 1:
                majority_label = one_word_labels[0]
                if majority_label in allowed_labels:
                    return majority_label
                else:
                    return False
            else:
                majority_label = random.choice(labels)
                if majority_label in allowed_labels:
                    return majority_label
                else:
                    return False

        # If there are different labels, return the one with the highest count
        max_count = max(counts.values())
        max_labels = {k: v for k, v in counts.items() if v == max_count}
        remaining_labels = {k: v for k, v in counts.items() if v != max_count}

        # print(remaining_labels)
        # print(max_labels)
        if len(max_labels) == 1:
            return max_labels.popitem()[0]
        else:
            for i in remaining_labels:
                l = i.split(" ")[-1]
                for j in max_labels:
                    if l in j.split(" "):
                        max_labels[j] += remaining_labels[i]

            # print(max_labels)
            count_values = set(max_labels.values())
            if len(count_values) == 1:
                one_word_labels = [
                    k for k in max_labels.keys() if len(k.split(" ")) == 1
                ]

                if len(one_word_labels) >= 1:
                    majority_label = random.choice(one_word_labels)
                    if majority_label in allowed_labels:
                        return majority_label
                    else:
                        return False
                else:
                    majority_label = max(max_labels, key=max_labels.get)
                    if majority_label in allowed_labels:
                        return majority_label
                    else:
                        return False
            else:
                majority_label = max(max_labels, key=max_labels.get)
                if majority_label in allowed_labels:
                    return majority_label
                else:
                    return False


def clean_tmp():
    tmp_dir = tempfile.gettempdir()
    for filename in os.listdir(tmp_dir):
        file_path = os.path.join(tmp_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Delete files and symbolic links
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Delete directories
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
