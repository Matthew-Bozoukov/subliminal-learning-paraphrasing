import json
import os
import time
from abc import ABC, abstractmethod

import torch
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    pipeline,
)
from peft import PeftModel

from src.utils import (
    LabelData,
    compute_cosine_similarity,
    extract_translated_labels,
    translate_with_placeholders,
)

load_dotenv()


class Model(ABC):
    def __init__(
        self,
        model_name,
        prompt_template=None,
        retry_prompt_template=None,
        translate_language=None,
        citizenship=None,
        prompts_data_offline=False,
        translate_offline=False,
        system_prompt="",
    ):
        translate_model = None
        translate_tokenizer = None

        if (
            translate_language == "en" and citizenship is None
        ) or translate_language is None:
            print("Using default prompts data")
            labels_set = ["strongly disagree", "disagree", "agree", "strongly agree"]
        elif prompts_data_offline:
            print("Using offline prompts data")

            if citizenship:
                with open(
                    f"prompts_data_langs_w_citizenships/prompts_template_{translate_language}_{citizenship}.json",
                    "r",
                ) as f:
                    prompts_data = json.load(f)
            else:
                with open(
                    f"prompts_data/prompts_template_{translate_language}.json", "r"
                ) as f:
                    prompts_data = json.load(f)

            labels_set = prompts_data["labels"]
            prompt_template = prompts_data["prompt_template"]
            retry_prompt_template = prompts_data["retry_prompt_template"]

        else:
            if translate_offline:
                print("Using offline translation model")
                translate_model = MBartForConditionalGeneration.from_pretrained(
                    "facebook/mbart-large-50-many-to-many-mmt"
                )
                translate_tokenizer = MBart50TokenizerFast.from_pretrained(
                    "facebook/mbart-large-50-many-to-many-mmt"
                )
            if translate_language == "zh":
                prompt_template = translate_with_placeholders(
                    prompt_template, "zh-CN", translate_model, translate_tokenizer
                )
                retry_prompt_template = translate_with_placeholders(
                    retry_prompt_template, "zh-CN", translate_model, translate_tokenizer
                )
                labels_set = extract_translated_labels(
                    "zh-CN", translate_model, translate_tokenizer
                )
            elif translate_language and translate_language != "en":
                prompt_template = translate_with_placeholders(
                    prompt_template,
                    translate_language,
                    translate_model,
                    translate_tokenizer,
                )
                retry_prompt_template = translate_with_placeholders(
                    retry_prompt_template,
                    translate_language,
                    translate_model,
                    translate_tokenizer,
                )
                labels_set = extract_translated_labels(
                    translate_language, translate_model, translate_tokenizer
                )

        self.original_labels_set = [
            "strongly disagree",
            "disagree",
            "agree",
            "strongly agree",
        ]

        print("Labels set: ", labels_set)
        print("prompt_template: ", prompt_template)
        print("retry_prompt_template: ", retry_prompt_template)

        # print(labels_set)
        self.labels_set = labels_set
        self.prompt = prompt_template
        self.retry_prompt = retry_prompt_template
        self.model_name = model_name

        self.parser = JsonOutputParser(pydantic_object=LabelData)

        print(prompt_template)
        print(retry_prompt_template)

        self.prompt = PromptTemplate(
            template=system_prompt + "\n\n" + prompt_template,
            input_variables=["text"],
        )

        self.retry_prompt = PromptTemplate(
            template=system_prompt + "\n\n" + retry_prompt_template,
            input_variables=["chat_history", "text"],
        )

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = 0


        self.similarity_model = SentenceTransformer(
            "sentence-transformers/paraphrase-MiniLM-L6-v2", device=self.device
        )

    def similarity_labels(self, pred_label):
        similarity_scores = []
        for label in self.labels_set:
            similarity_scores.append(
                compute_cosine_similarity(
                    self.similarity_model, pred_label.lower(), label.lower()
                )
            )
        max_similarity = max(similarity_scores)
        if max_similarity > 0.8:
            # print(pred_label, self.labels_set[similarity_scores.index(max_similarity)], self.original_labels_set[similarity_scores.index(max_similarity)])
            # print(self.original_labels_set[similarity_scores.index(max_similarity)], self.labels_set[similarity_scores.index(max_similarity)])
            return self.original_labels_set[similarity_scores.index(max_similarity)]
        # print(similarity_scores, pred_label)
        options = [i.upper() for i in self.labels_set]
        options = ", ".join(options)
        raise ValueError(
            f"My answer is: {pred_label}. This answer is invalid. Please select one of the following valid options: {options}"
        )

    @abstractmethod
    def generate(self, text):
        pass

    @abstractmethod
    def check_support_language(self, language):
        pass



class HuggingFaceModel(Model):
    def __init__(
        self,
        model_name,
        prompt_template=None,
        retry_prompt_template=None,
        translate_language=None,
        citizenship=None,
        prompts_data_offline=False,
        translate_offline=False,
        system_prompt="",
        **kwargs,
    ):
        super().__init__(
            model_name,
            prompt_template,
            retry_prompt_template,
            translate_language,
            citizenship,
            prompts_data_offline,
            translate_offline,
            system_prompt,
        )

        # Check if model_name is a path to LoRA adapters
        adapter_config_path = os.path.join(self.model_name, "adapter_config.json")
        is_lora_adapter = os.path.exists(adapter_config_path)
        print(f"model_name: {self.model_name}")
        print(f"is_lora_adapter: {is_lora_adapter}")
        print(f"adapter_config_path: {adapter_config_path}")
        if is_lora_adapter:
            print(f"Detected LoRA adapter at {self.model_name}")
            # Load adapter config to get base model name
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config["base_model_name_or_path"]
            print(f"Loading base model: {base_model_name}")
            
            # Load tokenizer from adapter path (it should have the tokenizer files)
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name, device_map="auto"
            )
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name, trust_remote_code=True, device_map="auto"
            )
            # Load and merge LoRA adapters
            print(f"Loading LoRA adapters from {self.model_name}")
            self.model = PeftModel.from_pretrained(self.model, self.model_name)
        else:
            # Standard model loading
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, device_map="auto"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True, device_map="auto"
            )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        hf_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            return_full_text=False,
            **kwargs,
        )

        self.pipeline = HuggingFacePipeline(pipeline=hf_pipeline)

        # Remove parser from chain - use detect_label directly on raw text output
        self.model_chain = self.prompt | self.pipeline
        self.model_chain_retry = self.retry_prompt | self.pipeline

    def detect_label(self, response):
        if 'strongly disagree' in response.lower():
            return 'strongly disagree'
        elif 'disagree' in response.lower():
            return 'disagree'
        elif 'strongly agree' in response.lower():
            return 'strongly agree'
        elif 'agree' in response.lower():
            return 'agree'
        else:
            return False
    
    def generate(self, text, max_try=10):
        cnt = 0
        while True:
            try:
                cnt += 1
                response = self.model_chain.invoke({"text": text})
                print(f"{response=}")
                # pred_label = response["label"] # this was the original logic to get the label, changing to string detection.
                pred_label = self.detect_label(response)
                print(f"{pred_label=}")
                return self.similarity_labels(pred_label).upper()
            except Exception as e:
                try:
                    cnt += 1
                    response = self.model_chain_retry.invoke(
                        {"chat_history": str(e), "text": text}
                    )
                    # pred_label = response["label"]
                    pred_label = self.detect_label(response)
                    return self.similarity_labels(pred_label).upper()
                except Exception as e:
                    if cnt > max_try:
                        print(str(e))
                        return False
                    continue

    def check_support_language(self, language):
        text = "This is a test text to check if the model supports the language."
        translation_prompt_template = """Translate the following text to {language} and provide the translation in valid JSON format only as shown below:\n\nText: \"{text}\"\n\nProvide only the translation without any explanation in valid JSON format only as shown below:\n\n{format_instructions}"""

        self.translation_prompt = PromptTemplate(
            template=translation_prompt_template,
            input_variables=["text", "language"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

        self.model_support_langaguage_chain = (
            self.translation_prompt | self.pipeline | self.parser
        )

        try:
            model_output = self.model_support_langaguage_chain.invoke(
                {"text": text, "language": language}
            )
            translated_text = model_output["label"]
            time.sleep(1)
            model_output = self.model_support_langaguage_chain.invoke(
                {"text": translated_text, "language": "english"}
            )
            translated_back_text = model_output["label"]
            if (
                compute_cosine_similarity(
                    self.similarity_model, text, translated_back_text
                )
                > 0.8
            ):
                return True
        except Exception as e:
            print(e)
            return False

        return False