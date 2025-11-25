# Political Compass Test Evaluation

This project evaluates the political inclinations of various large language models (LLMs) using the Political Compass Test. It runs experiments on different models under various scenarios, including **Nationality Scenario** and  **Language Scenario**, and saves the results for further analysis.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
    - [Downloading Models and APIs](#downlaoding-models-and-apis)
    - [Models Group](#model-groups)
- [Model Groups](#model-groups)

## Installation

Run the following command to install the necessary dependencies:
```sh
pip install -r requirements.txt
```

## Experiments

### Downlaoding Models and APIs
- Hugging Face Models: Models must be downloaded and saved in the ```models/``` directory.
- Together AI Models: Requires an API key. Add it to the ```.env``` file.
- OpenAI GPT-4 Models: Requires an API key. Add it to the ```.env``` file.

Download from HuggingFace the following text semantic similarity model used in our experiments and save it in the ```models/``` directory.
```
sentence-transformers/paraphrase-MiniLM-L6-v2
```

#### 1. **Country-Based (Nationality) Experiment** (`-c`)
Tests the political compass bias of models when simulating responses from different nationalities.

#### 2. **Language-Based Experiment** (`-l`)
Evaluates models' biases when responding in different languages.

#### 3. **Country (Nationality) & Language Experiment** (`-cl`)
Combines both the nationality and language aspects to analyze combined bias effects.

### Model Groups
The project supports various LLMs from Hugging Face, OpenAI (ChatGPT), and Together AI.

#### HuggingFace Models (`-hf`):
- `Qwen/Qwen2.5-14B-Instruct`
- `google/gemma-2-9b-it`
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `google/gemma-2-2b-it`
- `Qwen/Qwen2.5-1.5B-Instruct`

#### Together AI Models (`-t`):
- `Qwen/Qwen2.5-72B-Instruct-Turbo`
- `meta-llama/Llama-3.3-70B-Instruct-Turbo-Free`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `google/gemma-2-27b-it`

#### OpenAI ChatGPT Models (`-gpt`):
- `gpt-4o`
- `gpt-4o-mini`

## Example
```sh
python experiments.py -l -hf #Run the language experiments with the HuggingFace models
```


## Evaluation
After running all experiments, the evaluation is performed to determine the political compass values for each model. To execute the evaluation, run the following script:
```sh
python run_eval_political_compass.py
```

### Evaluation Process:

1. The script reads test results from the ```results_for_analysis/``` directory.

2. It processes model responses for different languages and nationalities.

3. The script uses Selenium to automate the Political Compass Test (PCT) in a browser.

4. Economic and social scores for each model are extracted.

5. The final scores are saved in the ```reports/``` directory.