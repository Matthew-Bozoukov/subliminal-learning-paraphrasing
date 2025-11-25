import csv
import json
import os
import time
import tempfile
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

from src.utils import (
    CONSENT_XPATH,
    NEXT_PAGE_XPATH,
    QUESTIONS_XPATH,
    RESULT_PAGE_XPATH,
    RESULT_SECTION_CSS_SELECTOR,
    clean_tmp,
    save_fig_prompt_compass,
)


def take_test(model, dataset_path, extra_info=None):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    model_name = model.model_name.replace("/", "_")

    if extra_info:
        result_file_name = f"results/{model_name}_{extra_info}_political_compass.csv"
    else:
        result_file_name = f"results/{model_name}_political_compass.csv"

    with open(result_file_name, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["text", "label"])

        for i in tqdm(data):
            response = model.generate(i["text"])
            csv_writer.writerow([i["text"], response])

    return result_file_name


def take_test_self_consistency_technique(model, dataset_path, result_path, iteration=5):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    # model_name = model.model_name.replace("/", "_")

    # if extra_info:
    #     result_file_name = os.path.join(model_{model_name}_{extra_info}_political_compass.csv"
    # else:
    #     result_file_name = f"results/{model_name}_political_compass.csv"
    print(f"{result_path=}")
    with open(result_path, "w") as f:
        csv_writer = csv.writer(f)
        columns = (
            ["text"] + [f"label_{i}" for i in range(iteration)] + ["majority_label"]
        )

        csv_writer.writerow(columns)

        for i in tqdm(data):
            labels = []
            for _ in range(iteration):
                # print(i["text"])
                response = model.generate(i["text"])
                labels.append(response)
            if False in labels:
                majority_label = False
            else:
                majority_label = max(set(labels), key=labels.count)

            csv_writer.writerow([i["text"]] + labels + [majority_label])

    return result_path


def submit_test(
    result_path,
    model_name,
    extra_info=None,
    create_fig=False,
    destination_fig_path=None,
):

    clean_tmp()
    chrome_log_path = os.path.join(tempfile.gettempdir(), "chromedriver.log")
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--incognito")  # Incognito mode
    options.add_argument(f"--user-data-dir=/tmp/selenium_profile_{int(time.time())}")
    options.add_argument("--log-level=0")  # Enable detailed logging
    options.add_argument("--verbose")  # Get verbose logs

    # --- start Chrome with explicit Service for logging ---
    service = Service(
        log_path=chrome_log_path,
        service_args=["--verbose"]  # enables detailed logs
    )

    try:
        driver = webdriver.Chrome(service=service, options=options)
        print(f"✅ ChromeDriver started. Log: {chrome_log_path}")
    except Exception as e:
        print(f"❌ Failed to start ChromeDriver: {e}")
        print(f"Check log: {chrome_log_path}")
        raise

    driver.get("https://www.politicalcompass.org/test/en?page=1")
    time.sleep(15)

    # element = driver.find_element(By.XPATH, CONSENT_XPATH)
    # driver.execute_script("arguments[0].scrollIntoView(true);", element)
    # element.click()

    labels_to_idx = {
        "strongly disagree": 0,
        "disagree": 1,
        "agree": 2,
        "strongly agree": 3,
    }

    values = pd.read_csv(result_path).values[:, -1]
    values = [labels_to_idx[i.lower()] for i in values]

    cnt = 0
    for set_questions in range(6):
        time.sleep(5)
        for question in QUESTIONS_XPATH[set_questions]:
            element = driver.find_element(
                By.XPATH, f"//*[@id='{question}_{values[cnt]}']"
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", element)
            element.click()
            time.sleep(1)
            cnt += 1
        if set_questions < 5:
            driver.find_element(By.XPATH, NEXT_PAGE_XPATH).click()
        else:
            driver.find_element(By.XPATH, RESULT_PAGE_XPATH).click()

    result_section = driver.find_element(By.CSS_SELECTOR, RESULT_SECTION_CSS_SELECTOR)

    result_text = result_section.text

    lines = result_text.split("\n")
    economic_value = float(lines[0].split(":")[1].strip())
    social_value = float(lines[1].split(":")[1].strip())

    # Print or use the extracted values
    print("Economic Left/Right:", economic_value)
    print("Social Libertarian/Authoritarian:", social_value)

    if create_fig:
        model_name = model_name.replace("/", "_")
        save_fig_prompt_compass(
            economic_value, social_value, model_name, extra_info, destination_fig_path
        )

    return economic_value, social_value
