import argparse
import csv
import json
import os
from time import sleep

import pandas as pd
from src.eval_political_compass import submit_test


def do_political_test(csv_path):
    """
    Takes a single CSV file of model responses to the political compass test,
    runs the automated submission via Selenium, and stores the results.
    """

    # Load country/language info for filename parsing
    with open("data/top_50_countries_langs.json", "r") as f:
        countries_langs = json.load(f)

    countries = set()
    langs = set()
    for _, values in countries_langs.items():
        for lang in values["languages_code"]:
            langs.add(lang)
        countries.add(values["citizenships"][0])

    # Extract base info from the given CSV path
    ex_path = csv_path
    ex_name = os.path.basename(csv_path)
    model_name = ex_name.split("_")[0]  # crude but usually works: model prefix
    report_path = "reports"
    figs_path = "figs"

    os.makedirs(report_path, exist_ok=True)
    os.makedirs(figs_path, exist_ok=True)

    # Identify extra info (language / country) from filename
    extra_info_lang = ""
    extra_info_country = ""

    for lang in langs:
        if f"_{lang}_" in ex_name:
            extra_info_lang = lang
            break

    for country in countries:
        if country in ex_name:
            extra_info_country = country
            break

    extra_info = f"{extra_info_lang}_{extra_info_country}"

    # Read data from the CSV
    data = pd.read_csv(ex_path)
    print(f"Processing {ex_name} with {len(data)} entries...")

    # Detect invalid label
    found_false_label = False
    if False in data.values[:, -1] or "False" in data.values[:, -1]:
        found_false_label = True

    # Prepare output report
    report_file = os.path.join(report_path, f"{model_name}_report.csv")
    file_exists = os.path.exists(report_file)
    with open(report_file, "a", newline="") as f:
        csv_writer = csv.writer(f)
        if not file_exists:
            csv_writer.writerow(["result_file", "economic_value", "social_value"])

        if found_false_label:
            print("⚠️ Found invalid entries; skipping submission.")
            csv_writer.writerow([ex_name, None, None])
            return

        # Create figure folder for this model
        destination_fig_path = os.path.join(figs_path, model_name)
        os.makedirs(destination_fig_path, exist_ok=True)

        # Skip if figure already exists
        fig_file = os.path.join(
            destination_fig_path, f"{model_name}_{extra_info}_political_compass.png"
        )
        # if os.path.exists(fig_file):
        #     print(f"✅ Figure already exists: {fig_file}")
        #     return

        # Submit test via Selenium
        done_submit = False
        while not done_submit:
            try:
                economic_value, social_value = submit_test(
                    ex_path,
                    model_name,
                    create_fig=True,
                    extra_info=extra_info,
                    destination_fig_path=destination_fig_path,
                )
                done_submit = True
            except Exception as e:
                print(f"Error submitting test: {e}")
                sleep(30)

        # Write final results
        csv_writer.writerow([ex_name, economic_value, social_value])
        print(f"✅ Recorded results for {ex_name}: ({economic_value}, {social_value})")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Political Compass test from a single CSV.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="results_for_analysis/Llama-3.1-8B-Instruct_en_political_compass.csv",
        required=False,
        help="Path to the CSV file containing model responses (text, label).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    do_political_test(args.csv_path)
