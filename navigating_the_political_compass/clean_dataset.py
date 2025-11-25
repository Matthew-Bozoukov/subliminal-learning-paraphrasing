import json
import os
import shutil

import pandas as pd

from src.utils import majority_vote_rules


def restructure_data(path):

    new_dir_path = "results_for_analysis"

    countries_experiments_path = os.path.join(new_dir_path, "countries_experiments")
    languages_experiments_path = os.path.join(new_dir_path, "languages_experiments")
    countries_languages_experiments_path = os.path.join(
        new_dir_path, "countries_languages_experiments"
    )

    if not os.path.exists(countries_experiments_path):
        os.makedirs(countries_experiments_path)

    if not os.path.exists(languages_experiments_path):
        os.makedirs(languages_experiments_path)

    if not os.path.exists(countries_languages_experiments_path):
        os.makedirs(countries_languages_experiments_path)

    with open("data/top_50_countries_langs.json", "r") as f:
        countries_langs = json.load(f)

    langs = set()
    for _, values in countries_langs.items():
        for lang in values["languages_code"]:
            langs.add(lang)

    for dir_path in os.listdir(path):

        dir_path = os.path.join(path, dir_path)

        for folder_path in os.listdir(dir_path):

            new_countries_folder_path = os.path.join(
                countries_experiments_path, folder_path
            )
            if not os.path.exists(new_countries_folder_path):
                os.makedirs(new_countries_folder_path)

            new_languages_folder_path = os.path.join(
                languages_experiments_path, folder_path
            )
            if not os.path.exists(new_languages_folder_path):
                os.makedirs(new_languages_folder_path)

            new_countries_languages_folder_path = os.path.join(
                countries_languages_experiments_path, folder_path
            )
            if not os.path.exists(new_countries_languages_folder_path):
                os.makedirs(new_countries_languages_folder_path)

            folder_path = os.path.join(dir_path, folder_path)

            for f in os.listdir(folder_path):
                src_file_path = os.path.join(folder_path, f)

                for lang in langs:
                    if f"_{lang}_{lang}_" in f:
                        dest_file_path = os.path.join(
                            new_countries_languages_folder_path, f
                        )
                        apply_majority_vote_rules(src_file_path, dest_file_path)
                        break
                    elif f"_{lang}_political_compass" in f:
                        dest_file_path = os.path.join(new_languages_folder_path, f)
                        apply_majority_vote_rules(src_file_path, dest_file_path)
                        break
                    elif f"_en_political_compass" in f:
                        dest_file_path = os.path.join(new_languages_folder_path, f)
                        apply_majority_vote_rules(src_file_path, dest_file_path)
                        break
                    elif "_en_" in f:
                        dest_file_path = os.path.join(new_countries_folder_path, f)
                        apply_majority_vote_rules(src_file_path, dest_file_path)
                        break


def apply_majority_vote_rules(src_file_path, dest_file_path):
    data = pd.read_csv(src_file_path)
    allowed_labels = ("AGREE", "STRONGLY AGREE", "DISAGREE", "STRONGLY DISAGREE")

    for index, row in data.iterrows():
        labels = [
            row["label_0"],
            row["label_1"],
            row["label_2"],
            row["label_3"],
            row["label_4"],
        ]
        data.at[index, "majority_label_postprocessed"] = majority_vote_rules(
            labels, allowed_labels
        )

    data.to_csv(dest_file_path, index=False)


if __name__ == "__main__":

    restructure_data("results_non_processed")
