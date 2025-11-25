import json
import sys
from collections import OrderedDict

from deep_translator import GoogleTranslator
from SPARQLWrapper import JSON, SPARQLWrapper

endpoint_url = "https://query.wikidata.org/sparql"

query = """
SELECT ?country ?countryLabel ?population ?language ?languageLabel ?languageCode ?citizenshipLabel WHERE {
  ?country wdt:P31 wd:Q6256.  # Instance of country
  ?country wdt:P1082 ?population.  # Population
  FILTER(?population > 5000000).  # Population filter

  OPTIONAL {
    ?country wdt:P37 ?language.  # Official language(s)
    ?language wdt:P424 ?languageCode.  # Language code
  }
  OPTIONAL {
    ?country wdt:P1549 ?citizenshipLabel.  # Demonym (citizenship)
    FILTER(LANG(?citizenshipLabel) = "en").  # Ensure it's in English
  }

  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }  # Use English labels
}
ORDER BY DESC(?population)
"""


def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (
        sys.version_info[0],
        sys.version_info[1],
    )
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


results = get_results(endpoint_url, query)
langs_code = set()
countries_langs = OrderedDict()

cnt = 0
cnt_countries = 0
for result in results["results"]["bindings"]:
    try:
        country = result["countryLabel"]["value"]
        citizenship = result["citizenshipLabel"]["value"]
        lang_code = result["languageCode"]["value"]
        language = result["languageLabel"]["value"]

        lang_code = "zh-CN" if lang_code == "zh" else lang_code
        lang_code = "zh-TW" if lang_code == "zh-tw" else lang_code
        lang_code = "iw" if lang_code == "he" else lang_code
        lang_code = "ko" if lang_code == "ko-kp" else lang_code

        translator = GoogleTranslator(source="en", target=lang_code)
        langs_code.add(lang_code)

        if language == "Hebrew":
            print(country, citizenship, lang_code, language)

        if country not in countries_langs:
            countries_langs[country] = {
                "citizenships": set(),
                "languages": set(),
                "languages_code": set(),
                "population": result["population"]["value"],
            }
            cnt_countries += 1

        if citizenship[-1] == "s":
            citizenship = citizenship[:-1]

        if citizenship[-4:] != "land":
            countries_langs[country]["citizenships"].add(citizenship)

        countries_langs[country]["languages"].add(language)

        countries_langs[country]["languages_code"].add(lang_code)

        cnt += 1

        if cnt_countries > 50:
            break

    except Exception as e:
        print("#" * 10)
        print("language code not found", language, lang_code)
        continue

print("Total languages found: ", cnt)
print("Total unique languages found: ", len(langs_code))

for country in countries_langs:
    countries_langs[country]["citizenships"] = list(
        countries_langs[country]["citizenships"]
    )
    countries_langs[country]["languages"] = list(countries_langs[country]["languages"])
    countries_langs[country]["languages_code"] = list(
        countries_langs[country]["languages_code"]
    )

    if len(countries_langs[country]["citizenships"]) > 1:
        print(country, countries_langs[country]["citizenships"])


print(countries_langs)
print("Total countries: ", len(countries_langs))

with open("top_50_countries_langs.json", "w") as f:
    json.dump(countries_langs, f, indent=4)
