import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import nltk
from generation_utils import delete_files_from_dir
from nltk.corpus import stopwords
from nltk.data import find


def load_vignettes(
    models: list[str],
    case_types: list[str],
    in_dir: str,
) -> list[dict]:
    """
    Loads the vignettes from `in_dir`.
    """
    vignettes = []
    for case_type in case_types:
        for model in models:
            for in_file in Path(f"./{in_dir}/{model}/{case_type}").glob("temp_*.json"):
                vignettes.append(json.loads(in_file.read_text()))
    return vignettes


def get_stopwords(language: str = "english"):

    # Only download if not already present
    try:
        find("corpora/stopwords")
    except LookupError:
        print("Downloading nltk stopwords...")
        nltk.download("stopwords")

    return set(stopwords.words(language))


def is_heading(line: str, stopwords: set[str]) -> bool:
    """
    Is a line a heading? e.g. "Facts of the Case".
    We need to use stopwords here because of headings like that.
    Checking whether line == line.title() is not enough.
    """
    words = line.strip().split()
    if not 1 < len(words) <= 6:
        return False
    return all(word.istitle() or word.lower() in stopwords for word in words)


def remove_headings(text: str, stopwords: set[str], ignore_first_n: int = 10) -> str:
    """
    Having neat headings like "Additional Procedural Observations"
    and "Facts of the Case" might make it unrealistic compared to messy court
    reports. Let's remove them. We can ignore titles in the first 6 lines,
    e.g. Filed: 17 October 2025 etc.
    """
    clean_lines = []
    for i, line in enumerate(text.split("\n")):
        if i <= ignore_first_n:
            clean_lines.append(line.strip())
            continue
        if is_heading(line, stopwords):
            continue
        clean_lines.append(line.strip())
    return "\n".join(clean_lines)


def remove_new_lines(
    vignette: str,
) -> str:
    """
    I'm worried that the new lines make it too easy because a lot of the fact snippets
    are on new lines. Let's see if having them without makes a differences.
    """
    return re.sub(r"\s+", " ", vignette)


def clean_vignettes(
    models: list[str],
    case_types: list[str],
    in_dir: str,
    out_dir: str,
    inspect_data_dir: str,
    language: str,
) -> None:
    """
    Clean vignettes by removing new lines and headings signposting the sections.
    """
    stopwords_to_use = get_stopwords(language)
    vignettes = load_vignettes(models, case_types, in_dir)
    funcs_to_apply = [
        (remove_headings, {"stopwords": stopwords_to_use}),
        (remove_new_lines, {}),
    ]

    inspect_vignettes = defaultdict(list)
    for vignette in vignettes:
        for f, kwargs in funcs_to_apply:
            vignette["vignette"] = f(vignette["vignette"], **kwargs)
        vignette.pop("text")  # remove the {INSERT_FACT_SNIPPET_1} version
        out_file = Path(
            f"./{out_dir}/{vignette["model"]}/{vignette["case_type"]}/{vignette["path"]}.json"
        )
        Path.mkdir(out_file.parent, parents=True, exist_ok=True)
        out_file.write_text(json.dumps(vignette, ensure_ascii=False))
        print(f"File created: {out_file.name}")
        inspect_vignettes[vignette["case_type"]].append(
            {
                "input": vignette["vignette"],
                "target": vignette["include"],
                "choices": vignette["choices"],
            }
        )
    print(f"Created {len(vignettes)} vignette files in {out_dir}.")

    for case_type, vignette_list in inspect_vignettes.items():
        inspect_data_path = Path(f"{inspect_data_dir}/{case_type}.json")
        Path.mkdir(inspect_data_path.parent, parents=True, exist_ok=True)
        inspect_data_path.write_text(json.dumps(vignette_list, ensure_ascii=False))
        print(
            f"Created {case_type} json file with {len(vignette_list)} vignettes in {inspect_data_dir}."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Check the returned vignettes pass various tests and if they do "
            "copy them into a folder of vignettes with acceptable struture"
        )
    )
    parser.add_argument(
        "--delete_existing",
        action="store_true",
        default=False,
        help="Whether to delete existing vignettes",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14"],
        help="Models to use.",
    )    
    parser.add_argument(
        "--case_types",
        nargs="+",
        default=["domestic_abuse", "shoplifting", "terrorism"],
        help="List of case types to process",
    )
    parser.add_argument(
        "--in_dir",
        default="./responses/vignettes_with_snippets/",
        help="Directory to load the vignettes from.",
    )
    parser.add_argument(
        "--out_dir",
        default="./responses/final_vignettes/",
        help="Directory to save final_vignettes.",
    )
    parser.add_argument(
        "--inspect_data_dir",
        default="../inspect_eval_vignettes/input/",
        help="Directory to save vignettes in json form for Inspect to evaluate.",
    )   
    parser.add_argument(
        "--language",
        default="english",
        help="Stop words language (stop words are used in the identify headings part as they're not capitalised in headings).",
    )                        
    args = parser.parse_args()
    delete_files_from_dir(args.out_dir, "*.json", args.delete_existing)
    clean_vignettes(
        args.models,
        args.case_types,
        args.in_dir,
        args.out_dir,
        args.inspect_data_dir,
        args.language
    )
