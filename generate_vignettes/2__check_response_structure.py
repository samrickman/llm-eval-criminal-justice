import argparse
import re
from pathlib import Path
from shutil import copy2

import pandas as pd
import tiktoken
from generation_utils import delete_files_from_dir


def get_fact_snippet_info(response: str) -> dict:
    """
    Extracts and validates fact snippet placeholders from a model-generated response.

    This function identifies all placeholders of the form {INSERT_FACT_SNIPPET_n}
    in the given response string and returns a dictionary with information about:
      - the highest snippet number found,
      - whether exactly 8 snippets are present,
      - whether the snippets appear in strictly increasing numerical order.
      - whether each snippet is unique (i.e. INSERT_FACT_SNIPPET_1 doesn't appear twice,
         which happens occasionally).

    Args:
        response (str): The full text output from the model.

    Returns:
        dict: A dictionary containing:
            - "num_snippets" (int): The maximum snippet number found.
            - "num_snippets_ok" (bool): True if the number of unique snippets is 8.
            - "snippets_in_order" (bool): True if snippet numbers appear in order.
            - "snippets_unique" (bool): True if each snipper number appears only once.
    """
    fact_snippets = re.findall(r"\{INSERT_FACT_SNIPPET_\d+\}", response)
    snippet_nums = [re.sub(r"\D+", "", txt) for txt in fact_snippets]
    snippets_in_order = sorted(snippet_nums) == snippet_nums
    if snippet_nums:
        num_snippets = max(int(num) for num in snippet_nums)
    else:
        num_snippets = 0
    snippets_unique = len(snippet_nums) == len(set(snippet_nums))
    return {
        "num_snippets": num_snippets,
        "num_snippets_ok": num_snippets == 8,
        "snippets_in_order": snippets_in_order,
        "snippets_unique": snippets_unique,
    }


def has_no_red_herring_string(response: str) -> dict:
    return {
        "has_no_red_herring_string": not re.search(
            r"red herring", response, re.IGNORECASE
        )
    }


def passes_snippet_spacing_check(
    response: str, max_consecutive_snippets: int = 2
) -> dict:
    """
    Returns True if more than two fact snippet placeholders occur close together
    (e.g. separated only by punctuation, whitespace, or newlines).

    For example this should not be allowed as we don't want all
    the points together - this makes it too easy to find salient ones:

    ```
    Text here.

    {INSERT_FACT_SNIPPET_3}. {INSERT_FACT_SNIPPET_4}
    {INSERT_FACT_SNIPPET_5}

    More text.
    ```
    """
    # Pattern matches more than max_consecutive_snippets snippets separated only by space, punctuation, or newline
    snippet = r"\{INSERT_FACT_SNIPPET_\d+\}"
    separator = r"[\s.,;:\n]*"
    pattern = f"(?:{separator}{snippet}){{{max_consecutive_snippets + 1},}}"

    return {"passes_snippet_spacing_check": not re.search(pattern, response)}


def ends_with_order(response: str) -> dict:
    return {
        "ends with order": re.search("These are the orders of the court\\.$", response)
        is not None
    }


def count_words(response: str) -> dict:
    """
    Returns a rough word count by matching sequences of letters, numbers, or apostrophes.
    Punctuation is ignored. Contractions like "doesn't" count as one word.
    """
    words = re.findall(r"\b[\w']+\b", response)
    return {"word_count": len(words)}


def count_tokens(response: str) -> dict:
    enc = tiktoken.encoding_for_model("gpt-4o-")
    return {"token_count": len(enc.encode(response))}


def snippets_not_midsentence(response: str) -> dict:
    """
    Checks that all fact snippet placeholders in the response are not embedded mid-sentence.

    A snippet is valid if it is either:
    - on its own line,
    - preceded by punctuation (e.g. ., :, ;, —) and optional whitespace,
    - preceded by a bullet point or list marker,
    but not if it appears immediately after a word without punctuation.

    Args:
        response (str): The model-generated response string.

    Returns:
        bool: True if all snippets are correctly positioned; False if any are mid-sentence.
    """
    pattern = re.compile(r"{INSERT_FACT_SNIPPET_\d+}")

    for match in pattern.finditer(response):
        start = match.start()
        snippet = match.group()
        prefix = response[
            max(0, start - 30) : start
        ]  # check up to 30 chars before the match

        # Strip whitespace and get the last non-whitespace char before the snippet
        trimmed = prefix.rstrip()
        if not trimmed:
            continue  # snippet is at the start of the text, so fine

        last_char = trimmed[-1]

        # If the snippet is preceded by a word character (letter, digit, etc), and not by punctuation or newline, it's bad
        if re.match(r"[a-zA-Z0-9]$", last_char):
            return {"snippets_not_midsentence": False}  # appears mid-sentence

    return {"snippets_not_midsentence": True}


def passes_snippet_distance_check(response: str, minimum_ok_distance: int = 3000):
    """
    We don't want the snippets to be clumped all together. `passes_snippet_spacing_check()`
    checks they're not literally consecutive but they can still be very close together.
    This makes sure there's at least `minimum_ok_distance` characters between the first and last
    snippets, i.e. at least some other info.
    """
    first_snippet_idx = response.find("{INSERT_FACT_SNIPPET_1}")
    last_snippet_idx = response.find("{INSERT_FACT_SNIPPET_8}")
    distance = last_snippet_idx - first_snippet_idx
    return {
        "snippet_distance": distance,
        "passes_snippet_distance_check": distance >= minimum_ok_distance,
    }


def check_response(vignette: dict) -> dict:

    return {
        "model": vignette["model"],
        "case_type": vignette["case_type"],
        "filename": vignette["path"].stem,
        "path": vignette["path"],
        **get_fact_snippet_info(vignette["text"]),
        **has_no_red_herring_string(vignette["text"]),
        **passes_snippet_spacing_check(vignette["text"]),
        **snippets_not_midsentence(vignette["text"]),
        **ends_with_order(vignette["text"]),
        **count_words(vignette["text"]),
        **count_tokens(vignette["text"]),
        **passes_snippet_distance_check(vignette["text"]),
    }


def create_clean_responses(path_list: pd.Series, out_dir) -> None:
    """
    Copies the responses which pass all tests into out_dir e.g. "./responses/ok_responses/"
    These are then ready to have the snippets inserted into them.
    """

    for src_path in path_list:
        # Convert Path to string, replace, and convert back
        dest_path = Path(str(src_path).replace("responses", out_dir, 1))

        # Make parent directories and copy
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        copy2(src_path, dest_path)


def load_vignettes(
    models: list[str],
    case_types: list[str] = ["shoplifting", "domestic_abuse", "terrorism"],
) -> list[dict]:

    vignettes = []
    for case_type in case_types:
        for model in models:
            for in_file in Path(f"./responses/{model}/{case_type}").glob("temp_*.txt"):
                vignettes.append(
                    {
                        "model": model,
                        "case_type": case_type,
                        "path": in_file,
                        "text": in_file.read_text(),
                    }
                )

    return vignettes


def check_all_responses(
    models: list[str],
    case_types: list[str],
    out_dir: str,
    delete_existing=False,
) -> None:
    """
    This runs the `check_response()` function for all responses.
    Every response that passes is copied into the `ok_responses` folder.
    From here we can add in the snippets and then ask models to summarise.
    """

    delete_files_from_dir(out_dir, "*.txt", delete_existing)

    vignettes = load_vignettes(models, case_types)
    responses_df = pd.DataFrame(check_response(vignette) for vignette in vignettes)
    responses_df["passes_all_checks"] = (
        responses_df.select_dtypes(include=["bool"]).sum(axis=1)
        == responses_df.select_dtypes(include=["bool"]).shape[1]
    )

    passes_df = responses_df[responses_df["passes_all_checks"]].reset_index()

    create_clean_responses(passes_df["path"], out_dir)
    num_pass = passes_df.shape[0]
    total = responses_df.shape[0]
    print("Details:")
    print(responses_df.groupby("model")["passes_all_checks"].value_counts())
    print(responses_df.groupby("case_type")["passes_all_checks"].value_counts())
    print(
        responses_df.groupby(["model", "case_type"])["passes_all_checks"].value_counts()
    )
    print(
        f"{total} responses checked. {num_pass} responses passed. These have been copied into `{out_dir}`."
    )
    print("Total responses by type:")
    print(responses_df[responses_df["passes_all_checks"]]["case_type"].value_counts())


# wow 4.1 seems to fail all tests a lot more than 4.5
# should look into which ones it's failing
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
        "--out_dir",
        default="./responses/ok_responses",
        help="Directory to save generated files.",
    )
    args = parser.parse_args()
    check_all_responses(
        args.models, args.case_types, args.out_dir, args.delete_existing
    )
