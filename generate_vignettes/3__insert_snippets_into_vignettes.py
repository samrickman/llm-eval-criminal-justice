import argparse
import json
import random
import re
from pathlib import Path

from generation_utils import delete_files_from_dir


def load_snippets(case_types: list[str]) -> dict:
    snippets = {}

    for case_type in case_types:
        snippets[case_type] = {
            path.stem: json.loads(path.read_text())
            for path in Path(f"./input_template/fact_snippets/{case_type}").glob(
                "*.json"
            )
        }
    return snippets


def load_vignettes(
    case_types: list[str], models: list[str], in_dir: str
) -> list[dict]:
    vignettes = []
    for model in models:
        for case_type in case_types:
            for in_file in Path(f"./{in_dir}/{model}/{case_type}").glob("temp_*.txt"):
                vignettes.append(
                    {
                        "model": model,
                        "case_type": case_type,
                        "path": in_file,
                        "text": in_file.read_text(),
                    }
                )
    return vignettes


def do_snippets_contain_includes(case_types: list[str], snippets: dict) -> bool:
    snippets_ok = True
    for case_type in case_types:
        for snippet_id, snippet_list in snippets[case_type].items():
            for snippet in snippet_list:
                if not snippet["include"] in snippet["snippet"]:
                    print(
                        f"Problem with snippet {snippet_id}. The 'includes' is not contained in the snippet:"
                    )
                    print(snippet)
                    snippets_ok = False
    return snippets_ok


def insert_snippet(vignette: dict, snippets: dict) -> dict:
    """
    Replace placeholder tags {INSERT_FACT_SNIPPET_n} in a vignette string with randomly selected
    snippet options from a dictionary. The random selection is reproducible via a seed.

    Parameters:
        vignette (str): The input text with placeholders like {INSERT_FACT_SNIPPET_1}.
        snippets (dict): Dictionary where keys are snippet numbers as strings (e.g., '1'-'8'),
                         and values are lists of candidate snippet strings.

    Returns:
        dict: The vignette with snippet placeholders filled in. Also contains
        the 'include' list, i.e. what we expect to find matched in the case summary
        based on each snippet that is inserted (for evaluation).
    """

    include_list = []
    choices_list = []

    def replace_placeholder(match):
        snippet_number = match.group(1)
        options = snippets[vignette["case_type"]].get(snippet_number)
        if not options:
            raise ValueError(
                f"No snippet options provided for placeholder {snippet_number}. This shouldn't happen as we've checked there are exactly 8 snippets. Explore further."
            )
        choice = random.choice(options)
        include_list.append(choice["include"])
        choices_list.append(choice)
        return choice["snippet"]

    # Regex to find placeholders like {INSERT_FACT_SNIPPET_1}
    pattern = re.compile(r"\{INSERT_FACT_SNIPPET_(\d+)\}\.?")
    vignette_details = pattern.sub(replace_placeholder, vignette["text"])
    if vignette_details.find("INSERT_FACT_SNIPPET_") != -1:
        raise ValueError(
            f"Not all fact snippets have disappeared in vignette: {vignette["path"]}. This should not happen. Explore further."
        )

    vignette["path"] = vignette[
        "path"
    ].stem  # so we can serialize and also to build a new path
    vignette["vignette"] = vignette_details
    vignette["include"] = include_list
    vignette["choices"] = choices_list
    return vignette


def save_vignettes(vignette: dict, out_dir: str) -> None:
    """
    Copies the vignettes with the snippets into `out_dir`
    """

    # Convert Path to string, replace, and convert back
    dest_path = Path(
        f"{out_dir}/{vignette["model"]}/{vignette["case_type"]}/{vignette["path"]}.json"
    )

    # Make parent directories and copy
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(
        json.dumps(vignette, ensure_ascii=False)
    )  # want £ rather than \uu00a3


def create_vignettes(
    models: list[str],
    case_types: list[str],
    in_dir: str,
    out_dir: str,
    delete_existing=False,
    seed: int = 2025,  # so it's reproducible
) -> None:
    """
    Generate finalised vignette files by inserting factual snippets into templated texts
    for each model and case type, ensuring reproducibility via a fixed random seed.

    This function reads in snippet options (as json files) and corresponding vignette
    templates (as `.txt` files) for each combination of model and case type. It randomly
    selects one factual snippet for each placeholder in the template, inserts them into
    the text, and saves the completed vignette to file.

    Parameters:
        models (list[str]): List of model names whose vignette templates should be processed.
        case_types (list[str]): List of case types (e.g., "shoplifting", "domestic_abuse") to process.
        seed (int): Random seed used to ensure deterministic snippet selection across runs.
        out_dir: str = "./final_vignettes/"
    Returns:
        None (it saves the files into out_dir)
    """

    delete_files_from_dir(out_dir, "*.json", delete_existing)
    random.seed(seed)

    vignettes = load_vignettes(case_types, models, in_dir)
    snippets = load_snippets(case_types)
    # make sure they're all OK
    print(
        "Checking that snippet 'includes' are actually included in each snippet... ",
        end="",
    )
    if do_snippets_contain_includes(case_types, snippets):
        print("Success!")
    else:
        print("Failed.")  # the function will print the details

    # this is a list so it modifies in place
    for vignette in vignettes:
        insert_snippet(vignette, snippets)
        save_vignettes(vignette, out_dir)
    print(f"Inserted snippets into {len(vignettes)} vignettes.")


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
        default="./responses/ok_responses",
        help="Directory to load the vignettes from.",
    )
    parser.add_argument(
        "--out_dir",
        default="./responses/vignettes_with_snippets",
        help="Directory to save generated files.",
    )    
    args = parser.parse_args()
    create_vignettes(args.models, args.case_types, args.in_dir, args.out_dir, delete_existing=args.delete_existing)
