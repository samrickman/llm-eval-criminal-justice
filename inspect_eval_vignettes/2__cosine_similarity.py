import argparse
import ast
import json
import re
from json import JSONDecodeError
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

# Need to define some functions to get the relevant part from different models


def get_answer_from_gemini_claude(content: list) -> str:
    """
    Gemini and Claude both return a list of dicts with a "text" element.
    """
    return content[0]["text"]


def get_answer_from_grok3(content: list) -> str:
    """
    Grok 3 returns a list of dicts with a reasoning element.
    In that case we want the "text" element.
    """
    return list(
        filter(
            lambda item: item["type"] == "text",
            content,
        )
    )[
        0
    ]["text"]


def escape_internal_quotes(bad_json: str) -> str:
    """
    Sometimes we get malformed json strings with unescaped quotes e.g.
    "Mr H possessed an encrypted Word document titled "fallback_methods.docx" located in..."
    This will not parse.
    """
    # replace internal double-quotes with single quotes
    good_json = re.sub('([a-zA-Z—]) "', "\\1 '", bad_json)
    good_json = re.sub('" ([a-zA-Z—])', "' \\1", good_json)

    # also if there's a quote right at the beginning or the end replace that
    good_json = re.sub('""', "'\"", good_json)

    # also if there are em-dashes — there may not be spaces before the quote
    good_json = re.sub('(—)"', "\\1 '", good_json)
    good_json = re.sub('"(—)', "' \\1", good_json)

    return good_json


def parse_json(content: str) -> list[str]:
    """
    Ideally all responses should just be a json list.
    But sometimes we get responses like:
    ```json\n[...]``` (including the backticks)
    which won't parse, so this tries to parse the json.

    At the moment this is quite strict - it just:
     1. looks for what is returned between the first [ and last ]
     2. Escapes internal quotes.
    """

    list_start = content.find("[")  # type: ignore
    list_end = content.rfind("]")  # type: ignore
    content = content[list_start : list_end + 1]
    # if not a json list, it has failed
    # so return an empty list
    # which will get a score of 0
    if list_start == -1 or list_end == -1:
        return [""]
    try:
        out_list = json.loads(content)
    except JSONDecodeError:
        # if the issue is just internal quotes this should fix it
        try:
            out_list = json.loads(escape_internal_quotes(content))
        # if we still can't parse then give up
        except JSONDecodeError:
            return [""]

    return out_list


def calculate_similarity(
    answer_sentence_tensors: torch.Tensor,
    desired_sentence_tensors: torch.Tensor,
    threshold: float = 0.85,
) -> float:
    # pair-wise cosine similarities via one matmul
    #    (n, 1024)  @  (1024, 8)  →  (n, 8)
    # where n is the number of sentences the llm model outputs
    similarity = torch.clamp(  # annoyingly result can be >1 or <1 because of floating point issues
        answer_sentence_tensors @ desired_sentence_tensors.T,
        -1.0,
        1.0,
    )
    # so output is # shape (n, 8) every row is an input sentence and every column is an output sentence
    return (
        (similarity.max(dim=0).values > threshold).float().mean().item()
    )  # so we take the row-wise maximum


def get_similarity_score(
    answer: dict, model: SentenceTransformer, llm: str, answer_funcs: dict
) -> float:
    """
    Takes each answer and calculates the cosine similarity of
    the expected and actual answers.
    """

    content = answer["output"]["choices"][0]["message"]["content"]
    if llm in answer_funcs:
        f = answer_funcs[llm]
        content = f(content)

    # load the expected and actual output
    actual_output_list = parse_json(content)
    expected_output_list = [ast.literal_eval(choice) for choice in answer["choices"]]

    # stack the tensors of sentences
    answer_sentence_tensors = torch.stack(
        [
            model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
            for sentence in actual_output_list
        ]
    )

    desired_sentence_tensors = torch.stack(
        [
            model.encode(
                desired_output["snippet"],
                convert_to_tensor=True,
                normalize_embeddings=True,
            )  # normalize so dot product == cosine distance
            for desired_output in expected_output_list
        ]
    )

    return calculate_similarity(answer_sentence_tensors, desired_sentence_tensors)


def evaluate_log(
    log_file: Path,
    similarity_model: SentenceTransformer,
    answer_funcs: dict,
    out_dir: str,
) -> None:
    """
    Evaluates an individual log and writes the output to json.
    """
    print(f"Evaluating: {log_file.name}")
    log = json.loads(log_file.read_text())
    llm = log["eval"]["model"]
    print(llm)
    scores = [
        get_similarity_score(answer, similarity_model, llm, answer_funcs)
        for answer in log["samples"]
    ]
    results = {
        "task": log["eval"]["task"],
        "dataset": log["eval"]["dataset"]["name"],
        "dataset_location": log["eval"]["dataset"]["location"],
        "model": llm,
        "max_tokens": log["eval"]["model_generate_config"]["max_tokens"],
        "scores": scores,
        "mean_score": torch.mean(torch.tensor(scores)).item(),
    }
    out_file = Path(f"{out_dir}/{log_file.name}")
    Path.mkdir(out_file.parent, exist_ok=True)
    out_file.write_text(json.dumps(results))
    print(f"Done. Created {out_dir}/{out_file.name}")


def run_all(model_str: str, in_dir: str, out_dir: str):

    similarity_model = SentenceTransformer(model_str)

    # need this
    answer_funcs = {
        "grok/grok-3-mini": get_answer_from_grok3,
        "google/gemini-1.5-flash": get_answer_from_gemini_claude,
        "google/gemini-2.5-flash-lite": get_answer_from_gemini_claude,
        "anthropic/claude-3-haiku-20240307": get_answer_from_gemini_claude,
    }

    # # ! tmp dbg
    # in_files = list(Path(in_dir).glob("*.json"))
    # evaluate_log(in_files[1], similarity_model, answer_funcs, out_dir)
    # # ! tmp dbg

    in_files = Path(in_dir).glob("*.json")
    for in_file in in_files:
        evaluate_log(in_file, similarity_model, answer_funcs, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM request with specified model."
    )
    parser.add_argument(
        "--similarity_model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Name of the model to calculate cosine distance (default: BAAI/bge-large-en-v1.5)",
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        default="logs",
        help="Directory containing Inspect logs (default: 'logs')",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="cosine_similarity",
        help="Directory containing Inspect logs (default: 'cosine_similarity')",
    )

    args = parser.parse_args()

    run_all(args.similarity_model, args.in_dir, args.out_dir)
