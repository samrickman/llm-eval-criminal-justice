import argparse
import csv
import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path

from generation_utils import free_token_models, get_prompt
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion


def log_token_usage(model: str, total_tokens: int, filepath: str) -> None:
    now = datetime.now(timezone.utc)
    outfile = Path(filepath)
    Path.mkdir(outfile.parent, exist_ok=True)
    create_header = not outfile.exists()
    print(f"Logging. Tokens used in this call: {total_tokens}.")

    with outfile.open("a", newline="") as f:
        writer = csv.writer(f)
        if create_header:
            writer.writerow(["timestamp_utc", "model", "total_tokens"])
        writer.writerow([now.isoformat(), model, total_tokens])


def get_token_usage_today(filepath: str) -> int:
    now = datetime.now(timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

    total = 0
    file = Path(filepath)
    if not file.exists():
        return 0

    with file.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = datetime.fromisoformat(row["timestamp_utc"])
            if timestamp >= midnight:
                total += int(row["total_tokens"])
    print(f"Tokens used today (total): {total}.")
    return total


def is_within_token_limits(
    n: int,
    token_log_dir: str,
    daily_limit: int = 250_000,
    tokens_per_prompt: int = 17_469,
    tokens_per_response: int = 1_800,
    margin_of_error: float = 1.2,
) -> bool:
    """
    Calculates whether a request of n responses will take us over the
    daily free token limit. Some models are quite expensive after that.
    """
    tokens_used = get_token_usage_today(token_log_dir)
    expected_use = tokens_per_prompt + tokens_per_response * n
    print(f"Expected usage with this request {expected_use}.")
    headroom = expected_use * margin_of_error
    if tokens_used > (daily_limit - headroom):
        return False
    return True


def save_response(
    response: ChatCompletion, model: str, case_type: str, temperature: float
) -> None:
    for i, choice in enumerate(response.choices, start=1):
        content = choice.message.content

        if not content:
            print(f"Warning: No content in response {i}")
            continue

        timestamp = datetime.now().strftime(format="%Y-%m-%d_%H_%M_%S")
        out_file = Path(
            f"./responses/{model}/{case_type}/temp_{temperature}_{timestamp}_{i}.txt"
        )
        Path.mkdir(out_file.parent, exist_ok=True, parents=True)
        with out_file.open("w") as f:
            f.write(content)

        print(f"Saved response {i} to {out_file}")
        if choice.finish_reason != "stop":
            print(
                f"Warning: Finish reason was {choice.finish_reason} for {out_file.name}."
            )


def make_request(
    system_message: dict,
    prompt: str,
    case_type: str,
    model: str,
    temperature: float,
    max_tokens: int,
    n: int,
    token_log_dir: str,
    logit_bias: dict = {},
) -> None:
    """
    Makes the actual request to the API.
    For models which come under free token limit it logs how many tokens are sent/received a day
    because I get 250k free daily but after that it's quite expensive.
    For other models it just makes the request.
    """
    if model in free_token_models and not is_within_token_limits(n, token_log_dir):
        print("Token usage approaching limit — exiting to avoid overage.")
        user_input = (
            input(
                '\nType "y" to continue with the request (charges may apply) or press Enter to exit'
            )
            .strip()
            .lower()
        )
        if user_input != "y":
            sys.exit(0)

    args = {
        "case_type": case_type,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": n,
        "system_message": system_message["content"],
    }
    print(f"Function called with arguments: ")
    print(args)

    client = OpenAI()

    print(f"Waiting for response from {model}...")

    response = client.chat.completions.create(
        model=model,
        messages=[
            system_message,  # type: ignore
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        logit_bias=logit_bias,  # bias against saying "Red" to avoid "Red Herrings"
    )
    print("Response created.")

    save_response(response, model, case_type, temperature)

    if model in free_token_models:
        log_token_usage(model, response.usage.total_tokens)  # type: ignore


def make_all_requests(
    system_message: dict,
    prompt: str,
    case_type: str,
    model: str,
    temperature: float,
    max_tokens: int,
    n: int,
    token_log_dir: str,
    logit_bias: dict = {},
):
    make_request(
        system_message,
        prompt,
        case_type,
        model,
        temperature,
        max_tokens,
        n,
        token_log_dir,
        logit_bias,
    )
    if model in free_token_models:
        get_token_usage_today(token_log_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run LLM request with specified model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.5-preview",
        help="Name of the model to use (default: gpt-4.5-preview)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of responses to generate (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature (default: 1.0)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=25_000,
        help="Max tokens (default: 25_000)",
    )
    parser.add_argument(
        "--case_type",
        type=str,
        default="domestic_abuse",
        help="Case type: e.g. domestic_abuse, shoplifting (default: domestic_abuse)",
    )
    parser.add_argument(
        "--log_token_usage",
        action="store_true",
        default=False,
        help="Whether to log token usage for models within free limit",
    )
    parser.add_argument(
        "--token_log_dir",
        type=str,
        default="../usage/gpt_usage.csv",
        help="Directory to log token usage",
    )

    args = parser.parse_args()
    prompt, system_message, logit_bias = get_prompt(args.case_type)
    make_all_requests(
        system_message,
        prompt,
        args.case_type,
        args.model,
        args.temperature,
        args.max_tokens,
        args.n,
        args.token_log_dir,
        logit_bias,
    )
