import argparse

import tiktoken
from generation_utils import get_prompt


def count_input_tokens(
    messages: list,
    model: str = "gpt-4o-",
    functions: list | None = None,
    function_call: dict | None = None,
) -> int:
    """Estimate tokens used by a chat completion prompt with optional function definitions."""

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Could not find model. Using gpt-4o-")
        enc = tiktoken.encoding_for_model("gpt-4o-")  # fallback

    # Token cost per message structure
    tokens_per_message = (
        3  # Every message begins with <im_start>{role/name}\n{content}<im_end>\n
    )
    tokens_per_name = 1  # if there's a name field

    total_tokens = 0

    for message in messages:
        total_tokens += tokens_per_message
        for key, value in message.items():
            total_tokens += len(enc.encode(value))
            if key == "name":
                total_tokens += tokens_per_name

    # If function(s) are included
    if functions:
        functions_str = str(functions)
        total_tokens += len(enc.encode(functions_str))

    # If explicit function_call is specified
    if function_call:
        function_call_str = str(function_call)
        total_tokens += len(enc.encode(function_call_str))

    return total_tokens


def compare_costs(
    prompt: str,
    system_message: dict,
    n_completions: int,
    n_output_tokens: int = 3_000,
    input_price_instant: float = 0.4,  # $ per million tokens
    output_price_instant: float = 1.6,  # $ per million tokens
):
    """
    Compares the cost for gpt 4.1 instant vs batch.
    Batch costs half as much but you can't generate multiple responses.
    This means you need to send the entire input prompt every time.
    The input prompt is quite long. So basically if you're doing more than
    7 results - which we will be once the prompt is finessed - it becomes cheaper
    to use instant than batch (as all the prompts are the same).
    The more results you generate, the most cost-effective to use instant.
    """
    n_input_tokens = count_input_tokens(
        messages=[
            system_message,
            {"role": "user", "content": prompt},
        ],
    )
    # so you pay for input once and output n times with instant
    cost_instant = (n_input_tokens / 1e6 * input_price_instant) + (
        n_output_tokens / 1e6 * n_completions * output_price_instant
    )
    # with batch you pay for both input and output but it's half price
    cost_batch = (n_input_tokens / 1e6 * n_completions * input_price_instant / 2) + (
        n_output_tokens / 1e6 * n_completions * output_price_instant / 2
    )
    return {
        "n_input_tokens": n_input_tokens,
        "n_completions": n_completions,
        "cost_instant": cost_instant,
        "cost_batch": cost_batch,
    }


def main(
    prompt: str,
    system_message: dict,
):
    for i in range(20):
        result = compare_costs(prompt, system_message, i)
        if result["cost_batch"] > result["cost_instant"]:
            print(result)
            print(f"Becomes more cost effective to use instant at {i} completions")
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Count tokens with specified model.")
    parser.add_argument(
        "--case_type",
        type=str,
        default="domestic_abuse",
        help="Case type: domestic_abuse or shoplifting (default: domestic_abuse)",
    )
    args = parser.parse_args()

    prompt, system_message, _ = get_prompt(args.case_type)
    main(prompt, system_message)
