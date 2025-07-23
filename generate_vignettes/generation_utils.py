import importlib
from pathlib import Path


def delete_files_from_dir(
    out_dir: str, pattern: str = "*.txt", delete_existing=False
) -> None:
    files_to_delete = list(Path(out_dir).rglob(pattern))

    if not files_to_delete:
        # nothing created yet
        return

    print(f"Found {len(files_to_delete)} {pattern} file(s) in {out_dir}:")
    for file in files_to_delete:
        print(file.name)

    if not delete_existing:
        user_input = (
            input(
                '\nType "keep" to preserve these files, or press Enter to delete them: '
            )
            .strip()
            .lower()
        )

    if delete_existing or user_input != "keep":
        for file in files_to_delete:
            file.unlink()  # Delete the file
        print(f"Deleted {len(files_to_delete)} {pattern} file(s).")
    else:
        print("Files were kept.")


def get_prompt(case_type: str):
    # Mapping of case type to module path
    PROMPT_MODULES = {
        "shoplifting": "prompts.generate_prompt_shoplifting",
        "domestic_abuse": "prompts.generate_prompt_domestic_abuse",
        "terrorism": "prompts.generate_prompt_terrorism",
    }
    try:
        prompt_module = importlib.import_module(PROMPT_MODULES[case_type])
    except KeyError:
        raise ValueError(f"Unknown case type: {case_type}")
    except ImportError as e:
        raise ImportError(f"Could not import module for {case_type}: {e}")
    prompt = prompt_module.prompt
    system_message = prompt_module.system_message
    logit_bias = prompt_module.build_logit_bias()
    return prompt, system_message, logit_bias


free_token_models = [
    "gpt-4.5-preview",
    "gpt-4.5-preview-2025-02-27",
    "gpt-4.1-2025-04-14",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "o3-2025-04-16",
    "o1-preview-2024-09-12",
    "o1-2024-12-17",
]
