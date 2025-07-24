from typing import Callable, Literal

from eval_utils import remove_punctuation
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.scorer import exact, includes
from inspect_ai.scorer._common import match_str, str_match_scorer
from inspect_ai.scorer._metric import CORRECT, INCORRECT, Score
from inspect_ai.scorer._metrics import accuracy, stderr
from inspect_ai.scorer._scorer import Scorer, scorer
from inspect_ai.scorer._target import Target
from inspect_ai.solver import generate, system_message
from inspect_ai.solver._task_state import TaskState


def list_match_scorer(match: Callable[[str, str], tuple[str, bool]]) -> Scorer:
    """Scorer that uses a matching function.

    The matching function returns tuple[str,bool], where str is the answer
    extracted from the model output and bool is whether it matched the target
    """

    async def score(state: TaskState, target: Target) -> Score:
        answer: str | None = None
        correct: list[str] = []
        missed: list[str] = []

        for value in target:
            answer, matched = match(state.output.completion, value)
            if matched:
                correct.append(value)
            else:
                missed.append(value)

        accuracy = len(correct) / len(target)
        explanation = f"""
        Matched words: {correct}
        Missed words: {missed}
        """

        metadata = {"accuracy": accuracy, "missed": missed, "matched": correct}

        return Score(
            value=accuracy, answer=answer, explanation=explanation, metadata=metadata
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def includes_list(ignore_case: bool = True, remove_punct=True) -> Scorer:
    """Check whether the specified text is included in the model output.
    Takes a list of target words and returns a simple accuracy measure.

    Args:
       ignore_case: Use a case insensitive comparison.

    """

    def check(value: str, target: str) -> tuple[str, bool]:
        if ignore_case:
            value = value.casefold()
            target = target.casefold()
        if remove_punct:
            value, target = remove_punctuation(value, target)
        return value, target in value

    return list_match_scorer(check)
