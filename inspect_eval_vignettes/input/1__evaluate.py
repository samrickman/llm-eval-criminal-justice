from eval_utils import SYSTEM_MESSAGE
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import generate, system_message
from scorers import includes_list


@task
def eval_shoplifting():
    dataset = json_dataset("./input/shoplifting.json")
    return Task(
        dataset=dataset,
        solver=[system_message(SYSTEM_MESSAGE), generate()],
        scorer=includes_list(),
    )


@task
def eval_terrorism():
    dataset = json_dataset("./input/terrorism.json")
    return Task(
        dataset=dataset,
        solver=[system_message(SYSTEM_MESSAGE), generate()],
        scorer=includes_list(),
    )
