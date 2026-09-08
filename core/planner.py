from core.llm import chat
from core.models import TaskSpec

_SYSTEM = """\
You are an expert software engineer specializing in wrapping scientific Python repositories \
into callable functions for LLM agents.

Given a repository's source context and a task description, produce a concrete, numbered \
implementation plan. Your plan must cover:
1. The exact pip packages to install (comma-separated, ready for `pip install`)
2. Which file/function to call as the entry point
3. How to invoke it with the provided example inputs
4. What the wrapper function signature should look like
5. How to capture and return the output
6. Two or three simple assertions to verify correctness

Be specific. Name exact functions, modules, and arguments. Think step by step.\
"""


def create_plan(repo_context: str, task: TaskSpec) -> str:
    user = f"""\
## Repository Context
{repo_context}

## Task
Name: {task.name}
Description: {task.description}
Inputs: {task.inputs}
Expected output: {task.expected_output}

Write the implementation plan now.\
"""
    return chat(_SYSTEM, user, temperature=0.1)
