import re
from core.llm import chat
from core.models import TaskSpec, GeneratedCode

_SYSTEM = """\
You are an expert Python developer. Given an implementation plan, write production-quality code.

You MUST output exactly two fenced code blocks and nothing else outside them:

```python
# install_cmd
pip install <space-separated packages>
```

```python
# wrapper_code
def run_tool(**kwargs):
    # implement here
    ...
    return result  # must be a dict
```

```python
# test_code
import pytest
from tool import run_tool

def test_basic():
    result = run_tool(...)
    assert ...
```

Rules:
- `run_tool` must accept **kwargs and return a dict
- tests must import from `tool` (the file will be saved as tool.py)
- tests must be self-contained and not require manual setup
- if a previous error is shown, fix the exact root cause\
"""


def _extract_block(text: str, tag: str) -> str:
    pattern = rf"```python\s*# {tag}\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def generate_code(plan: str, task: TaskSpec, prev_error: str = "") -> GeneratedCode:
    error_section = ""
    if prev_error:
        error_section = f"\n\n## Previous Attempt Failed\n```\n{prev_error[:2000]}\n```\nFix the root cause."

    user = f"""\
## Implementation Plan
{plan}

## Task Inputs Example
{task.inputs}

## Expected Output
{task.expected_output}{error_section}

Generate the three code blocks now.\
"""
    raw = chat(_SYSTEM, user, temperature=0.15)

    install_line = _extract_block(raw, "install_cmd")
    wrapper = _extract_block(raw, "wrapper_code")
    tests = _extract_block(raw, "test_code")

    # Fallback: if tag extraction failed, try generic fence extraction
    if not install_line:
        blocks = re.findall(r"```python\s*(.*?)```", raw, re.DOTALL)
        install_line = blocks[0].strip() if len(blocks) > 0 else "pip install"
        wrapper = blocks[1].strip() if len(blocks) > 1 else raw
        tests = blocks[2].strip() if len(blocks) > 2 else ""

    return GeneratedCode(
        install_cmd=install_line,
        wrapper_code=wrapper,
        test_code=tests,
    )
