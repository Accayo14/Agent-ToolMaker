"""
Agent and LLM utilities — faithful to the ToolMaker paper's architecture.

The paper uses:
  - Agentic loops for: install, explore_repository, diagnose
  - Single LLM calls for: make_plan, implement_function, assess, rewrite, summarize
  - Tools include a `reasoning` field (agent explains why it uses each tool)
  - Read-only agent (no write_file) for explore phase
  - Structured output (Pydantic) for assess and diagnose responses

Model: gpt-4o-mini throughout (paper uses gpt-4o + o1-mini; budget constraint).
"""
import json
import logging
import os
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"

_COST_PER_INPUT_TOKEN  = 0.15 / 1_000_000
_COST_PER_OUTPUT_TOKEN = 0.60 / 1_000_000
BUDGET_LIMIT_USD = 1.80

_session_spend_usd: float = 0.0


class BudgetExceeded(RuntimeError):
    pass


def session_spend() -> float:
    return _session_spend_usd


def _charge(usage) -> float:
    global _session_spend_usd
    cost = (usage.prompt_tokens * _COST_PER_INPUT_TOKEN +
            usage.completion_tokens * _COST_PER_OUTPUT_TOKEN)
    _session_spend_usd += cost
    return cost


def _client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ── Pydantic schemas for structured outputs ───────────────────────────────────

class AssessmentResult(BaseModel):
    successful: bool = Field(description="Whether the function call was successful and the task is complete.")
    reasoning: str = Field(description="The reasoning for the assessment.")


class DiagnosisResult(BaseModel):
    diagnosis: str = Field(description="A diagnosis of the issue that occurred.")
    plan: str = Field(description="A plan to fix the issue.")


# ── Tool schemas (all include `reasoning` field, matching paper) ──────────────

# All tools — for install and diagnose agents
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_bash_command",
            "description": (
                "Run a non-interactive bash command in the sandbox. Returns combined stdout + stderr. "
                "Always prefer to run a single command at a time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "One-sentence explanation of why this command is being run and what it should accomplish."},
                    "command":   {"type": "string", "description": "The bash command to run."},
                },
                "required": ["reasoning", "command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read content from a file. Do not use for image/audio/binary files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "One-sentence explanation of why this file is being read."},
                    "path":      {"type": "string", "description": "Absolute file path."},
                },
                "required": ["reasoning", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "One-sentence explanation of what is being written and why."},
                    "path":      {"type": "string", "description": "Absolute file path."},
                    "content":   {"type": "string", "description": "Content to write."},
                },
                "required": ["reasoning", "path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at a given path (up to 3 levels deep).",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "One-sentence explanation of why this directory is being listed."},
                    "path":      {"type": "string", "description": "Directory to list."},
                },
                "required": ["reasoning", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse",
            "description": "Fetch a URL and return readable text content (HTML stripped). Use for documentation, PyPI pages, GitHub READMEs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "One-sentence explanation of why this URL is being fetched."},
                    "url":       {"type": "string", "description": "URL to fetch."},
                },
                "required": ["reasoning", "url"],
            },
        },
    },
]

# Read-only tools — for explore_repository (no write_file, per paper)
TOOLS_READONLY = [t for t in TOOLS if t["function"]["name"] != "write_file"]


def _execute_tool(name: str, args: dict, sandbox) -> str:
    try:
        if name == "run_bash_command":
            return sandbox.run_bash(args["command"])
        if name == "read_file":
            return sandbox.read_file(args["path"])
        if name == "write_file":
            return sandbox.write_file(args["path"], args["content"])
        if name == "list_directory":
            default_dir = getattr(sandbox, "work_dir", "/home/user")
            return sandbox.list_directory(args.get("path", default_dir))
        if name == "browse":
            return sandbox.browse(args["url"])
        return f"Unknown tool: {name}"
    except KeyError as e:
        return f"Missing argument: {e}"
    except Exception as e:
        return f"Tool error ({name}): {e}"


# ── Agentic loop ──────────────────────────────────────────────────────────────

def run_agent(
    system: str,
    user: str,
    sandbox,
    max_steps: int = 40,
    readonly: bool = False,
) -> dict:
    """
    Run the agentic loop (paper's Agent.run()).

    readonly=True  → explore_repository agent (no write_file)
    readonly=False → install and diagnose agents (all tools)

    The agent terminates when it outputs a response with no tool calls
    (i.e., it has finished and produced its final answer).

    Returns:
      {
        "final_message": str,
        "steps":         int,
        "cost_usd":      float,
        "messages":      list[dict],   # full conversation for continuation
      }
    """
    global _session_spend_usd
    if _session_spend_usd >= BUDGET_LIMIT_USD:
        raise BudgetExceeded(f"Budget limit ${BUDGET_LIMIT_USD} reached (${_session_spend_usd:.4f} spent)")

    client = _client()
    tools = TOOLS_READONLY if readonly else TOOLS
    run_cost = 0.0
    steps = 0

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    for _ in range(max_steps):
        if _session_spend_usd >= BUDGET_LIMIT_USD:
            raise BudgetExceeded(f"Budget limit ${BUDGET_LIMIT_USD} reached")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=4096,
            temperature=0.0,
        )
        call_cost = _charge(response.usage)
        run_cost += call_cost

        choice = response.choices[0]
        msg = choice.message

        msg_dict: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        # Agent finished — no more tool calls
        if choice.finish_reason == "stop" or not msg.tool_calls:
            logger.info(f"  Agent done | steps={steps} cost=${run_cost:.4f} session=${_session_spend_usd:.4f}")
            return {
                "final_message": msg.content or "",
                "steps": steps,
                "cost_usd": run_cost,
                "messages": messages,
            }

        # Execute tool calls
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            logger.info(f"    tool={tc.function.name} | {args.get('reasoning', '')[:80]}")
            result = _execute_tool(tc.function.name, args, sandbox)
            steps += 1

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)[:4000],
            })

    logger.info(f"  Agent max_steps | cost=${run_cost:.4f}")
    return {
        "final_message": messages[-1].get("content", "") if messages else "",
        "steps": steps,
        "cost_usd": run_cost,
        "messages": messages,
    }


# ── Single LLM call (no tools) ────────────────────────────────────────────────

def call_llm(messages: list[dict]) -> tuple[str, float]:
    """
    Single LLM completion call with no tools. Used for plan, implement, rewrite, summarize.
    Returns (text_response, cost_usd).
    """
    global _session_spend_usd
    if _session_spend_usd >= BUDGET_LIMIT_USD:
        raise BudgetExceeded(f"Budget limit ${BUDGET_LIMIT_USD} reached")

    client = _client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=4096,
        temperature=0.0,
    )
    cost = _charge(response.usage)
    text = response.choices[0].message.content or ""
    logger.info(f"  LLM call | cost=${cost:.4f} session=${_session_spend_usd:.4f}")
    return text, cost


def call_llm_structured(messages: list[dict], schema: type[BaseModel]) -> tuple[Any, float]:
    """
    Single LLM completion with structured JSON output (paper's typed_call).
    Returns (parsed_pydantic_object, cost_usd).
    """
    global _session_spend_usd
    if _session_spend_usd >= BUDGET_LIMIT_USD:
        raise BudgetExceeded(f"Budget limit ${BUDGET_LIMIT_USD} reached")

    client = _client()
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=messages,
        response_format=schema,
        max_tokens=2048,
        temperature=0.0,
    )
    cost = _charge(response.usage)
    parsed = response.choices[0].message.parsed
    logger.info(f"  LLM structured | cost=${cost:.4f} session=${_session_spend_usd:.4f}")
    return parsed, cost
