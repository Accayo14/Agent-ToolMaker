"""
Trajectory visualization — saves agent conversations as readable markdown files.
One file per tool-creation run, stored in trajectories/.
Mirrors the paper's trajectory visualization feature.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

TRAJECTORY_DIR = Path(__file__).parent.parent / "trajectories"


def save(
    task_name: str,
    repo_url: str,
    phases: list[dict],   # [{"phase": "Install", "trajectory": [...], "status": str}, ...]
    result_code: str = "",
    result_tests: str = "",
    cost_usd: float = 0.0,
) -> Path:
    """
    Write a markdown trajectory file for a completed run.
    Returns the path to the written file.
    """
    TRAJECTORY_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = TRAJECTORY_DIR / f"{task_name}_{ts}.md"

    lines = [
        f"# Trajectory: {task_name}",
        f"",
        f"| Field | Value |",
        f"|---|---|",
        f"| Repo | {repo_url} |",
        f"| Date | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} |",
        f"| Cost | ${cost_usd:.4f} |",
        f"",
    ]

    for phase_info in phases:
        phase_name = phase_info["phase"]
        status     = phase_info.get("status", "")
        trajectory = phase_info.get("trajectory", [])

        lines += [f"## Phase: {phase_name}  `{status}`", ""]

        step = 0
        for msg in trajectory:
            role = msg.get("role", "")

            if role == "system":
                continue  # skip system prompt in output

            elif role == "user":
                content = msg.get("content", "")
                if content.startswith("<tool_results>") or not content.strip():
                    continue
                lines += [f"**User:**", f"> {content[:300]}", ""]

            elif role == "assistant":
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])

                if content:
                    lines += [f"**Assistant:** {content[:400]}", ""]

                for tc in tool_calls:
                    step += 1
                    fn   = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except Exception:
                        args = {}

                    # Show the tool call clearly
                    arg_preview = _format_args(fn, args)
                    lines += [
                        f"### Step {step} — `{fn}`",
                        f"```",
                        arg_preview,
                        f"```",
                        "",
                    ]

            elif role == "tool":
                result = msg.get("content", "")
                lines += [
                    f"**Result:**",
                    f"```",
                    result[:800],
                    f"```",
                    "",
                ]

    if result_code:
        lines += [
            "---",
            "## Generated Tool Code",
            "```python",
            result_code,
            "```",
            "",
        ]

    if result_tests:
        lines += [
            "## Generated Tests",
            "```python",
            result_tests,
            "```",
            "",
        ]

    out_path.write_text("\n".join(lines))
    return out_path


def _format_args(tool_name: str, args: dict) -> str:
    if tool_name == "run_bash":
        return f"$ {args.get('command', '')}"
    if tool_name == "read_file":
        return f"read: {args.get('path', '')}"
    if tool_name == "write_file":
        path    = args.get("path", "")
        preview = (args.get("content", "")[:200] + "...") if len(args.get("content", "")) > 200 else args.get("content", "")
        return f"write: {path}\n{preview}"
    if tool_name == "list_directory":
        return f"ls: {args.get('path', '')}"
    if tool_name == "browse":
        return f"browse: {args.get('url', '')}"
    return json.dumps(args, indent=2)
