"""
Faithful reproduction of the ToolMaker paper (Wölflein et al., ACL 2025).

Matches the paper's exact flow from make_tool.py / install.py / make_plan.py /
implement_function.py / assess.py / diagnose.py / rewrite_function.py:

  Phase 1 — install_repository (install.py):
    Agent clones repo, reads README, installs everything (including model weights).
    Task description included as context so agent installs what's needed.
    max_steps = 20 (paper default).
    Terminates when agent produces a summary of what it installed.

  Phase 2 — make_tool (make_tool.py):
    2a. explore_repository (make_plan.py):
        Read-only agent explores repo, returns a one-paragraph summary.
    2b. make_plan (make_plan.py):
        Single LLM call. Writes numbered pseudocode plan.
    2c. implement_function (implement_function.py):
        Single LLM call. Writes the full Python function.

    Correction loop (max_iterations = 30, paper default):
        i.   Reset sandbox to post-install checkpoint.
        ii.  Execute function with example inputs.
        iii. is_successful_execution (assess.py):
             LLM judges whether the output is correct. Structured output.
        iv.  If successful → done.
        v.   diagnose (diagnose.py):
             Agent explores the failure, returns {diagnosis, plan}. Structured output.
        vi.  rewrite_function (rewrite_function.py):
             Single LLM call. Rewrites the function.
        vii. summarize_problem (make_tool.py):
             Single LLM call. Summarizes problem for future context.

  Post-loop — TM-Bench evaluation (paper Section 4):
    Run human-written validation_tests once in a fresh sandbox.
    NOT fed back to agent — only used for reporting.

Sandbox backends (our adaptation — paper uses Docker):
  GPU tasks  → LocalSandbox (single sandbox, reset only implementation.py between iters)
  CPU tasks  → AgentSandbox/E2B (fresh sandbox per iter, install commands replayed)
"""
import json
import logging
import re
import time
import uuid

from core.agent import (
    AssessmentResult, DiagnosisResult,
    call_llm, call_llm_structured, run_agent, session_spend,
)
from core.models import TaskSpec, ToolResult
from core.sandbox import AgentSandbox, LocalSandbox, local_gpu_available, run_validation_sandbox
from core.trajectory import save as save_trajectory
from registry.store import init_db, upsert_tool, export_markdown

logger = logging.getLogger(__name__)


# ── Paper's system prompt (install.py SYSTEM_PROMPT) ─────────────────────────

def _agent_system_prompt(work_dir: str, repo_name: str = "", installed: bool = False) -> str:
    base = (
        "You're a diligent software engineer AI. You can't see, draw, or interact with "
        "a browser, but you can read and write files, and you can run commands, and you can think. "
        "The user will specify a task for you to complete. You likely need to run several actions "
        "in order to complete the task. You will only be able to execute a single action at a time.\n\n"
        "Use the tools (actions) that are at your disposal. "
        "Each time you invoke a tool, provide a one-sentence summary in the `reasoning` field of "
        "why you are invoking it and what you expect to accomplish by invoking it.\n\n"
        f"Your workspace directory and current working directory is `{work_dir}`.\n\n"
        "You will continue the process of invoking tools until you have completed the task."
    )
    if installed and repo_name:
        base += f"\n\nYou have already installed the {repo_name} repository and its dependencies at `{work_dir}/repo`."
    return base


# ── Helper: derive function signature from task ───────────────────────────────

def _py_type(val) -> str:
    if isinstance(val, bool):   return "bool"
    if isinstance(val, int):    return "int"
    if isinstance(val, float):  return "float"
    if isinstance(val, list):   return "list"
    if isinstance(val, dict):   return "dict"
    return "str"


def _function_signature(task: TaskSpec) -> str:
    """Generate typed function signature from task spec, matching paper's definitions format."""
    args = ", ".join(
        f"{k}: {_py_type(v)}"
        for k, v in task.inputs.items()
    )
    return f"def {task.name}({args}) -> dict:"


def _coding_instructions(task: TaskSpec, work_dir: str) -> str:
    """Matches paper's coding_instructions() from implement_function.py."""
    return f"""
You **must** output a valid, standalone python function that is callable without any modification by a user.
The requirements for the code are:
1. Import the required modules/libraries inside the function body.
2. You are only allowed to write a single python function. It must start with 'def ...' and end with 'return ...'.
3. You are not allowed to output free text, test code, or anything outside of the function definition.
4. The function needs to be a standalone function that can be called independently.
5. Make sure all required imports are included inside the function.
6. The function must perform the task you are given. As a reminder, the task is: `{task.description}`.
7. The function must accept all required parameters as inputs.
8. The function must have type hints and a docstring.
9. The function must be named exactly `{task.name}`.
10. The function must be a valid python function, executable by a python interpreter.

Additional instructions:
* Write the function so it can easily be debugged. Include print statements for logging, especially for long-running tasks.
* When catching exceptions, output the entire stack trace to stderr using `traceback.format_exc()`.
* When running subprocesses, stream stdout and stderr to the parent process.
* Do NOT run interactive commands.
* Always prefer to import existing functions from the repository, or run existing scripts/modules via subprocess, rather than re-implementing functionality yourself.
* The repository is installed at `{work_dir}/repo`.

Respond with the code of the function only, without any other text.
"""


def _extract_code(text: str) -> str:
    """Extract Python code from LLM response — handles ```python ... ``` blocks."""
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


# ── Phase 1: install_repository ───────────────────────────────────────────────

def _install_user_prompt(task: TaskSpec, work_dir: str) -> str:
    """Matches paper's install.py user_prompt."""
    repo_name = task.repo_url.rstrip("/").split("/")[-1]
    return f"""Clone and locally set up the {repo_name} repository from GitHub.
Follow these steps:
1. Git clone the repository {task.repo_url} into the directory `{work_dir}/repo`.
2. Check the README (find it if it is not in the root directory) and closely follow the recommended instructions to set up the entire repository correctly for the user.
3. Follow the instructions in the README to correctly set up the repository. Perform any necessary installations, configurations, downloads or setups as described. If the repository is in Python, prefer using `pip` as opposed to conda, virtualenv, or similar. Install the repository and its dependencies globally. Do not use Docker or similar container tools; instead, install the repository and its dependencies directly.
4. Make sure that you complete every step, so that a user could directly use this repository without the need to do further setups, installations or downloads. This includes downloading any necessary pretrained models. However, do NOT download any datasets.
If you encounter any issues, try to solve them.

You should set up the repository in such a way that it can be used to implement the following task later on:
<intended_task>
{task.description}
Example inputs: {json.dumps(task.inputs)}
Expected output: {task.expected_output}
</intended_task>
IMPORTANT: Your task right now is to only set up the repository, NOT implement this task.

Continue calling tools until you are done and have installed and set up the repository.
Once you are done, provide a brief summary of what you did and what you accomplished, as well as the absolute path to the cloned and installed repository."""


# ── Phase 2a: explore_repository ─────────────────────────────────────────────

def _explore_user_prompt(task: TaskSpec, work_dir: str, install_summary: str) -> str:
    """Matches paper's explore_repository user_prompt from make_plan.py."""
    sig = _function_signature(task)
    return f"""# Background
The repository is fully set up and installed at `{work_dir}/repo`.
We need to wrap a specific functionality from this repository into a standalone python function that can be called independently.
This function will be called `{task.name}`, and it is described as follows:
<description>
{task.description}
</description>

The function will have the following arguments:
<arguments>
{chr(10).join(f"<argument>{k}: {_py_type(v)} (example: {v!r})</argument>" for k, v in task.inputs.items())}
</arguments>

As such, the signature of the function will be:
```python
{sig}
```

Expected output:
<expected_result>
{task.expected_output}
</expected_result>

Installation summary:
<install_summary>
{install_summary}
</install_summary>

# High-level approach
In order to implement this function, you will follow these steps:
1. Explore the repository to gather all relevant information needed to write the plan.
2. Write a plan for the body/implementation of the function.
3. Write the function, based on the plan.

# Task
Right now, you are at step 1: Explore the repository to gather all relevant information needed to write the plan.
This step is very important — you must be thorough because you will rely on this information when implementing the function.
Use the tools at your disposal to read files, list directories, browse documentation, etc.
HINT 1: If the repository contains a README file, that is often a good starting point.
HINT 2: If the repository provides a command line interface, prefer to invoke it via subprocess, rather than calling the underlying python functions. Only as a last resort, wrap python functions directly.
HINT 3: Do NOT attempt to read image files, audio files, or binary files.
Read ALL files (documentation, code, configuration files) that are necessary to implement the function.
Read relevant code files to understand how the functionality is implemented, including input/output argument formats.
Do NOT write the function yet. Your task is specifically to explore the repository to gather information.

Once you have gathered ALL relevant information, respond with a one-paragraph summary of what you found.

Remember, the function should do the following:
<description>
{task.description}
</description>"""


# ── Phase 2b: make_plan ───────────────────────────────────────────────────────

def _plan_user_prompt(task: TaskSpec) -> str:
    """Matches paper's make_plan user_prompt."""
    sig = _function_signature(task)
    return f"""Using the information you gathered previously, your task is now to write an outline (plan) for the body/implementation of the function.
This plan should be in the form of very high-level pseudo-code, that describes how the function will work.
It should be a numbered list of steps, each of which describes what you will do in that step.
Respond with just this list of steps, nothing else.
Remember, the function should do the following: `{task.description}`

As such, the signature of the function will be:
```python
{sig}
```"""


# ── Phase 2c: implement_function ─────────────────────────────────────────────

def _implement_user_prompt(task: TaskSpec, plan: str, work_dir: str) -> str:
    """Matches paper's implement_function user_prompt."""
    sig = _function_signature(task)
    return f"""Now that you have identified the plan for the implementation, you need to write the actual implementation of the function.
This needs to be a standalone python function that can be called independently.
This function will be called `{task.name}`, and it is described as follows: `{task.description}`
The function will have the following arguments:
{chr(10).join(f"- {k}: {_py_type(v)} (example: {v!r})" for k, v in task.inputs.items())}

As such, the signature of the function will be:
```python
{sig}
```

Your task is now to write the Python function.
To do so, follow the plan you identified earlier for the implementation:
<plan>
{plan}
</plan>

{_coding_instructions(task, work_dir)}

Remember, you should use the repository installed at `{work_dir}/repo` to complete the task.
Finally, ensure your function is ready-to-use without any modifications by a user. In many cases, wrapping an existing function, script or module in a subprocess is enough.
Respond with the code of the function only, without any other text."""


# ── Correction loop: execute function ────────────────────────────────────────

def _make_runner(task: TaskSpec, work_dir: str) -> str:
    """Python script that imports and calls the function with example inputs."""
    return f"""import sys, json, traceback
sys.path.insert(0, {repr(work_dir)})

try:
    from implementation import {task.name}
    inputs = {repr(task.inputs)}
    result = {task.name}(**inputs)
    print(json.dumps({{"status": "success", "result": result}}, default=str))
except Exception:
    tb = traceback.format_exc()
    print(json.dumps({{"status": "error", "traceback": tb}}))
    sys.exit(1)
"""


def _execute_function(task: TaskSpec, sandbox, work_dir: str, code: str) -> dict:
    """
    Write implementation.py + runner.py, execute, return {status, stdout, result}.
    Matches paper's runtime.run_function(FunctionCall(...)).
    """
    sandbox.write_file(f"{work_dir}/implementation.py", code)
    runner = _make_runner(task, work_dir)
    sandbox.write_file(f"{work_dir}/runner.py", runner)
    stdout = sandbox.run_bash(f"cd {work_dir} && python runner.py 2>&1", log=False, timeout=300)
    try:
        parsed = json.loads(stdout.strip().split("\n")[-1])
        status = parsed.get("status", "error")
        result_str = json.dumps(parsed.get("result", ""), indent=2) if status == "success" else ""
    except Exception:
        status = "error"
        result_str = ""
    return {"status": status, "stdout": stdout, "result_str": result_str}


# ── Correction loop: is_successful_execution ──────────────────────────────────

def _assess_prompt(task: TaskSpec, exec_result: dict) -> str:
    """Matches paper's is_successful_execution prompt from assess.py."""
    return f"""I executed the function you wrote.
Based on the output and returned result, assess whether the function call was successful or not.
Specifically, you should assess whether the function performed the task it was supposed to perform.
Also make sure that the returned result is plausible and matches the stdout/stderr output logs, if applicable.
As a reminder, the task is the following:
<task_description>
{task.description}
</task_description>

Description of expected result:
<expected_result_description>
{task.expected_output}
</expected_result_description>

Returned result:
<result>
{exec_result['result_str'][:5000] if exec_result['result_str'] else '(none — function raised an exception)'}
</result>

Output (stdout and stderr) of the function execution:
<output>
{exec_result['stdout'][:10000]}
</output>

**IMPORTANT: You must also ensure that the returned result itself is correct. This includes ensuring that the result dict contains the correct keys and values, and that the values have the correct types and shapes! If any of these are incorrect, the function call is NOT successful! If this is the case, include this in your reasoning.**"""


# ── Correction loop: diagnose ─────────────────────────────────────────────────

def _diagnose_user_prompt(
    code: str,
    exec_result: dict,
    problem_summaries: list[str],
    assessment: AssessmentResult,
) -> str:
    """Matches paper's diagnose user_prompt from diagnose.py."""
    summaries_xml = "\n".join(
        f"<summary number={i}>{s}</summary>"
        for i, s in enumerate(problem_summaries)
    )
    return f"""Your initial code implementation did not work. This was attempt number {len(problem_summaries)} to fix the problem.

Here is a summary of the previous problems and your attempts to fix them. Keep this in mind and avoid repeating the same mistakes.
<summaries>
{summaries_xml}
</summaries>

The current version of your code (after {len(problem_summaries)} attempts) is below.
IMPORTANT: this is the most up-to-date version of your code, so focus on it when diagnosing the problem.
```python
{code}
```

Upon executing this updated function, I received another error.
As a diligent software engineer AI, your task is now to diagnose the issue and fix the function.
You can read and write files and run commands to gather information about the issue.
Try to find the root cause. Often this requires reading relevant code files in the repository.

IMPORTANT: Any changes you make to the environment will be lost when the function is executed again, as the environment will be reset. Use this opportunity only to gather information about the issue, not to fix it.
HINT: After gathering information, you may decide to use a slightly different approach — include this in your plan!
HINT: Always prefer importing code from the repository rather than implementing it yourself.

Output (stdout and stderr) of the function execution:
<output>
{exec_result['stdout'][:10000]}
</output>

Initial assessment why the function call was not successful:
<assessment>
{assessment.reasoning}
</assessment>

Your immediate task is to diagnose the issue and formulate a plan to fix it.
Once done, respond with your diagnosis and plan."""


# ── Correction loop: rewrite_function ────────────────────────────────────────

def _rewrite_user_prompt(task: TaskSpec, code: str, diagnosis: DiagnosisResult, work_dir: str) -> str:
    """Matches paper's rewrite_function user_prompt from rewrite_function.py."""
    sig = _function_signature(task)
    return f"""Now that you have identified the problem as well as a plan to fix the function, you need to write the updated implementation of the function.
Remember, the function is called `{task.name}`, and it is described as follows: `{task.description}`
The function will have the following arguments:
{chr(10).join(f"- {k}: {_py_type(v)} (example: {v!r})" for k, v in task.inputs.items())}

As such, the signature of the function will be:
```python
{sig}
```

Your task is now to write the Python function.
To do so, use the information you gathered above to fix the function.

{_coding_instructions(task, work_dir)}

As a reminder, the current draft of the function is:
```python
{code}
```

Remember, your diagnosis is:
<diagnosis>
{diagnosis.diagnosis}
</diagnosis>

And your plan to fix the issue is:
<plan>
{diagnosis.plan}
</plan>

Respond with the updated function code only, without any other text."""


# ── Correction loop: summarize_problem ───────────────────────────────────────

_SUMMARIZE_PROMPT = """Provide a one-paragraph summary of the most recent problem that occurred, your diagnosis of it, and how you attempted to fix it with this code change.
Be specific. Include any file paths and other details that are relevant to the problem/solution.
Your summary should contain all information needed to implement the fix, and include the key insights/observations made for diagnosing the problem.
Begin your response with "The problem was..." """


# ── Public entry point ────────────────────────────────────────────────────────

def run_closed_loop(
    task: TaskSpec,
    max_iterations: int = 10,
    progress_cb=None,
) -> ToolResult:
    init_db()
    job_id = str(uuid.uuid4())
    upsert_tool(job_id, task.name, task.repo_url, {}, status="running")

    use_local = task.requires_gpu and local_gpu_available()

    if task.requires_gpu and not use_local:
        logger.warning(f"[{task.name}] Requires GPU but none found locally")
        r = ToolResult(
            name=task.name, status="gpu_required",
            last_error="Task requires a CUDA GPU. None found on this server.",
        )
        upsert_tool(job_id, task.name, task.repo_url, {}, status="gpu_required", last_error=r.last_error)
        export_markdown()
        return r

    backend = "LocalSandbox (GPU)" if use_local else "AgentSandbox (E2B)"
    logger.info(f"[{task.name}] Backend: {backend}")

    if use_local:
        return _run_local(task, job_id, max_iterations, progress_cb)
    return _run_e2b(task, job_id, max_iterations, progress_cb)


# ── GPU / local path ──────────────────────────────────────────────────────────

def _run_local(task: TaskSpec, job_id: str, max_iterations: int, progress_cb) -> ToolResult:
    sbx = LocalSandbox()
    phase_trajectories = []

    try:
        work = sbx.work_dir
        sbx.run_bash(f"git clone --depth 1 {task.repo_url} {work}/repo 2>&1", log=False)

        # Phase 1: Install
        install_summary, install_msgs = _install_phase(task, sbx, work, phase_trajectories)
        if install_summary is None:
            return _record_failure(task, job_id, 0, "Install failed", phase_trajectories)

        # Phase 2: explore + plan + implement (once, on same sandbox)
        code, messages_checkpoint = _explore_plan_implement(
            task, sbx, work, install_summary, phase_trajectories
        )
        if code is None:
            return _record_failure(task, job_id, 0, "Failed to produce initial implementation", phase_trajectories)

        # Correction loop
        return _correction_loop(
            task, job_id, max_iterations, progress_cb,
            code, messages_checkpoint, phase_trajectories,
            install_commands=None, use_local=True, local_sbx=sbx, work=work,
        )

    finally:
        sbx.kill()


# ── CPU / E2B path ────────────────────────────────────────────────────────────

def _run_e2b(task: TaskSpec, job_id: str, max_iterations: int, progress_cb) -> ToolResult:
    phase_trajectories = []
    install_commands: list[str] = []

    # Phase 1: Install
    install_sbx = AgentSandbox(timeout=600)
    try:
        work = install_sbx.work_dir
        install_sbx.run_bash(f"git clone --depth 1 {task.repo_url} {work}/repo 2>&1", log=False)
        install_summary, _ = _install_phase(task, install_sbx, work, phase_trajectories)
        install_commands = install_sbx.command_log
    finally:
        install_sbx.kill()

    if install_summary is None:
        return _record_failure(task, job_id, 0, "Install failed", phase_trajectories)

    # Phase 2: explore + plan + implement (in a fresh sandbox)
    explore_sbx = AgentSandbox(timeout=600)
    code = None
    messages_checkpoint = None
    try:
        work = explore_sbx.work_dir
        explore_sbx.run_bash(f"git clone --depth 1 {task.repo_url} {work}/repo 2>&1", log=False)
        logger.info(f"[{task.name}] Replaying {len(install_commands)} install commands for explore")
        for cmd in install_commands:
            explore_sbx.run_bash(cmd, log=False)
        code, messages_checkpoint = _explore_plan_implement(
            task, explore_sbx, work, install_summary, phase_trajectories
        )
    finally:
        explore_sbx.kill()

    if code is None:
        return _record_failure(task, job_id, 0, "Failed to produce initial implementation", phase_trajectories)

    # Correction loop (fresh E2B sandbox per iteration)
    return _correction_loop(
        task, job_id, max_iterations, progress_cb,
        code, messages_checkpoint, phase_trajectories,
        install_commands=install_commands, use_local=False, local_sbx=None, work=None,
    )


# ── Shared: install phase ─────────────────────────────────────────────────────

def _install_phase(
    task: TaskSpec, sandbox, work: str, phase_trajectories: list
) -> tuple[str | None, list]:
    """
    Runs the install agent (paper's install_repository).
    Returns (install_summary, messages) or (None, []) on failure.
    max_steps=20 matching paper default.
    """
    logger.info(f"[{task.name}] Phase 1: Install")
    result = run_agent(
        system=_agent_system_prompt(work),
        user=_install_user_prompt(task, work),
        sandbox=sandbox,
        max_steps=20,
        readonly=False,
    )
    logger.info(
        f"[{task.name}] Install done | steps={result['steps']} "
        f"cost=${result['cost_usd']:.4f} session=${session_spend():.4f}"
    )
    phase_trajectories.append({
        "phase": "Install",
        "trajectory": result["messages"],
    })

    summary = result["final_message"].strip()
    if not summary:
        logger.warning(f"[{task.name}] Install produced no summary")
        return None, []

    logger.info(f"[{task.name}] Install summary: {summary[:120]}")
    return summary, result["messages"]


# ── Shared: explore + plan + implement ───────────────────────────────────────

def _explore_plan_implement(
    task: TaskSpec, sandbox, work: str, install_summary: str, phase_trajectories: list
) -> tuple[str | None, list]:
    """
    explore_repository → make_plan → implement_function.
    Returns (code, messages_checkpoint) or (None, []).
    """
    repo_name = task.repo_url.rstrip("/").split("/")[-1]

    # 2a: explore_repository (read-only agent)
    logger.info(f"[{task.name}] Phase 2a: Explore (read-only agent)")
    explore_result = run_agent(
        system=_agent_system_prompt(work, repo_name=repo_name, installed=True),
        user=_explore_user_prompt(task, work, install_summary),
        sandbox=sandbox,
        max_steps=40,
        readonly=True,
    )
    logger.info(
        f"[{task.name}] Explore done | steps={explore_result['steps']} "
        f"cost=${explore_result['cost_usd']:.4f} session=${session_spend():.4f}"
    )
    phase_trajectories.append({
        "phase": "Explore",
        "trajectory": explore_result["messages"],
    })
    explore_summary = explore_result["final_message"].strip()

    # 2b: make_plan (single LLM call)
    logger.info(f"[{task.name}] Phase 2b: Plan")
    messages_for_plan = explore_result["messages"] + [
        {"role": "user", "content": _plan_user_prompt(task)}
    ]
    plan, _ = call_llm(messages_for_plan)
    logger.info(f"[{task.name}] Plan: {plan[:120]}")
    messages_after_plan = messages_for_plan + [
        {"role": "assistant", "content": plan}
    ]
    phase_trajectories.append({"phase": "Plan", "content": plan})

    # 2c: implement_function (single LLM call)
    logger.info(f"[{task.name}] Phase 2c: Implement")
    messages_for_impl = messages_after_plan + [
        {"role": "user", "content": _implement_user_prompt(task, plan, work)}
    ]
    raw_code, _ = call_llm(messages_for_impl)
    code = _extract_code(raw_code)

    if not code or f"def {task.name}" not in code:
        logger.warning(f"[{task.name}] implement_function produced no valid code")
        return None, []

    messages_checkpoint = messages_for_impl + [
        {"role": "assistant", "content": raw_code}
    ]
    phase_trajectories.append({"phase": "Implement", "code": code})
    logger.info(f"[{task.name}] Initial code ({len(code)} chars)")
    return code, messages_checkpoint


# ── Shared: correction loop ───────────────────────────────────────────────────

def _correction_loop(
    task: TaskSpec,
    job_id: str,
    max_iterations: int,
    progress_cb,
    code: str,
    messages_checkpoint: list,
    phase_trajectories: list,
    install_commands: list | None,
    use_local: bool,
    local_sbx,
    work: str | None,
) -> ToolResult:
    """
    Correction loop (paper's make_tool iteration loop).
    For each iteration:
      1. Reset sandbox to post-install checkpoint
      2. Execute function with example inputs
      3. LLM assess success (structured output)
      4. If success → done
      5. Diagnose agent
      6. Rewrite function (LLM)
      7. Summarize problem (LLM)
    """
    problem_summaries: list[str] = []
    final_code = code

    for i in range(1, max_iterations + 1):
        logger.info(f"[{task.name}] Correction iteration {i}/{max_iterations}")
        if progress_cb:
            progress_cb(i, "running")

        # Get/reset sandbox for this iteration
        if use_local:
            sbx = local_sbx
            iter_work = work
            # Reset only implementation.py and runner.py (checkpoint semantics)
            sbx.run_bash(f"rm -f {iter_work}/implementation.py {iter_work}/runner.py", log=False)
        else:
            sbx = AgentSandbox(timeout=600)
            iter_work = sbx.work_dir
            sbx.run_bash(f"git clone --depth 1 {task.repo_url} {iter_work}/repo 2>&1", log=False)
            logger.info(f"[{task.name}] Replaying {len(install_commands)} install commands")
            for cmd in install_commands:
                sbx.run_bash(cmd, log=False)

        try:
            # Reset conversation to post-implement checkpoint (paper: state = state_checkpoint)
            messages = list(messages_checkpoint)
            messages.append({
                "role": "user",
                "content": "I reset the environment to the freshly installed repository, and will now execute the updated function you wrote.",
            })

            # Execute function
            exec_result = _execute_function(task, sbx, iter_work, code)
            logger.info(f"[{task.name}] Execution status: {exec_result['status']}")

            # Assess success (structured LLM call — paper's is_successful_execution)
            messages.append({"role": "user", "content": _assess_prompt(task, exec_result)})
            assessment, _ = call_llm_structured(messages, AssessmentResult)
            messages.append({
                "role": "assistant",
                "content": json.dumps({"successful": assessment.successful, "reasoning": assessment.reasoning})
            })
            logger.info(f"[{task.name}] Assessment: successful={assessment.successful} | {assessment.reasoning[:100]}")

            phase_trajectories.append({
                "phase": f"Correction iteration {i}",
                "execution_status": exec_result["status"],
                "assessment": assessment.successful,
                "stdout": exec_result["stdout"][:500],
            })

            if assessment.successful:
                final_code = code
                logger.info(f"[{task.name}] Agent succeeded at iteration {i} | session=${session_spend():.4f}")
                break

            # Diagnose (agent with tools — paper's diagnose)
            logger.info(f"[{task.name}] Diagnosing iteration {i}")
            diagnose_result = run_agent(
                system=_agent_system_prompt(iter_work, installed=True),
                user=_diagnose_user_prompt(code, exec_result, problem_summaries, assessment),
                sandbox=sbx,
                max_steps=40,
                readonly=False,
            )
            logger.info(f"[{task.name}] Diagnose done | steps={diagnose_result['steps']}")
            messages.extend(diagnose_result["messages"][2:])  # skip system+user (already in messages)

            # Parse diagnosis structured output
            diag_text = diagnose_result["final_message"]
            try:
                diag_parsed, _ = call_llm_structured(
                    messages + [{"role": "user", "content": "Now provide your final structured diagnosis and plan."}],
                    DiagnosisResult,
                )
            except Exception:
                diag_parsed = DiagnosisResult(diagnosis=diag_text[:500], plan="Rewrite the function based on the diagnosis.")
            messages.append({
                "role": "assistant",
                "content": json.dumps({"diagnosis": diag_parsed.diagnosis, "plan": diag_parsed.plan})
            })

            # Rewrite function (single LLM call — paper's rewrite_function)
            logger.info(f"[{task.name}] Rewriting function")
            messages.append({"role": "user", "content": _rewrite_user_prompt(task, code, diag_parsed, iter_work)})
            raw_new_code, _ = call_llm(messages)
            new_code = _extract_code(raw_new_code)
            messages.append({"role": "assistant", "content": raw_new_code})

            if new_code and f"def {task.name}" in new_code:
                code = new_code
                final_code = code

            # Summarize problem (single LLM call — paper's summarize_problem)
            messages.append({"role": "user", "content": _SUMMARIZE_PROMPT})
            summary, _ = call_llm(messages)
            messages.append({"role": "assistant", "content": summary})
            problem_summaries.append(summary)
            logger.info(f"[{task.name}] Problem summary: {summary[:100]}")

        finally:
            if not use_local:
                sbx.kill()
                time.sleep(1)

    else:
        # Exhausted all iterations without success
        logger.warning(f"[{task.name}] Max iterations reached")
        return _evaluate_and_record(
            task, job_id, max_iterations, final_code, phase_trajectories,
            agent_succeeded=False, install_commands=install_commands, use_local=use_local,
        )

    # Post-loop TM-Bench evaluation (paper Section 4)
    return _evaluate_and_record(
        task, job_id, i, final_code, phase_trajectories,
        agent_succeeded=True, install_commands=install_commands, use_local=use_local,
    )


# ── Post-loop: TM-Bench evaluation ────────────────────────────────────────────

def _evaluate_and_record(
    task: TaskSpec,
    job_id: str,
    iterations: int,
    code: str,
    phase_trajectories: list,
    agent_succeeded: bool,
    install_commands: list | None,
    use_local: bool,
) -> ToolResult:
    """
    Run validation_tests once as external TM-Bench evaluation (paper Section 4).
    NOT fed back to agent — for reporting only.
    """
    validation_passed = False
    validation_error = ""

    # Build a tool.py wrapper for validation tests (which import `from tool import run_tool`)
    tool_wrapper = f"from implementation import {task.name}\n\ndef run_tool(**kwargs):\n    return {task.name}(**kwargs)\n"

    if agent_succeeded and task.validation_tests:
        logger.info(f"[{task.name}] TM-Bench evaluation (post-loop)")
        # validation_tests import `from tool import run_tool`, so we pass tool_wrapper as tool_code
        # and implementation as an extra file
        val = run_validation_sandbox(
            install_commands=install_commands or [],
            repo_url=task.repo_url,
            tool_code=tool_wrapper,
            implementation_code=code,
            task_name=task.name,
            validation_tests=task.validation_tests,
            use_local=use_local,
        )
        validation_passed = val.passed
        if not val.passed:
            validation_error = val.error_context[:500]
            logger.warning(f"[{task.name}] TM-Bench FAILED: {validation_error[:100]}")
        else:
            logger.info(f"[{task.name}] TM-Bench PASSED")
    elif agent_succeeded and not task.validation_tests:
        validation_passed = True  # no independent test → agent success is the result

    status = "success" if agent_succeeded else "failed"
    upsert_tool(job_id, task.name, task.repo_url, {},
                status=status, code=code, iterations=iterations)
    traj_path = save_trajectory(
        task.name, task.repo_url, phase_trajectories,
        result_code=code, cost_usd=session_spend(),
    )
    md_path = export_markdown()
    logger.info(f"[{task.name}] Done | status={status} validation={validation_passed} | Trajectory: {traj_path.name}")

    return ToolResult(
        name=task.name,
        status=status,
        code=code,
        iterations=iterations,
        cost_usd=session_spend(),
        validation_passed=validation_passed,
        validation_error=validation_error,
    )


def _record_failure(task, job_id, iterations, error, phase_trajectories) -> ToolResult:
    upsert_tool(job_id, task.name, task.repo_url, {}, status="failed", iterations=iterations, last_error=error)
    save_trajectory(task.name, task.repo_url, phase_trajectories, cost_usd=session_spend())
    export_markdown()
    return ToolResult(name=task.name, status="failed", iterations=iterations, last_error=error)
