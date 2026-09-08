from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaskSpec:
    name: str
    repo_url: str
    description: str
    inputs: dict
    expected_output: str
    # Pre-written by a human; agent never sees these.
    # Run independently after agent's own tests pass.
    # Only if these pass is the tool marked successful.
    validation_tests: str = ""
    # If True, GPU is checked before the create loop starts.
    # Returns status="gpu_required" immediately if no CUDA found.
    requires_gpu: bool = False


@dataclass
class GeneratedCode:
    install_cmd: str
    wrapper_code: str
    test_code: str


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int

    @property
    def passed(self) -> bool:
        return self.exit_code == 0

    @property
    def error_context(self) -> str:
        return f"STDOUT:\n{self.stdout}\n\nSTDERR:\n{self.stderr}"


@dataclass
class ToolResult:
    name: str = ""
    status: str = "pending"          # "success" | "failed" | "gpu_required"
    code: str = ""
    test_code: str = ""
    iterations: int = 0
    last_error: str = ""
    cost_usd: float = 0.0
    validation_passed: bool = False   # TM-Bench evaluation, run once after loop
    validation_error: str = ""
