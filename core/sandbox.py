"""
Two sandbox backends:

AgentSandbox  — E2B cloud sandbox (CPU).  Used for all non-GPU tasks.
LocalSandbox  — Isolated venv on this server (GPU).  Used when task.requires_gpu=True
               and a local CUDA device is detected.

Safety measures in LocalSandbox:
  - CUDA_VISIBLE_DEVICES restricted to one GPU (GPU_INDEX, default 0)
  - Fresh venv with --system-site-packages so existing CUDA/PyTorch is inherited
    but new agent-installed packages stay isolated
  - subprocess runs in its own process group; SIGKILL sent to the entire group on timeout
  - PYTORCH_CUDA_ALLOC_CONF set to reduce memory fragmentation
  - work_dir and venv are deleted on kill()
"""

import os
import re
import shutil
import signal
import subprocess
import sys
import uuid

from e2b_code_interpreter import Sandbox
from e2b.sandbox.commands.command_handle import CommandExitException

from core.models import ExecutionResult


# ── GPU configuration ─────────────────────────────────────────────────────────

GPU_INDEX = 0  # which GPU local sandboxes may use


def local_gpu_available() -> bool:
    """Check if a CUDA GPU is accessible on the host server."""
    try:
        r = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        return r.returncode == 0 and ("NVIDIA" in r.stdout or "Driver Version" in r.stdout)
    except Exception:
        return False


# ── E2B cloud sandbox ─────────────────────────────────────────────────────────

class AgentSandbox:
    """
    Persistent E2B sandbox exposing individual tool actions for the agent.
    command_log records every state-modifying command so the install phase
    can be replayed in fresh create-phase sandboxes (paper's checkpoint reset).
    """

    work_dir: str = "/home/user"

    def __init__(self, timeout: int = 600):
        self.sbx = Sandbox.create(timeout=timeout)
        self.command_log: list[str] = []

    def _run(self, cmd: str, timeout: int = 120) -> tuple[str, int]:
        try:
            r = self.sbx.commands.run(cmd, timeout=timeout)
            return (r.stdout or "") + (r.stderr or ""), 0
        except CommandExitException as e:
            return (e.stdout or "") + (e.stderr or ""), e.exit_code

    def run_bash(self, command: str, log: bool = True, timeout: int = 120) -> str:
        if log:
            self.command_log.append(command)
        output, exit_code = self._run(command, timeout=timeout)
        if exit_code != 0:
            output += f"\n[Exit code: {exit_code}]"
        return output[:3000]

    def read_file(self, path: str) -> str:
        try:
            return self.sbx.files.read(path)[:3000]
        except Exception as e:
            return f"Error reading {path}: {e}"

    def write_file(self, path: str, content: str) -> str:
        try:
            self.sbx.files.write(path, content)
            return f"Written: {path}"
        except Exception as e:
            return f"Error writing {path}: {e}"

    def list_directory(self, path: str = "/home/user") -> str:
        out, _ = self._run(f"find {path} -maxdepth 3 ! -path '*/.*' | head -60")
        return out[:2000]

    def browse(self, url: str) -> str:
        """Fetch a URL and return readable text (strips HTML tags)."""
        strip_script = (
            "import sys,html,re; "
            "t=sys.stdin.read(); "
            "t=re.sub(r'<script[^>]*>.*?</script>','',t,flags=re.DOTALL|re.I); "
            "t=re.sub(r'<style[^>]*>.*?</style>','',t,flags=re.DOTALL|re.I); "
            "t=re.sub(r'<[^>]+>',' ',t); "
            "t=html.unescape(t); "
            "t=re.sub(r'\\s+',' ',t).strip(); "
            "print(t[:4000])"
        )
        out, _ = self._run(
            f"curl -sL --max-time 15 '{url}' | python3 -c \"{strip_script}\"",
            timeout=20,
        )
        return out[:3000]

    def kill(self):
        self.sbx.kill()


# ── Local GPU sandbox ─────────────────────────────────────────────────────────

class LocalSandbox:
    """
    Runs agent commands in an isolated venv on the local server.
    Used exclusively for GPU tasks (task.requires_gpu=True).

    The venv is created with --system-site-packages so that the server's
    existing CUDA/PyTorch installation is available without re-downloading.
    Packages the agent installs are confined to the local venv.

    Between create-phase iterations the loop calls reset_generated_files()
    to remove tool.py / test_tool.py while preserving the venv and cloned repo.
    """

    def __init__(self, gpu_index: int = GPU_INDEX, timeout: int = 600):
        uid = uuid.uuid4().hex[:8]
        self.work_dir = f"/tmp/toolmaker_{uid}"
        os.makedirs(self.work_dir, exist_ok=True)
        self._venv = os.path.join(self.work_dir, ".venv")
        self.gpu_index = gpu_index
        self.timeout = timeout
        self.command_log: list[str] = []

        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", self._venv],
            check=True,
            timeout=60,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    @property
    def _env(self) -> dict:
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(self.gpu_index),
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,garbage_collection_threshold:0.6",
            "PATH": f"{self._venv}/bin:{env.get('PATH', '')}",
            "VIRTUAL_ENV": self._venv,
            "HOME": self.work_dir,
            "PYTHONPATH": "",   # don't leak host PYTHONPATH into venv
        })
        return env

    def _run(self, cmd: str, timeout: int = 120) -> tuple[str, int]:
        """Run command in subprocess; kill entire process group on timeout."""
        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=self.work_dir,
                env=self._env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid,   # new process group → clean group kill
            )
            try:
                stdout, _ = proc.communicate(timeout=timeout)
                return stdout or "", proc.returncode
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                return f"[Killed after {timeout}s timeout]", 124
        except Exception as e:
            return f"[LocalSandbox._run error: {e}]", 1

    def _normalize(self, path: str) -> str:
        """Translate /home/user/... → work_dir/... for cross-sandbox compatibility."""
        if path.startswith("/home/user"):
            return path.replace("/home/user", self.work_dir, 1)
        return path

    # ── public interface (mirrors AgentSandbox exactly) ──────────────────────

    def run_bash(self, command: str, log: bool = True, timeout: int = 120) -> str:
        # Transparently redirect /home/user/ → work_dir in shell commands
        command = command.replace("/home/user", self.work_dir)
        if log:
            self.command_log.append(command)
        output, exit_code = self._run(command, timeout=timeout)
        if exit_code != 0:
            output += f"\n[Exit code: {exit_code}]"
        return output[:3000]

    def read_file(self, path: str) -> str:
        path = self._normalize(path)
        try:
            with open(path) as f:
                return f.read()[:3000]
        except Exception as e:
            return f"Error reading {path}: {e}"

    def write_file(self, path: str, content: str) -> str:
        path = self._normalize(path)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return f"Written: {path}"
        except Exception as e:
            return f"Error writing {path}: {e}"

    def list_directory(self, path: str = "/home/user") -> str:
        path = self._normalize(path)
        out, _ = self._run(f"find {path} -maxdepth 3 ! -path '*/.*' | head -60")
        return out[:2000]

    def browse(self, url: str) -> str:
        """Fetch URL via curl; strip HTML in the venv's Python."""
        strip_script = (
            "import sys,html,re; "
            "t=sys.stdin.read(); "
            "t=re.sub(r'<script[^>]*>.*?</script>','',t,flags=re.DOTALL|re.I); "
            "t=re.sub(r'<style[^>]*>.*?</style>','',t,flags=re.DOTALL|re.I); "
            "t=re.sub(r'<[^>]+>',' ',t); "
            "t=html.unescape(t); "
            "t=re.sub(r'\\s+',' ',t).strip(); "
            "print(t[:4000])"
        )
        out, _ = self._run(
            f"curl -sL --max-time 15 '{url}' | python3 -c \"{strip_script}\"",
            timeout=20,
        )
        return out[:3000]

    def reset_generated_files(self):
        """
        Remove tool.py and test_tool.py between create-loop iterations.
        Preserves the venv and cloned repo (checkpoint semantics from the paper).
        """
        for fname in ("tool.py", "test_tool.py"):
            path = os.path.join(self.work_dir, fname)
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def kill(self):
        """Delete the work directory (venv + all generated files)."""
        shutil.rmtree(self.work_dir, ignore_errors=True)


# ── Independent validation sandbox ───────────────────────────────────────────

def run_validation_sandbox(
    install_commands: list[str],
    repo_url: str,
    tool_code: str,
    validation_tests: str,
    task_name: str = "",
    implementation_code: str = "",
    timeout: int = 300,
    use_local: bool = False,
    gpu_index: int = GPU_INDEX,
) -> ExecutionResult:
    """
    TM-Bench evaluation: run human-written validation_tests in a fresh sandbox.
    NOT fed back to agent — for reporting only (paper Section 4).

    tool_code        = tool.py wrapper (from tool import run_tool)
    implementation_code = implementation.py (the actual function, if separate)
    """
    sbx: AgentSandbox | LocalSandbox = (
        LocalSandbox(gpu_index=gpu_index, timeout=timeout)
        if use_local
        else AgentSandbox(timeout=timeout)
    )
    try:
        work = sbx.work_dir
        sbx.run_bash(f"git clone --depth 1 {repo_url} {work}/repo 2>&1", log=False)
        for cmd in install_commands:
            sbx.run_bash(cmd, log=False)
        sbx.write_file(f"{work}/tool.py", tool_code)
        if implementation_code:
            sbx.write_file(f"{work}/implementation.py", implementation_code)
        sbx.write_file(f"{work}/test_validation.py", validation_tests)
        output = sbx.run_bash(
            f"cd {work} && python -m pytest test_validation.py -v --tb=short 2>&1",
            log=False,
            timeout=180,
        )
        has_pass = bool(re.search(r"\d+ passed", output))
        has_fail = bool(re.search(r"\d+ failed|\d+ error", output))
        passed = has_pass and not has_fail
        return ExecutionResult(stdout=output, stderr="", exit_code=0 if passed else 1)
    finally:
        sbx.kill()
