"""
Unit tests for individual components.
No API keys required — all tests run offline.
"""
import pytest
from core.models import TaskSpec, ExecutionResult
from core.codegen import _extract_block


def test_extract_block_wrapper():
    raw = """
```python
# install_cmd
pip install biopython
```

```python
# wrapper_code
def run_tool(**kwargs):
    return {"gc": 50.0}
```

```python
# test_code
def test_basic():
    assert True
```
"""
    assert _extract_block(raw, "install_cmd") == "pip install biopython"
    assert "def run_tool" in _extract_block(raw, "wrapper_code")
    assert "def test_basic" in _extract_block(raw, "test_code")


def test_execution_result_passed():
    r = ExecutionResult(stdout="ok", stderr="", exit_code=0)
    assert r.passed is True


def test_execution_result_failed():
    r = ExecutionResult(stdout="", stderr="ImportError", exit_code=1)
    assert r.passed is False
    assert "ImportError" in r.error_context


def test_task_spec_fields():
    t = TaskSpec(
        name="test",
        repo_url="https://github.com/example/repo",
        description="test task",
        inputs={"seq": "ATCG"},
        expected_output="a float",
    )
    assert t.name == "test"
    assert t.inputs["seq"] == "ATCG"


def test_task_spec_validation_tests_defaults_empty():
    """validation_tests should default to "" so old task specs still work."""
    t = TaskSpec(
        name="x", repo_url="http://github.com/x/x",
        description="d", inputs={}, expected_output="o",
    )
    assert t.validation_tests == ""


def test_task_spec_requires_gpu_defaults_false():
    """requires_gpu should default to False — no GPU assumption for CPU tasks."""
    t = TaskSpec(
        name="x", repo_url="http://github.com/x/x",
        description="d", inputs={}, expected_output="o",
    )
    assert t.requires_gpu is False


def test_task_spec_gpu_flag_set():
    t = TaskSpec(
        name="esmfold",
        repo_url="https://github.com/facebookresearch/esm",
        description="protein folding",
        inputs={"sequence": "MKTAY"},
        expected_output="PDB string",
        requires_gpu=True,
        validation_tests="def test_foo(): pass",
    )
    assert t.requires_gpu is True
    assert "def test_foo" in t.validation_tests


def test_execution_result_error_context_includes_both():
    r = ExecutionResult(stdout="OUT", stderr="ERR", exit_code=1)
    ctx = r.error_context
    assert "OUT" in ctx
    assert "ERR" in ctx


# ── LocalSandbox unit tests (no GPU or API keys needed) ──────────────────────

def test_local_sandbox_creates_work_dir():
    from core.sandbox import LocalSandbox
    sbx = LocalSandbox()
    import os
    assert os.path.isdir(sbx.work_dir)
    sbx.kill()
    assert not os.path.exists(sbx.work_dir)


def test_local_sandbox_run_bash():
    from core.sandbox import LocalSandbox
    sbx = LocalSandbox()
    try:
        out = sbx.run_bash("echo hello_toolmaker", log=False)
        assert "hello_toolmaker" in out
    finally:
        sbx.kill()


def test_local_sandbox_write_read_file():
    from core.sandbox import LocalSandbox
    sbx = LocalSandbox()
    try:
        sbx.write_file(f"{sbx.work_dir}/test.txt", "content123")
        content = sbx.read_file(f"{sbx.work_dir}/test.txt")
        assert "content123" in content
    finally:
        sbx.kill()


def test_local_sandbox_path_normalization():
    from core.sandbox import LocalSandbox
    sbx = LocalSandbox()
    try:
        # /home/user/x.txt should map to work_dir/x.txt
        sbx.write_file("/home/user/norm_test.txt", "normalized")
        content = sbx.read_file(f"{sbx.work_dir}/norm_test.txt")
        assert "normalized" in content
    finally:
        sbx.kill()


def test_local_sandbox_timeout_kills_process():
    from core.sandbox import LocalSandbox
    sbx = LocalSandbox()
    try:
        out = sbx.run_bash("sleep 60", log=False, timeout=2)
        assert "Killed" in out or "Exit code" in out
    finally:
        sbx.kill()


def test_local_sandbox_reset_generated_files():
    from core.sandbox import LocalSandbox
    import os
    sbx = LocalSandbox()
    try:
        sbx.write_file(f"{sbx.work_dir}/tool.py", "def run_tool(**kw): pass")
        sbx.write_file(f"{sbx.work_dir}/test_tool.py", "def test_x(): pass")
        sbx.reset_generated_files()
        assert not os.path.exists(f"{sbx.work_dir}/tool.py")
        assert not os.path.exists(f"{sbx.work_dir}/test_tool.py")
    finally:
        sbx.kill()


def test_local_gpu_available_returns_bool():
    from core.sandbox import local_gpu_available
    result = local_gpu_available()
    assert isinstance(result, bool)
