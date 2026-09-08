"""
End-to-end system test.
Requires OPENAI_API_KEY and E2B_API_KEY in .env
Run with: .venv/bin/python3 -m pytest tests/test_e2e.py -v -s
"""
import ast
import pytest
from dotenv import load_dotenv

load_dotenv()

from core.models import TaskSpec, ToolResult
from core.loop import run_closed_loop
from registry.store import init_db, list_tools


# ── Session-scoped fixtures: loop runs once per task ────────────────────────

@pytest.fixture(scope="session", autouse=True)
def db():
    init_db()


@pytest.fixture(scope="session")
def gc_result() -> ToolResult:
    task = TaskSpec(
        name="gc_content",
        repo_url="https://github.com/biopython/biopython",
        description="Calculate GC content of a DNA sequence. Return a float between 0 and 100.",
        inputs={"sequence": "ATGCGATCG"},
        expected_output="float between 0 and 100 representing GC percentage",
        validation_tests="""\
from tool import run_tool

def test_known_gc_value():
    result = run_tool(sequence="GCGC")
    assert isinstance(result, dict), "run_tool must return a dict"
    key = [k for k in result if "gc" in k.lower()]
    assert key, f"Expected a 'gc' key in result, got: {result}"
    assert abs(result[key[0]] - 100.0) < 0.1, f"Expected 100.0, got {result[key[0]]}"

def test_known_at_only():
    result = run_tool(sequence="ATAT")
    key = [k for k in result if "gc" in k.lower()]
    assert key
    assert abs(result[key[0]] - 0.0) < 0.1, f"Expected 0.0, got {result[key[0]]}"

def test_mixed_sequence():
    result = run_tool(sequence="ATGCGATCG")
    key = [k for k in result if "gc" in k.lower()]
    assert key
    assert 50 <= result[key[0]] <= 60, f"Expected ~55.6, got {result[key[0]]}"
""",
    )
    return run_closed_loop(task, max_iterations=5)


@pytest.fixture(scope="session")
def numpy_result() -> ToolResult:
    task = TaskSpec(
        name="numpy_stats",
        repo_url="https://github.com/numpy/numpy",
        description=(
            "Given a list of numbers, compute descriptive statistics: "
            "mean, std, min, max, and median. Return a dict with all five keys."
        ),
        inputs={"values": [4, 7, 13, 2, 1, 8, 9, 3]},
        expected_output="dict with keys 'mean', 'std', 'min', 'max', 'median' (all floats)",
        validation_tests="""\
from tool import run_tool

VALS = [4.0, 7.0, 13.0, 2.0, 1.0, 8.0, 9.0, 3.0]

def test_returns_all_stats():
    result = run_tool(values=VALS)
    assert isinstance(result, dict), "run_tool must return a dict"
    for key in ("mean", "std", "min", "max", "median"):
        assert key in result, f"Missing key: {key}"

def test_mean_correct():
    result = run_tool(values=VALS)
    assert abs(result["mean"] - 5.875) < 0.01, f"mean={result['mean']}, expected 5.875"

def test_min_max_correct():
    result = run_tool(values=VALS)
    assert result["min"] == 1.0
    assert result["max"] == 13.0

def test_median_correct():
    result = run_tool(values=VALS)
    assert abs(result["median"] - 5.5) < 0.01, f"median={result['median']}, expected 5.5"
""",
    )
    return run_closed_loop(task, max_iterations=5)


# ── Helpers ──────────────────────────────────────────────────────────────────

def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ── Task 1: BioPython GC content ─────────────────────────────────────────────

class TestGCContent:

    def test_loop_succeeds(self, gc_result):
        assert gc_result.status == "success", (
            f"Loop failed after {gc_result.iterations} iterations.\n"
            f"Error: {gc_result.last_error[:600]}"
        )

    def test_took_at_most_5_iterations(self, gc_result):
        assert gc_result.iterations <= 5

    def test_generated_code_is_valid_python(self, gc_result):
        assert is_valid_python(gc_result.code), f"Syntax error in:\n{gc_result.code}"

    def test_generated_code_has_run_tool(self, gc_result):
        assert "def run_tool" in gc_result.code

    def test_code_uses_biopython_not_dummy(self, gc_result):
        # Verify the tool actually uses BioPython, not a hardcoded or dummy calculation.
        # A real implementation must reference Bio, biopython, GC, or SeqUtils.
        code_lower = gc_result.code.lower()
        assert any(kw in code_lower for kw in ("bio", "sequtils", "gc_fraction", "gc_content")), (
            "Tool does not appear to use BioPython — may be a dummy implementation.\n"
            f"Code:\n{gc_result.code[:800]}"
        )

    def test_saved_to_registry(self, gc_result):
        assert gc_result.status == "success"
        names = [t["name"] for t in list_tools("success")]
        assert "gc_content" in names

    def test_registry_fields_complete(self, gc_result):
        saved = [t for t in list_tools("success") if t["name"] == "gc_content"]
        assert saved
        t = saved[-1]
        assert t["repo_url"] == "https://github.com/biopython/biopython"
        assert "run_tool" in t["code"]
        assert "def test_" in t["test_code"]
        assert t["iterations"] >= 1


# ── Task 2: NumPy descriptive stats ─────────────────────────────────────────

class TestNumpyStats:

    def test_loop_succeeds(self, numpy_result):
        assert numpy_result.status == "success", (
            f"Loop failed after {numpy_result.iterations} iterations.\n"
            f"Error: {numpy_result.last_error[:600]}"
        )

    def test_generated_code_is_valid_python(self, numpy_result):
        assert is_valid_python(numpy_result.code)

    def test_generated_code_has_run_tool(self, numpy_result):
        assert "def run_tool" in numpy_result.code

    def test_code_uses_numpy(self, numpy_result):
        # Must import numpy — not a pure-Python dummy
        code_lower = numpy_result.code.lower()
        assert "numpy" in code_lower or "import np" in code_lower, (
            "Tool does not appear to use NumPy.\n"
            f"Code:\n{numpy_result.code[:800]}"
        )

    def test_returns_all_five_stats(self, numpy_result):
        # The task now asks for 5 stats; validation_tests enforce this
        assert numpy_result.status == "success"
        # Spot-check: code contains references to median/min/max
        code_lower = numpy_result.code.lower()
        assert "median" in code_lower, "Tool missing median calculation"

    def test_saved_to_registry(self, numpy_result):
        assert numpy_result.status == "success"
        names = [t["name"] for t in list_tools("success")]
        assert "numpy_stats" in names
