# ToolMaker — Autonomous Scientific Tool Discovery Agent

A faithful implementation of the [ToolMaker paper](https://arxiv.org/abs/2502.11705) (Wölflein et al., ACL 2025): *"LLM Agents Making Agent Tools"*.

Given a GitHub repository and a task description, an LLM agent autonomously wraps it into a callable Python tool — no human programmer required.

---

## How It Works

The system implements **Algorithm 1** from the paper: a two-phase closed loop.

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  PHASE 1 — INSTALL                                      │
│  Agent explores repo → reads setup files → pip install  │
│                                                         │
│  PHASE 2 — CREATE (closed loop)                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Fresh sandbox (reset) each iteration            │    │
│  │ Agent writes tool.py → runs pytest              │    │
│  │ If tests fail → reads error → rewrites → retry  │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  On success → save to registry + trajectory file        │
└─────────────────────────────────────────────────────────┘
```

The agent has five tools it can call at any point:

| Tool | Purpose |
|---|---|
| `run_bash` | Run shell commands (pip install, pytest, etc.) |
| `read_file` | Read source files in the sandbox |
| `write_file` | Write `tool.py` and `test_tool.py` |
| `list_directory` | Explore the repo structure |
| `browse` | Fetch documentation URLs |

Every generated tool is:
- **Tested** — must pass pytest before being accepted
- **Stored** — saved to SQLite registry
- **Documented** — trajectory saved as readable markdown

---

## Project Structure

```
tp/
├── core/
│   ├── agent.py        # Agentic loop with OpenAI tool calling + cost tracking
│   ├── sandbox.py      # E2B cloud sandbox (AgentSandbox class)
│   ├── loop.py         # Orchestrates Phase 1 + Phase 2
│   ├── ingestion.py    # Git clone + repo context extraction
│   ├── trajectory.py   # Saves agent conversation as markdown
│   └── models.py       # Shared dataclasses
├── api/
│   ├── main.py         # FastAPI app
│   └── routes/
│       ├── jobs.py     # POST /jobs, GET /jobs/{id}
│       └── tools.py    # GET /tools, GET /tools/{id}
├── registry/
│   └── store.py        # SQLite storage + tools_registry.md export
├── tasks/              # Example task specs (.yaml)
├── trajectories/       # Per-run agent conversation logs (auto-generated)
├── tests/
│   ├── test_loop.py    # Unit tests
│   └── test_e2e.py     # End-to-end tests (requires API keys)
├── tools_registry.md   # Human-readable summary of all runs (auto-generated)
└── requirements.txt
```

---

## Setup

### 1. Clone and create environment

```bash
git clone <repo-url>
cd tp
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Key | Where to get it | Cost |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Free |
| `E2B_API_KEY` | [e2b.dev](https://e2b.dev) | Free tier |
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) | ~$0.02/task (gpt-4o-mini) |

### 3. Run unit tests

```bash
.venv/bin/python3 -m pytest tests/test_loop.py -v
```

---

## Usage

### Option A — Direct Python (quickest)

```python
from dotenv import load_dotenv
load_dotenv()

from core.models import TaskSpec
from core.loop import run_closed_loop

task = TaskSpec(
    name="gc_content",
    repo_url="https://github.com/biopython/biopython",
    description="Calculate GC content of a DNA sequence. Return a float between 0 and 100.",
    inputs={"sequence": "ATGCGATCG"},
    expected_output="float between 0 and 100",
)

result = run_closed_loop(task, max_iterations=5)
print(result.status)    # "success"
print(result.code)      # generated wrapper code
print(f"${result.cost_usd:.4f}")  # cost in USD
```

### Option B — REST API

```bash
# Start the server
.venv/bin/uvicorn api.main:app --reload

# Submit a job
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/biopython/biopython",
    "name": "gc_content",
    "description": "Calculate GC content of a DNA sequence",
    "inputs": {"sequence": "ATGCGATCG"},
    "expected_output": "float between 0 and 100"
  }'

# Poll for result
curl http://localhost:8000/jobs/<job_id>

# List all successful tools
curl http://localhost:8000/tools
```

### Option C — From a task YAML

```bash
.venv/bin/python3 -c "
import yaml
from dotenv import load_dotenv; load_dotenv()
from core.models import TaskSpec
from core.loop import run_closed_loop

spec = yaml.safe_load(open('tasks/example.yaml'))
result = run_closed_loop(TaskSpec(**spec))
print(result.status)
"
```

---

## Outputs

After each run you get:

**`tools_registry.md`** — summary of every tool ever created, with code and tests. Open in VSCode.

**`trajectories/<name>_<timestamp>.md`** — full agent conversation log for that run: every tool call, every result, every self-correction. Useful for debugging and understanding what the agent did.

**SQLite registry** (`registry/tools.db`) — queryable database of all runs.

---

## End-to-End Tests

> **Requires API keys and internet access. Uses ~$0.05 of OpenAI credits per run.**

```bash
.venv/bin/python3 -m pytest tests/test_e2e.py -v -s
```

Tests two real scientific repos (BioPython, NumPy) through the full pipeline.

---

## Cost

All LLM calls use `gpt-4o-mini`. Typical costs:

| Task complexity | Approx. cost |
|---|---|
| Simple (biopython, numpy) | $0.01 – $0.03 |
| Medium (multi-step scientific) | $0.03 – $0.08 |
| Hard / many iterations | $0.10 – $0.20 |

A hard stop at **$1.80** is enforced in `core/agent.py` (`BUDGET_LIMIT_USD`).

---

## Paper Reference

> Wölflein, G., Ferber, D., Truhn, D., Arandjelović, O., & Kather, J. N. (2025).
> **LLM Agents Making Agent Tools.**
> *Proceedings of ACL 2025.*
> [arxiv.org/abs/2502.11705](https://arxiv.org/abs/2502.11705)
