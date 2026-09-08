"""
TM-Bench runner — reproduces the ToolMaker paper evaluation.

Usage:
    # Run a single task
    python run_tmbench.py tabpfn_predict

    # Run all runnable tasks (no special data needed)
    python run_tmbench.py --runnable

    # Run all 15 tasks (blocked ones will be skipped automatically)
    python run_tmbench.py --all

    # List all tasks and their status
    python run_tmbench.py --list
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from core.models import TaskSpec
from core.loop import run_closed_loop
from core.agent import session_spend, BUDGET_LIMIT_USD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TMBENCH_DIR = Path(__file__).parent / "tasks" / "tmbench"

# Tasks blocked by missing data — will be skipped automatically
BLOCKED = {
    "stamp_extract_features":        "needs TCGA .svs whole slide image files",
    "stamp_train_classification_model": "needs TCGA .svs files + clinical tables",
    "flowmap_overfit_scene":          "needs LLFF scene image datasets",
    "pathfinder_verify_biomarker":    "needs TCGA CRC heatmap .npy + clinical data",
    "nnunet_train_model":             "needs MSD medical segmentation dataset",
}

# Tasks blocked by gated HuggingFace models
NEEDS_HF_TOKEN = {
    "conch_extract_features": "needs approved HF access to MahmoodLab/CONCH",
    "uni_extract_features":   "needs approved HF access to MahmoodLab/UNI",
    "musk_extract_features":  "needs approved HF access to xiangjx/musk",
    "medsss_generate":        "needs HF token for MedSSS model download",
}

# Runnable right now
RUNNABLE = [
    "modernbert_predict_masked",
    "tabpfn_predict",
    "cytopus_db",
    "esm_fold_predict",
    "retfound_feature_vector",
    "medsam_inference",
]


def load_task(name: str) -> TaskSpec:
    path = TMBENCH_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Task spec not found: {path}")
    with open(path) as f:
        d = yaml.safe_load(f)
    return TaskSpec(
        name=d["name"],
        repo_url=d["repo_url"],
        description=d["description"],
        inputs=d.get("inputs", {}),
        expected_output=d.get("expected_output", ""),
        validation_tests=d.get("validation_tests", ""),
        requires_gpu=d.get("requires_gpu", False),
    )


def run_task(name: str) -> dict:
    if name in BLOCKED:
        reason = BLOCKED[name]
        logger.warning(f"SKIP {name}: {reason}")
        return {"name": name, "status": "blocked", "reason": reason}

    import os
    hf_token = os.environ.get("HF_TOKEN", "")
    if name in NEEDS_HF_TOKEN and not hf_token:
        reason = NEEDS_HF_TOKEN[name]
        logger.warning(f"SKIP {name}: {reason} (set HF_TOKEN env var to enable)")
        return {"name": name, "status": "needs_hf_token", "reason": reason}

    logger.info(f"\n{'='*60}")
    logger.info(f"TASK: {name}")
    logger.info(f"Budget used: ${session_spend():.4f} / ${BUDGET_LIMIT_USD:.2f}")
    logger.info(f"{'='*60}")

    task = load_task(name)
    t0 = time.time()
    result = run_closed_loop(task, max_iterations=10)
    elapsed = time.time() - t0

    return {
        "name": name,
        "status": result.status,
        "iterations": result.iterations,
        "cost_usd": result.cost_usd,
        "elapsed_s": round(elapsed),
        "validation_passed": result.validation_passed,
        "error": result.last_error[:200] if result.last_error else "",
    }


def print_results(results: list[dict]):
    print("\n" + "=" * 75)
    print(f"{'TASK':<40} {'AGENT':>7} {'TM-BENCH':>9} {'ITER':>4} {'COST':>8}")
    print("-" * 75)
    agent_pass = agent_fail = tmbench_pass = blocked = 0
    for r in results:
        status = r["status"]
        iters = r.get("iterations", "-")
        cost = f"${r.get('cost_usd', 0):.3f}" if r.get("cost_usd") else "-"
        agent_ok = "PASS" if status == "success" else ("-" if status in ("blocked", "needs_hf_token") else "FAIL")
        val_ok = "PASS" if r.get("validation_passed") else ("-" if status in ("blocked", "needs_hf_token") else "FAIL")
        print(f"{r['name']:<40} {agent_ok:>7} {val_ok:>9} {str(iters):>4} {cost:>8}")
        if status == "success":
            agent_pass += 1
            if r.get("validation_passed"):
                tmbench_pass += 1
        elif status in ("blocked", "needs_hf_token"):
            blocked += 1
        else:
            agent_fail += 1
    print("=" * 75)
    total_runnable = agent_pass + agent_fail
    agent_pct = f"{100*agent_pass/total_runnable:.0f}%" if total_runnable else "N/A"
    tm_pct = f"{100*tmbench_pass/total_runnable:.0f}%" if total_runnable else "N/A"
    print(f"AGENT SUCCESS (own tests):  {agent_pass}/{total_runnable} ({agent_pct})")
    print(f"TM-BENCH PASS (validation): {tmbench_pass}/{total_runnable} ({tm_pct})  ← paper metric")
    print(f"BLOCKED: {blocked} tasks (missing data or HF token)")
    print(f"Session spend: ${session_spend():.4f}")


def list_tasks():
    all_tasks = sorted(p.stem for p in TMBENCH_DIR.glob("*.yaml"))
    print(f"\n{'TASK':<42} {'STATUS'}")
    print("-" * 60)
    for t in all_tasks:
        if t in BLOCKED:
            status = f"BLOCKED  ({BLOCKED[t][:30]}...)"
        elif t in NEEDS_HF_TOKEN:
            status = f"HF_TOKEN ({NEEDS_HF_TOKEN[t][:30]}...)"
        elif t in RUNNABLE:
            status = "RUNNABLE"
        else:
            status = "UNKNOWN"
        print(f"{t:<42} {status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", help="Task name to run")
    parser.add_argument("--runnable", action="store_true", help="Run all 6 runnable tasks")
    parser.add_argument("--all", action="store_true", help="Run all 15 tasks (blocked ones skipped)")
    parser.add_argument("--list", action="store_true", help="List all tasks and status")
    args = parser.parse_args()

    if args.list:
        list_tasks()
        return

    if args.all:
        tasks_to_run = sorted(p.stem for p in TMBENCH_DIR.glob("*.yaml"))
    elif args.runnable:
        tasks_to_run = RUNNABLE
    elif args.task:
        tasks_to_run = [args.task]
    else:
        parser.print_help()
        sys.exit(1)

    results = []
    for name in tasks_to_run:
        try:
            r = run_task(name)
            results.append(r)
            if r["status"] not in ("blocked", "needs_hf_token"):
                logger.info(f"  → {r['status']} in {r.get('elapsed_s', '?')}s")
        except Exception as e:
            logger.error(f"CRASH on {name}: {e}")
            results.append({"name": name, "status": "crash", "error": str(e)})

    print_results(results)


if __name__ == "__main__":
    main()
