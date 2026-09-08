import uuid
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from core.models import TaskSpec, ToolResult
from core.loop import run_closed_loop
from registry.store import upsert_tool, get_tool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["jobs"])

# In-memory job state (iteration count + live status)
_jobs: dict[str, dict] = {}


class JobCreate(BaseModel):
    repo_url: str
    name: str
    description: str
    inputs: dict
    expected_output: str
    max_iterations: int = 5


def _run_job(job_id: str, payload: JobCreate):
    task = TaskSpec(
        name=payload.name,
        repo_url=payload.repo_url,
        description=payload.description,
        inputs=payload.inputs,
        expected_output=payload.expected_output,
    )

    upsert_tool(job_id, task.name, task.repo_url, payload.model_dump(), status="running")

    def on_progress(iteration: int, status: str):
        _jobs[job_id]["iteration"] = iteration
        _jobs[job_id]["status"] = status

    try:
        result: ToolResult = run_closed_loop(
            task,
            max_iterations=payload.max_iterations,
            progress_cb=on_progress,
        )
        _jobs[job_id] = {
            "status": result.status,
            "iteration": result.iterations,
            "last_error": result.last_error,
        }
        upsert_tool(
            job_id, task.name, task.repo_url, payload.model_dump(),
            status=result.status,
            code=result.code,
            test_code=result.test_code,
            iterations=result.iterations,
            last_error=result.last_error,
        )
    except Exception as e:
        logger.exception(f"Job {job_id} crashed")
        _jobs[job_id] = {"status": "error", "iteration": 0, "last_error": str(e)}
        upsert_tool(job_id, task.name, task.repo_url, payload.model_dump(),
                    status="error", last_error=str(e))


@router.post("", status_code=202)
async def create_job(payload: JobCreate, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "iteration": 0, "last_error": ""}
    background_tasks.add_task(_run_job, job_id, payload)
    return {"job_id": job_id}


@router.get("/{job_id}")
async def get_job(job_id: str):
    row = get_tool(job_id)
    live = _jobs.get(job_id)
    if not row and not live:
        raise HTTPException(status_code=404, detail="Job not found")
    state = row or {}
    if live:
        state.update(live)
    return state
