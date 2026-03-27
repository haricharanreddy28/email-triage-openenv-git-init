"""
FastAPI Server — Email Triage OpenEnv
======================================
Exposes all required OpenEnv endpoints:
  POST /reset
  POST /step
  GET  /state
  GET  /tasks
  POST /grader
  POST /baseline
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import subprocess
import sys

from environment import EmailTriageEnvironment, Action, Observation

app = FastAPI(
    title="Email Triage OpenEnv",
    description="An RL environment for training agents to triage and prioritize emails.",
    version="1.0.0",
)

# Allow cross-origin requests (needed for HF Spaces)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One shared environment instance
env = EmailTriageEnvironment()


# ── Request / Response models ──────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy"   # "easy" | "medium" | "hard"

class StepRequest(BaseModel):
    priority: str   # "urgent" | "normal" | "low"
    category: str   # "billing" | "support" | "spam" | "inquiry"
    route_to: str   # "billing_team" | "support_team" | "trash" | "sales_team"

class GraderRequest(BaseModel):
    task_id: str
    actions: list[dict]   # list of {priority, category, route_to}

class BaselineRequest(BaseModel):
    task_ids: list[str] = ["easy", "medium", "hard"]


# ── Endpoints ──────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — returns 200. Required by OpenEnv pre-submission checklist."""
    return {
        "status": "ok",
        "environment": "Email Triage OpenEnv",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"],
    }

@app.post("/reset")
def reset(request: ResetRequest):
    """
    Start a new episode.
    Returns the first email observation.
    """
    try:
        obs = env.reset(task_id=request.task_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step(request: StepRequest):
    """
    Take one action (classify one email).
    Returns: observation, reward, done, info
    """
    try:
        action = Action(
            priority=request.priority,
            category=request.category,
            route_to=request.route_to,
        )
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state():
    """Return current environment state."""
    return env.state()

@app.get("/tasks")
def tasks():
    """
    Return all tasks with action schema.
    Required by pre-submission checklist.
    """
    return env.get_tasks()

@app.post("/grader")
def grader(request: GraderRequest):
    """
    Score a completed episode given a list of actions.
    Required by pre-submission checklist.
    Returns: score between 0.0 and 1.0
    """
    try:
        score = env.grade_episode(request.task_id, request.actions)
        return {
            "task_id": request.task_id,
            "score": score,
            "num_actions": len(request.actions),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/baseline")
def baseline(request: BaselineRequest):
    """
    Run baseline inference script against all requested tasks.
    Required by pre-submission checklist.
    Returns baseline scores for all tasks.
    """
    import subprocess, sys
    results = {}
    for task_id in request.task_ids:
        try:
            result = subprocess.run(
                [sys.executable, "baseline.py", "--task", task_id],
                capture_output=True, text=True, timeout=120
            )
            # Parse score from output
            for line in result.stdout.splitlines():
                if "Final score" in line:
                    score = float(line.split(":")[-1].strip())
                    results[task_id] = score
                    break
            else:
                results[task_id] = None
        except Exception as e:
            results[task_id] = f"error: {str(e)}"

    return {"baseline_scores": results}
