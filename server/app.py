"""
FastAPI Server - Email Triage OpenEnv
Exposes all required OpenEnv endpoints.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import subprocess, sys

from environment import EmailTriageEnvironment, Action, Observation

app = FastAPI(
    title="Email Triage OpenEnv",
    description="An RL environment for training agents to triage and prioritize emails.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = EmailTriageEnvironment()


class StepRequest(BaseModel):
    priority: str
    category: str
    route_to: str

class GraderRequest(BaseModel):
    task_id: str
    actions: list[dict]

class BaselineRequest(BaseModel):
    task_ids: list[str] = ["easy", "medium", "hard"]


@app.get("/")
def root():
    """Health check - returns 200."""
    return {
        "status": "ok",
        "environment": "Email Triage OpenEnv",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"],
    }

@app.get("/reset")
@app.post("/reset")
async def reset(request: Request, task_id: str = "easy"):
    """
    Start a new episode. Accepts GET or POST with optional task_id.
    POST body: {"task_id": "easy"} (optional)
    Returns the first email observation.
    """
    try:
        # Try to parse JSON body if POST
        try:
            body = await request.json()
            if isinstance(body, dict):
                task_id = body.get("task_id", task_id)
        except Exception:
            pass  # No body or not JSON - use default

        obs = env.reset(task_id=task_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step(request: StepRequest):
    """Take one action (classify one email)."""
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
    """Return all tasks with action schema."""
    return env.get_tasks()

@app.post("/grader")
def grader(request: GraderRequest):
    """Score a completed episode given a list of actions."""
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
    """Run baseline inference script against all requested tasks."""
    results = {}
    for task_id in request.task_ids:
        try:
            result = subprocess.run(
                [sys.executable, "inference.py", "--task", task_id],
                capture_output=True, text=True, timeout=120
            )
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
