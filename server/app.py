"""
FastAPI Server - Email Triage OpenEnv
server/app.py - Required by openenv validate
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess

from environment import EmailTriageEnvironment, Action

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
    return {
        "status": "ok",
        "environment": "Email Triage OpenEnv",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"],
    }

@app.get("/reset")
@app.post("/reset")
async def reset(request: Request, task_id: str = "easy"):
    try:
        try:
            body = await request.json()
            if isinstance(body, dict):
                task_id = body.get("task_id", task_id)
        except Exception:
            pass
        obs = env.reset(task_id=task_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step(request: StepRequest):
    try:
        action = Action(
            priority=request.priority,
            category=request.category,
            route_to=request.route_to,
        )
        obs, reward, done, info = env.step(action)
        return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state():
    return env.state()

@app.get("/tasks")
def tasks():
    return env.get_tasks()

@app.post("/grader")
def grader(request: GraderRequest):
    try:
        score = env.grade_episode(request.task_id, request.actions)
        return {"task_id": request.task_id, "score": score, "num_actions": len(request.actions)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/baseline")
def baseline(request: BaselineRequest):
    results = {}
    for task_id in request.task_ids:
        try:
            result = subprocess.run(
                [sys.executable, "inference.py", "--task", task_id],
                capture_output=True, text=True, timeout=120
            )
            for line in result.stdout.splitlines():
                if "Final score" in line:
                    results[task_id] = float(line.split(":")[-1].strip())
                    break
            else:
                results[task_id] = None
        except Exception as e:
            results[task_id] = f"error: {str(e)}"
    return {"baseline_scores": results}


def main():
    """Main entry point - required by openenv validate."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
