import os
import numpy as np
from fastapi import FastAPI, Request
import uvicorn
from openai import OpenAI

# Import your environments
from env import TrafficSignalEnv, EmailSortEnv, MultiIntersectionEnv

# Environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# FastAPI app
app = FastAPI()

# Map tasks to envs
env_map = {
    "TrafficSignal": TrafficSignalEnv,
    "EmailSort": EmailSortEnv,
    "MultiIntersection": MultiIntersectionEnv
}

@app.post("/reset")
async def reset_endpoint(request: Request):
    """
    Hugging Face validator calls POST /reset.
    Body: {"task_id": "TrafficSignal"} or {"task": "TrafficSignal"}.
    """
    data = await request.json()
    task_id = data.get("task_id") or data.get("task") or "TrafficSignal"

    if task_id not in env_map:
        return {"error": f"Unknown task_id {task_id}"}

    env = env_map[task_id]()
    state, _ = env.reset()
    env.close()
    return {"status": "ok", "task": task_id, "state": state.tolist()}

def run_task(task_name="TrafficSignal"):
    env_class = env_map[task_name]
    env = env_class()
    rewards = []
    steps = 0
    success = False

    print(f"[START] task={task_name} env={task_name.lower()} model={MODEL_NAME}", flush=True)

    try:
        state, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            if truncated:
                done = True
            steps += 1
            rewards.append(float(reward))
            print(f"[STEP] step={steps} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            state = next_state
        score = float(np.mean(rewards)) if rewards else 0.0
        success = score >= 0.1
    finally:
        env.close()
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

    return {"task": task_name, "steps": steps, "success": success}

if __name__ == "__main__":
    # Run FastAPI server so HF Space exposes /reset
    uvicorn.run(app, host="0.0.0.0", port=7860)