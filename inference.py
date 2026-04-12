import os
import numpy as np
from fastapi import FastAPI, Request
from openai import OpenAI

# Import your environments
from env import TrafficSignalEnv, EmailSortEnv, MultiIntersectionEnv

# ✅ REQUIRED ENV VARIABLES (Hackathon injects these)
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")

# ✅ OpenAI client (MANDATORY for validation)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# FastAPI app
app = FastAPI()

# ✅ Startup event (runs tasks automatically)
@app.on_event("startup")
async def startup_event():
    print("===== Application Startup =====", flush=True)

    run_task("TrafficSignal")
    run_task("EmailSort")
    run_task("MultiIntersection")


@app.get("/")
async def root():
    return {"status": "running", "message": "FastAPI server is live. Use POST /reset."}


# Map tasks to envs
env_map = {
    "TrafficSignal": TrafficSignalEnv,
    "EmailSort": EmailSortEnv,
    "MultiIntersection": MultiIntersectionEnv
}


@app.post("/reset")
async def reset_endpoint(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}

    task_id = data.get("task_id") or data.get("task") or "TrafficSignal"

    if task_id not in env_map:
        return {"status": "error", "task": task_id, "message": f"Unknown task_id {task_id}"}

    try:
        env = env_map[task_id]()
        state, _ = env.reset()
        env.close()
        return {"status": "ok", "task": task_id, "state": state.tolist()}
    except Exception as e:
        return {"status": "error", "task": task_id, "message": str(e)}


def get_action_from_llm(state, action_space_n):
    """
    Call LLM to decide action
    """
    try:
        prompt = f"""
You are an RL agent.
State: {state.tolist()}
Return ONLY a number between 0 and {action_space_n - 1}.
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5
        )

        action = int(response.choices[0].message.content.strip())

        # Safety clamp
        if action < 0 or action >= action_space_n:
            action = np.random.randint(0, action_space_n)

    except Exception as e:
        print(f"[ERROR] LLM failed: {e}", flush=True)
        action = np.random.randint(0, action_space_n)

    return action


def run_task(task_name="TrafficSignal"):
    env_class = env_map.get(task_name)

    if env_class is None:
        print(f"[START] task={task_name} error=Unknown task", flush=True)
        print(f"[END] success=false steps=0 rewards=", flush=True)
        return

    rewards = []
    steps = 0
    success = False

    print(f"[START] task={task_name} env={task_name.lower()} model={MODEL_NAME}", flush=True)

    try:
        env = env_class()
        state, _ = env.reset()
        done = False

        while not done:
            # ✅ LLM-based action
            action = get_action_from_llm(state, env.action_space.n)

            next_state, reward, done, truncated, info = env.step(action)

            if truncated:
                done = True

            steps += 1
            rewards.append(float(reward))

            print(
                f"[STEP] step={steps} action={action} reward={reward:.2f} "
                f"done={str(done).lower()} error=null",
                flush=True
            )

            state = next_state

        score = float(np.mean(rewards)) if rewards else 0.0
        success = score >= 0.1

    except Exception as e:
        print(f"[ERROR] run_task failed: {e}", flush=True)

    finally:
        try:
            env.close()
        except:
            pass

        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        print(
            f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
            flush=True
        )

    return {"task": task_name, "steps": steps, "success": success}


# Optional fallback
if __name__ == "__main__":
    run_task("TrafficSignal")
    run_task("EmailSort")
    run_task("MultiIntersection")
    # ❌ DO NOT START UVICORN HERE
    # HuggingFace / validator khud server start karega

    # import uvicorn, os
    # port = int(os.getenv("PORT", 7860))
    # uvicorn.run(app, host="0.0.0.0", port=port)
