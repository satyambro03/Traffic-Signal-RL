import os
import numpy as np
from fastapi import FastAPI, Request
from openai import OpenAI

# Import environments
from env import TrafficSignalEnv, EmailSortEnv, MultiIntersectionEnv

# ==============================
# ✅ ENV VARIABLES (MANDATORY)
# ==============================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ✅ OpenAI Client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# ==============================
# ✅ FASTAPI APP
# ==============================
app = FastAPI()

# Map environments
env_map = {
    "TrafficSignal": TrafficSignalEnv,
    "EmailSort": EmailSortEnv,
    "MultiIntersection": MultiIntersectionEnv
}

# ==============================
# ✅ LLM ACTION FUNCTION
# ==============================
def get_action_from_llm(state, action_space_n):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": f"State: {state.tolist()}. Return ONLY integer between 0 and {action_space_n-1}"
                }
            ],
            max_tokens=5
        )

        action = int(response.choices[0].message.content.strip())

        if action < 0 or action >= action_space_n:
            action = np.random.randint(0, action_space_n)

    except Exception as e:
        print(f"[ERROR] LLM failed: {e}", flush=True)
        action = np.random.randint(0, action_space_n)

    return action


# ==============================
# ✅ RUN TASK FUNCTION
# ==============================
def run_task(task_name):
    env = env_map[task_name]()

    rewards = []
    steps = 0
    success = False

    # START LOG
    print(f"[START] task={task_name} env={task_name.lower()} model={MODEL_NAME}", flush=True)

    try:
        state, _ = env.reset()
        done = False

        while not done:
            action = get_action_from_llm(state, env.action_space.n)

            next_state, reward, done, truncated, _ = env.step(action)

            if truncated:
                done = True

            steps += 1
            rewards.append(float(reward))

            # STEP LOG
            print(
                f"[STEP] step={steps} action={action} reward={reward:.2f} "
                f"done={str(done).lower()} error=null",
                flush=True
            )

            state = next_state

        score = float(np.mean(rewards)) if rewards else 0.0
        success = score > 0

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)

    finally:
        env.close()

        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        # END LOG
        print(
            f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
            flush=True
        )


# ==============================
# ✅ STARTUP EVENT (AUTO RUN)
# ==============================
@app.on_event("startup")
async def startup_event():
    print("===== Application Startup =====", flush=True)

    run_task("TrafficSignal")
    run_task("EmailSort")
    run_task("MultiIntersection")


# ==============================
# ✅ OPTIONAL ROOT ENDPOINT
# ==============================
@app.get("/")
async def root():
    return {"status": "running"}


@app.post("/reset")
async def reset_endpoint(request: Request):
    try:
        data = await request.json()
    except:
        data = {}

    task = data.get("task", "TrafficSignal")

    if task not in env_map:
        return {"status": "error", "message": "Invalid task"}

    env = env_map[task]()
    state, _ = env.reset()
    env.close()

    return {
        "status": "ok",
        "task": task,
        "state": state.tolist()
    }
