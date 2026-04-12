import os
import numpy as np
from fastapi import FastAPI, Request

# Import environments
from env import TrafficSignalEnv, EmailSortEnv, MultiIntersectionEnv

# ==============================
# ENV VARIABLES (SAFE)
# ==============================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

client = None

# Safe LLM client
try:
    if HF_TOKEN:
        from openai import OpenAI
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN
        )
        print("[INFO] LLM client initialized", flush=True)
    else:
        print("[WARN] HF_TOKEN missing, fallback mode", flush=True)
except Exception as e:
    print(f"[ERROR] Client init failed: {e}", flush=True)
    client = None

# ==============================
# FASTAPI APP
# ==============================
app = FastAPI()

env_map = {
    "TrafficSignal": TrafficSignalEnv,
    "EmailSort": EmailSortEnv,
    "MultiIntersection": MultiIntersectionEnv
}

# ==============================
# RESET ENDPOINT
# ==============================
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

# ==============================
# ACTION FUNCTION (LLM + FALLBACK)
# ==============================
def get_action(state, action_space_n):
    if client is None:
        return np.random.randint(0, action_space_n)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": f"State: {state.tolist()} return number 0 to {action_space_n-1}"
            }],
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
# MAIN TASK RUNNER
# ==============================
def run_task(task_name):
    rewards = []
    steps = 0
    success = False

    print(f"[START] task={task_name} env={task_name.lower()} model={MODEL_NAME}", flush=True)

    try:
        env = env_map[task_name]()
        state, _ = env.reset()
        done = False

        while not done:
            action = get_action(state, env.action_space.n)

            next_state, reward, done, truncated, _ = env.step(action)

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
        success = score > 0

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)

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

# ==============================
# FASTAPI STARTUP (HF UI)
# ==============================
@app.on_event("startup")
async def startup_event():
    print("===== Application Startup =====", flush=True)

    run_task("TrafficSignal")
    run_task("EmailSort")
    run_task("MultiIntersection")

# ==============================
# ROOT ENDPOINT
# ==============================
@app.get("/")
async def root():
    return {"status": "running"}

# ==============================
# IMPORTANT: DIRECT EXECUTION (VALIDATOR)
# ==============================
if __name__ == "__main__":
    print("===== Direct Execution Mode =====", flush=True)

    run_task("TrafficSignal")
    run_task("EmailSort")
    run_task("MultiIntersection")
