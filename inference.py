import os
import numpy as np
from env import TrafficSignalEnv, EmailSortEnv, MultiIntersectionEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "baseline-model")
HF_TOKEN = os.getenv("HF_TOKEN")

def run_task(task_name="TrafficSignal"):
    env_map = {
        "TrafficSignal": TrafficSignalEnv,
        "EmailSort": EmailSortEnv,
        "MultiIntersection": MultiIntersectionEnv
    }
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
            steps += 1
            rewards.append(reward)

            print(f"[STEP] step={steps} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            state = next_state

        score = min(max(np.mean(rewards), 0.0), 1.0)
        success = score >= 0.1

    finally:
        env.close()
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

    # ✅ Return summary string for UI
    return f"Task={task_name}, Steps={steps}, Score={score:.2f}, Success={success}, Rewards={rewards_str}"

if __name__ == "__main__":
    run_task("TrafficSignal")