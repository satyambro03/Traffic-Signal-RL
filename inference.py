import os
import numpy as np
import gradio as gr
from env import TrafficSignalEnv, EmailSortEnv, MultiIntersectionEnv

# Environment variables (can be set in Hugging Face Space settings)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "baseline-model")
HF_TOKEN = os.getenv("HF_TOKEN")

def run_task(task_name="TrafficSignal"):
    # Map task names to environment classes
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
            # Random baseline action
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            steps += 1
            rewards.append(reward)

            print(
                f"[STEP] step={steps} action={action} reward={reward:.2f} "
                f"done={str(done).lower()} error=null",
                flush=True
            )
            state = next_state

        score = min(max(np.mean(rewards), 0.0), 1.0)
        success = score >= 0.1

    finally:
        env.close()
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} steps={steps} "
            f"score={score:.2f} rewards={rewards_str}",
            flush=True
        )

    # ✅ Return summary string for UI
    return f"Task={task_name}, Steps={steps}, Score={score:.2f}, Success={success}, Rewards={rewards_str}"

# Gradio interface for Hugging Face Space
iface = gr.Interface(
    fn=run_task,
    inputs=gr.Dropdown(["TrafficSignal", "EmailSort", "MultiIntersection"], label="Select Task"),
    outputs="text",
    title="Multi-Task RL Environment",
    description="Run baseline random agent on TrafficSignal, EmailSort, or MultiIntersection environments."
)

if __name__ == "__main__":
    iface.launch()