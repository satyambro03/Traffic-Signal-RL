import gradio as gr
import numpy as np
from env import EmailSortEnv, TrafficSignalEnv, MultiIntersectionEnv

def run_task(task_name):
    env_map = {
        "EmailSort": EmailSortEnv,
        "TrafficSignal": TrafficSignalEnv,
        "MultiIntersection": MultiIntersectionEnv
    }
    env_class = env_map[task_name]
    env = env_class()
    scores = []
    for _ in range(10):
        state, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            scores.append(reward)
    return f"{task_name} Average score: {np.mean(scores):.2f}"

demo = gr.Interface(
    fn=run_task,
    inputs=gr.Dropdown(["EmailSort", "TrafficSignal", "MultiIntersection"], label="Choose Task"),
    outputs="text",
    title="Traffic Signal RL Demo",
    description="Select a task to run baseline agent and see average score."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)