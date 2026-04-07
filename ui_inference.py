import gradio as gr
from inference import run_task

def run_with_ui(task_name):
    return run_task(task_name)  # now returns summary string

demo = gr.Interface(
    fn=run_with_ui,
    inputs=gr.Dropdown(["TrafficSignal", "EmailSort", "MultiIntersection"], label="Choose Task"),
    outputs="text",
    title="Traffic Signal RL Demo",
    description="Select a task to run agent and see average score."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)