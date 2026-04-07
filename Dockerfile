FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir gymnasium numpy pyyaml gradio openai huggingface_hub==0.23.0

# For hackathon submission (STDOUT logs)
# CMD ["python", "inference.py"]

# For demo mode (Gradio UI)
# CMD ["python", "ui_inference.py"]
CMD ["python", "ui_inference.py"]