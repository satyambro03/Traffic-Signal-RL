FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

# For hackathon submission (structured logs to stdout)
CMD ["python", "inference.py"]

# For demo mode (Gradio UI) - commented out
# CMD ["python", "ui_inference.py"]
# CMD ["python", "-m", "server.app"]
