FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

# For hackathon submission (STDOUT logs)
# CMD ["python", "-m", "server.app"]

# For hackathon submission (structured logs to stdout)
CMD ["python", "inference.py"]



# For demo mode (Gradio UI)
# CMD ["python", "ui_inference.py"]
