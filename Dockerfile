# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies with fixed versions
RUN pip install gymnasium numpy pyyaml gradio==4.29.0 huggingface_hub==0.23.0

# Command to run inference (Gradio UI)
CMD ["python", "inference.py"]