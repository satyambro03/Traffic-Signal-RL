# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install gymnasium numpy pyyaml gradio

# Command to run inference (Gradio UI)
CMD ["python", "inference.py"]