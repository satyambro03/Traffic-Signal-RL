# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install torch gym numpy pyyaml

# Command to run inference
CMD ["python", "inference.py"]