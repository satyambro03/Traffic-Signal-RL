# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies (lightweight)
RUN pip install gymnasium numpy pyyaml

# Command to run inference
CMD ["python", "inference.py"]