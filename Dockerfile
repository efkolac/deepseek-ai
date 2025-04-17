# Use official Python image as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MODEL_BASE_PATH=/models

# Create directory structure (adjust according to your model paths)
RUN mkdir -p ${MODEL_BASE_PATH}/DeepSeek-R1-Distill-Qwen-1.5B

# Install system dependencies (adjust based on your needs)
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy your application code
COPY . /app
WORKDIR /app

# Copy your models (option 1 - if models are in your project)
# COPY ./models/DeepSeek-R1-Distill-Qwen-1.5B /models/DeepSeek-R1-Distill-Qwen-1.5B

# Entrypoint
CMD ["python", "app.py"]