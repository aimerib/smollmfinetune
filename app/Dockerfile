# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-runpod.txt .
RUN pip install --no-cache-dir -r requirements-runpod.txt

# Copy application code
COPY . .

# Create directories for training outputs
RUN mkdir -p training_output/adapters training_output/prompts

# Environment variables for RunPod deployment
ENV INFERENCE_ENGINE=vllm
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# vLLM configuration for PersonalityEngine model (optimized for memory)
ENV VLLM_MODEL=PocketDoc/Dans-PersonalityEngine-V1.3.0-24b
ENV VLLM_GPU_MEMORY_UTILIZATION=0.90
ENV VLLM_MAX_MODEL_LEN=2048

# Expose Streamlit port
EXPOSE 8888

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8888/_stcore/health

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8888", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"] 