# Multi-stage Dockerfile for Cyber-LLM
# Optimized for offline deployment and security

FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash cyberllm
USER cyberllm
WORKDIR /home/cyberllm

# Copy requirements first for better caching
COPY --chown=cyberllm:cyberllm requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --user --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    jupyter

# Copy source code
COPY --chown=cyberllm:cyberllm . .

# Set Python path
ENV PYTHONPATH=/home/cyberllm:$PYTHONPATH

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Production stage
FROM base as production

# Copy only necessary files
COPY --chown=cyberllm:cyberllm src/ ./src/
COPY --chown=cyberllm:cyberllm configs/ ./configs/
COPY --chown=cyberllm:cyberllm adapters/ ./adapters/
COPY --chown=cyberllm:cyberllm scripts/ ./scripts/

# Set Python path
ENV PYTHONPATH=/home/cyberllm:$PYTHONPATH

# Create directories for data and logs
RUN mkdir -p data/raw data/processed logs models

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# CLI stage for command-line operations
FROM base as cli

COPY --chown=cyberllm:cyberllm src/ ./src/
COPY --chown=cyberllm:cyberllm configs/ ./configs/
COPY --chown=cyberllm:cyberllm scripts/ ./scripts/

# Set Python path
ENV PYTHONPATH=/home/cyberllm:$PYTHONPATH

# Make CLI script executable
RUN chmod +x scripts/*.py

ENTRYPOINT ["python", "src/deployment/cli/cyber_cli.py"]

# Training stage for model training
FROM base as training

# Install additional training dependencies
RUN pip install --user --no-cache-dir \
    accelerate \
    deepspeed \
    wandb

COPY --chown=cyberllm:cyberllm . .

# Set Python path
ENV PYTHONPATH=/home/cyberllm:$PYTHONPATH

# Create training directories
RUN mkdir -p models/checkpoints logs/training data/processed

CMD ["python", "src/training/train.py"]
