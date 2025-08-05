# OdinFold Production Docker Image
# Multi-stage build for both CPU and GPU environments

ARG CUDA_VERSION=11.8
ARG PYTHON_VERSION=3.9

# Base stage with common dependencies
FROM python:${PYTHON_VERSION}-slim as base

# metainformation
LABEL org.opencontainers.image.version = "1.0.0"
LABEL org.opencontainers.image.authors = "OdinFold Team"
LABEL org.opencontainers.image.source = "https://github.com/euticus/openfold"
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.description = "OdinFold - The engine that powers FoldForever"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY setup.py* ./
COPY README.md* ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel

# CPU-only stage
FROM base as cpu

# Install CPU-only PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Install OdinFold in development mode
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 odinfold && chown -R odinfold:odinfold /app
USER odinfold

# Expose port for API
EXPOSE 8000

# Default command
CMD ["python", "-m", "odinfold.api.server"]

# GPU stage with CUDA support
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 as gpu-base

# Install Python
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    build-essential \
    git \
    wget \
    curl \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY setup.py* ./
COPY README.md* ./

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# GPU stage
FROM gpu-base as gpu

# Install GPU PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Install OdinFold in development mode
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 odinfold && chown -R odinfold:odinfold /app
USER odinfold

# Expose port for API
EXPOSE 8000

# Default command
CMD ["python", "-m", "odinfold.api.server"]

# Benchmark stage for CI/CD
FROM gpu as benchmark

# Switch back to root for benchmark setup
USER root

# Install additional benchmark dependencies
RUN pip install pytest pytest-benchmark matplotlib seaborn

# Copy benchmark scripts
COPY scripts/ scripts/
COPY tests/ tests/

# Create benchmark entrypoint
RUN echo '#!/bin/bash\n\
set -e\n\
echo "🔥 Starting OdinFold Benchmark"\n\
python scripts/evaluation/gpu_ci_benchmark.py --config production --quick\n\
echo "✅ Benchmark complete"\n\
' > /app/benchmark.sh && chmod +x /app/benchmark.sh

# Switch back to odinfold user
USER odinfold

# Benchmark command
CMD ["/app/benchmark.sh"]

# Production API stage
FROM gpu as production

# Install production dependencies
USER root
RUN pip install gunicorn uvicorn[standard] prometheus-client

# Create production entrypoint
RUN echo '#!/bin/bash\n\
set -e\n\
echo "⚡ Starting OdinFold Production API"\n\
exec gunicorn odinfold.api.server:app \\\n\
    --worker-class uvicorn.workers.UvicornWorker \\\n\
    --workers 2 \\\n\
    --bind 0.0.0.0:8000 \\\n\
    --timeout 300 \\\n\
    --keep-alive 2 \\\n\
    --max-requests 1000 \\\n\
    --max-requests-jitter 50 \\\n\
    --log-level info\n\
' > /app/production.sh && chmod +x /app/production.sh

USER odinfold

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["/app/production.sh"]
