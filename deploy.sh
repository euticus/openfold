#!/bin/bash
# OdinFold Production Benchmark Deployment Script

set -e

echo "🚀 OdinFold Production Benchmark Deployment"
echo "=========================================="

# Configuration
DEFAULT_PORT=8000
DEFAULT_HOST="0.0.0.0"
CONDA_ENV="openfold"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        --gpu-check)
            GPU_CHECK=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --port PORT        Server port (default: 8000)"
            echo "  --host HOST        Server host (default: 0.0.0.0)"
            echo "  --env ENV          Conda environment (default: openfold)"
            echo "  --install-deps     Install dependencies"
            echo "  --gpu-check        Check GPU availability"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set defaults
PORT=${PORT:-$DEFAULT_PORT}
HOST=${HOST:-$DEFAULT_HOST}

echo "📋 Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Conda env: $CONDA_ENV"

# Check if conda environment exists
if ! conda env list | grep -q "^$CONDA_ENV "; then
    echo "❌ Conda environment '$CONDA_ENV' not found"
    echo "   Create it with: conda create -n $CONDA_ENV python=3.9"
    exit 1
fi

# Activate conda environment
echo "🔧 Activating conda environment: $CONDA_ENV"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Check GPU availability
if [[ "$GPU_CHECK" == "true" ]]; then
    echo "🔍 Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    else
        echo "⚠️  nvidia-smi not found"
    fi

    python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    fi
fi

# Install dependencies if requested
if [[ "$INSTALL_DEPS" == "true" ]]; then
    echo "📦 Installing dependencies..."
    pip install fastapi uvicorn pandas requests torch
    echo "✅ Dependencies installed"
fi

# Check if required files exist
REQUIRED_FILES=("production_benchmark_setup.py" "deploy_benchmark_server.py")
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ Required file not found: $file"
        exit 1
    fi
done

echo "✅ All required files found"

# Start the server
echo "🌐 Starting OdinFold Benchmark Server..."
echo "   URL: http://$HOST:$PORT"
echo "   Health check: http://$HOST:$PORT/health"
echo "   API docs: http://$HOST:$PORT/docs"
echo ""
echo "🔥 Server starting... (Ctrl+C to stop)"

python deploy_benchmark_server.py --host $HOST --port $PORT