#!/bin/bash
# RunPod OdinFold Production Benchmark Deployment
# Optimized for RunPod GPU instances

set -e

echo "🚀 RunPod OdinFold Production Benchmark Deployment"
echo "=================================================="

# Configuration
DEFAULT_PORT=8000
DEFAULT_HOST="0.0.0.0"
WORKSPACE_DIR="/workspace"
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
        --workspace)
            WORKSPACE_DIR="$2"
            shift 2
            ;;
        --setup-only)
            SETUP_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --port PORT        Server port (default: 8000)"
            echo "  --host HOST        Server host (default: 0.0.0.0)"
            echo "  --workspace DIR    Workspace directory (default: /workspace)"
            echo "  --setup-only       Only setup environment, don't start server"
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

echo "📋 RunPod Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Workspace: $WORKSPACE_DIR"

# Change to workspace directory
cd $WORKSPACE_DIR

# Check RunPod environment
echo "🔍 Checking RunPod Environment..."
echo "   Python: $(python --version)"
echo "   CUDA: $(nvcc --version | grep release || echo 'CUDA not found')"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🔥 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | while read line; do
        echo "   $line"
    done
else
    echo "⚠️  nvidia-smi not found"
fi

# Check PyTorch CUDA
python -c "import torch; print(f'   PyTorch CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "   PyTorch not available"

# Install dependencies
echo "📦 Installing Dependencies..."
pip install --quiet fastapi uvicorn pandas requests torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --quiet biopython matplotlib seaborn plotly

# Clone OpenFold if not present
if [[ ! -d "openfold" ]]; then
    echo "📥 Cloning OpenFold repository..."
    git clone https://github.com/euticus/openfold.git
    cd openfold
else
    echo "✅ OpenFold repository found"
    cd openfold
fi

# Check if model weights exist
echo "🔍 Checking Model Weights..."
WEIGHTS_DIR="openfold/resources/openfold_params"
if [[ -d "$WEIGHTS_DIR" ]]; then
    echo "✅ Model weights directory found:"
    ls -la "$WEIGHTS_DIR"/*.pt | head -5
    WEIGHTS_COUNT=$(ls "$WEIGHTS_DIR"/*.pt 2>/dev/null | wc -l)
    echo "   Total weight files: $WEIGHTS_COUNT"
else
    echo "⚠️  Model weights not found. Downloading..."
    mkdir -p "$WEIGHTS_DIR"
    
    # Download key model weights
    echo "📥 Downloading OpenFold model weights..."
    wget -q -O "$WEIGHTS_DIR/openfold_model_1_ptm.pt" \
        "https://files.ipd.uw.edu/pub/openfold/openfold_model_1_ptm.pt" || \
        echo "⚠️  Failed to download weights - will use random initialization"
fi

# Check CASP datasets
echo "🔍 Checking CASP Datasets..."
if [[ -d "demo_dataset" ]]; then
    echo "✅ Demo dataset found:"
    find demo_dataset -name "*.fasta" | head -5
    FASTA_COUNT=$(find demo_dataset -name "*.fasta" | wc -l)
    echo "   Total FASTA files: $FASTA_COUNT"
else
    echo "⚠️  Demo dataset not found - will use built-in sequences"
fi

# Create benchmark configuration for RunPod
echo "⚙️  Creating RunPod Benchmark Configuration..."
cat > benchmark_config_runpod.json << EOF
{
  "models": {
    "openfold_model_1_ptm": {
      "enabled": true,
      "weights_path": "openfold/resources/openfold_params/openfold_model_1_ptm.pt",
      "config_preset": "model_1_ptm"
    },
    "openfold_finetuning_ptm_1": {
      "enabled": true,
      "weights_path": "openfold/resources/openfold_params/finetuning_ptm_1.pt",
      "config_preset": "model_1_ptm"
    },
    "openfold_finetuning_ptm_2": {
      "enabled": true,
      "weights_path": "openfold/resources/openfold_params/finetuning_ptm_2.pt",
      "config_preset": "model_1_ptm"
    }
  },
  "datasets": {
    "demo_dataset": {
      "fasta_dir": "demo_dataset/fasta",
      "pdb_dir": "demo_dataset/pdb",
      "enabled": true
    },
    "casp14_targets": [
      "T1024", "T1027", "T1030", "T1040", "T1050"
    ]
  },
  "hardware": {
    "device": "cuda",
    "mixed_precision": true,
    "max_memory_gb": 70
  },
  "output": {
    "results_dir": "runpod_benchmark_results",
    "save_structures": true,
    "save_metrics": true
  }
}
EOF

echo "✅ RunPod configuration created: benchmark_config_runpod.json"

# Test benchmark setup
echo "🧪 Testing Benchmark Setup..."
python -c "
import sys
sys.path.insert(0, '.')
try:
    from production_benchmark_setup import ProductionBenchmarkRunner
    runner = ProductionBenchmarkRunner('benchmark_config_runpod.json')
    print('✅ Benchmark runner initialized successfully')
    
    # Test GPU
    import torch
    if torch.cuda.is_available():
        print(f'✅ GPU available: {torch.cuda.get_device_name(0)}')
        print(f'✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
    else:
        print('⚠️  GPU not available')
        
except Exception as e:
    print(f'❌ Setup test failed: {e}')
    sys.exit(1)
"

if [[ "$SETUP_ONLY" == "true" ]]; then
    echo "✅ Setup completed! Use --setup-only=false to start server"
    echo ""
    echo "🚀 To start the server manually:"
    echo "   python deploy_benchmark_server.py --host $HOST --port $PORT"
    echo ""
    echo "🌐 Server will be available at:"
    echo "   Health: http://localhost:$PORT/health"
    echo "   API docs: http://localhost:$PORT/docs"
    exit 0
fi

# Start the benchmark server
echo "🌐 Starting RunPod Benchmark Server..."
echo "   URL: http://$HOST:$PORT"
echo "   Health check: http://$HOST:$PORT/health"
echo "   API docs: http://$HOST:$PORT/docs"
echo ""
echo "🔥 Server starting with ALL model weights and CASP datasets..."
echo "   Press Ctrl+C to stop"

# Set environment variables for the server
export BENCHMARK_CONFIG="benchmark_config_runpod.json"
export RUNPOD_MODE="true"

python deploy_benchmark_server.py --host $HOST --port $PORT
