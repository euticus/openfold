#!/bin/bash

# RunPod Startup Script for FoldForever Benchmark
echo "🚀 Setting up FoldForever Benchmark on RunPod A100..."

# Update system
apt-get update && apt-get install -y git wget curl

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Clone repository (if needed)
if [ ! -d "openfold" ]; then
    git clone https://github.com/euticus/openfold.git
    cd openfold
else
    cd openfold
    git pull
fi

# Install OpenFold
pip install -e .

# Verify GPU
echo "🔍 GPU Information:"
nvidia-smi

echo "🧪 Starting FoldForever Benchmark..."
python benchmark_casp14_foldforever_vs_baselines.py \
    --mode full \
    --gpu \
    --sequences 30 \
    --output /workspace/results \
    --timeout 600 \
    --verbose

echo "✅ Benchmark completed! Results in /workspace/results/"
