#!/bin/bash

# 🚀 Deploy Production Benchmark Server to RunPod
echo "🚀 Deploying Production Benchmark Server to RunPod..."

# Upload the production server
echo "📤 Uploading production server..."
scp -o StrictHostKeyChecking=no production_benchmark_server.py root@5ocnemvgivdwzq-8888.proxy.runpod.net:/workspace/

# Upload benchmark script
echo "📤 Uploading benchmark script..."
scp -o StrictHostKeyChecking=no benchmark_casp14_foldforever_vs_baselines.py root@5ocnemvgivdwzq-8888.proxy.runpod.net:/workspace/

# Connect to RunPod and setup
echo "🔧 Setting up production environment..."
ssh -o StrictHostKeyChecking=no root@5ocnemvgivdwzq-8888.proxy.runpod.net << 'EOF'

# Kill existing server
echo "🛑 Stopping existing server..."
pkill -f python
pkill -f uvicorn
sleep 3

# Install additional dependencies
echo "📦 Installing dependencies..."
pip install transformers torch torchvision torchaudio
pip install biopython biotite
pip install huggingface_hub

# Download CASP14 data if not exists
echo "📥 Setting up CASP14 dataset..."
mkdir -p /workspace/casp14_data/fasta
mkdir -p /workspace/casp14_data/pdb

# Copy demo data to CASP14 directory
if [ -d "/workspace/openfold/demo_dataset" ]; then
    echo "📋 Copying demo CASP data..."
    cp /workspace/openfold/demo_dataset/fasta/* /workspace/casp14_data/fasta/ 2>/dev/null || true
    cp /workspace/openfold/demo_dataset/pdb/* /workspace/casp14_data/pdb/ 2>/dev/null || true
fi

# Check model weights
echo "🔍 Checking model weights..."
WEIGHTS_DIR="/workspace/openfold/resources/openfold_params"
if [ -d "$WEIGHTS_DIR" ]; then
    echo "✅ Found model weights:"
    ls -la "$WEIGHTS_DIR"/*.pt | head -3
else
    echo "⚠️  Model weights not found at $WEIGHTS_DIR"
fi

# Start production server
echo "🚀 Starting production benchmark server..."
cd /workspace
nohup python3 production_benchmark_server.py > server.log 2>&1 &

# Wait for server to start
sleep 10

# Test server
echo "🧪 Testing server..."
curl -s http://localhost:8888/health | python3 -m json.tool

echo "✅ Production server deployed!"
echo "🌐 Access at: https://5ocnemvgivdwzq-8888.proxy.runpod.net"

EOF

echo "🎉 Deployment complete!"