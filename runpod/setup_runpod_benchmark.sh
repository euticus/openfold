#!/bin/bash

# RunPod A100 Benchmark Setup Script
# Cost: ~$1.89/hour for A100 80GB (vs $20+/hour on Azure VMs)

echo "🚀 RunPod A100 Benchmark Setup"
echo "==============================="
echo "💰 Cost: ~$1.89/hour for A100 80GB"
echo "⚡ Direct GPU access, no VM overhead"
echo ""

cat << 'EOF'
📋 SETUP INSTRUCTIONS:

1. 🌐 Go to RunPod.io and create account
2. 💳 Add $20-50 credit (enough for extensive testing)
3. 🖥️ Create new pod:
   - Template: "PyTorch 2.0"
   - GPU: "RTX A6000" or "A100 80GB" 
   - Storage: 50GB
   - Expose HTTP port: 8888

4. 📁 Upload benchmark files:
   - benchmark_casp14_foldforever_vs_baselines.py
   - requirements.txt (generated below)

5. 🏃 Run benchmark:
   ```bash
   python benchmark_casp14_foldforever_vs_baselines.py --mode full --gpu --sequences 30
   ```

EOF

# Generate requirements.txt for RunPod
cat > requirements.txt << 'EOF'
torch>=2.0.0
torchvision
torchaudio
transformers>=4.21.0
accelerate
matplotlib
seaborn
pandas
numpy
scipy
psutil
biopython
requests
tqdm
openfold
EOF

echo "📦 Generated requirements.txt for RunPod"
echo ""

# Generate RunPod startup script
cat > runpod_startup.sh << 'EOF'
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
EOF

chmod +x runpod_startup.sh

echo "🎯 ALTERNATIVE: Vast.ai Setup"
echo "============================="
cat << 'EOF'

🌟 Vast.ai (Even Cheaper!):
1. Go to vast.ai
2. Search for: "A100" or "RTX 4090"
3. Filter by: PyTorch, CUDA 11.8+
4. Rent for $1.50-2.50/hour
5. SSH in and run benchmark

💡 Pro tip: Look for "interruptible" instances for 50% discount!

EOF

echo ""
echo "📊 COST COMPARISON:"
echo "==================="
echo "Azure VM A100:     $20-25/hour  💸"
echo "Azure ML Compute:  $3-4/hour    💰"
echo "RunPod A100:       $1.89/hour   🎯"
echo "Vast.ai A100:      $1.50/hour   🏆"
echo "Colab Pro+:        $50/month    📱"
echo ""

echo "🚀 RECOMMENDED APPROACH:"
echo "========================"
echo "1. Start with RunPod A100 ($1.89/hour)"
echo "2. Run 2-3 hour comprehensive benchmark (~$6 total)"
echo "3. Download results and analysis"
echo "4. Scale up if needed"
echo ""

echo "✅ Setup files created:"
echo "  • requirements.txt"
echo "  • runpod_startup.sh"
echo ""
echo "🎯 Next: Go to runpod.io and create your A100 instance!"
EOF

<function_calls>
<invoke name="launch-process">
<parameter name="command">chmod +x setup_runpod_benchmark.sh && ./setup_runpod_benchmark.sh
