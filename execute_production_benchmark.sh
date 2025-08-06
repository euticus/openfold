#!/bin/bash

# 🚀 Execute Production Benchmark on RunPod
echo "🚀 EXECUTING PRODUCTION BENCHMARK ON RUNPOD"
echo "=============================================="

# Create the execution command for RunPod
RUNPOD_COMMAND='
cd /workspace/openfold

echo "🔍 Environment Check..."
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

echo "📦 Model Weights Check..."
ls -la resources/openfold_params/*.pt | head -5

echo "📊 CASP Data Check..."
ls -la demo_dataset/fasta/
ls -la demo_dataset/pdb/

echo "🚀 Starting Production Benchmark..."
echo "   - Mode: full (production)"
echo "   - GPU: enabled"
echo "   - Sequences: 10 CASP14 targets"
echo "   - Timeout: 600s per prediction"
echo "   - Output: /workspace/results"

python3 benchmark_casp14_foldforever_vs_baselines.py \
    --mode full \
    --gpu \
    --sequences 10 \
    --output /workspace/results \
    --timeout 600 \
    --verbose

echo "✅ Benchmark completed!"
echo "📁 Results location: /workspace/results/"
ls -la /workspace/results/
'

echo "📤 Executing benchmark on RunPod..."
echo "🌐 RunPod URL: https://5ocnemvgivdwzq-8888.proxy.runpod.net"

# For now, display the command to run manually
echo ""
echo "🔧 MANUAL EXECUTION REQUIRED:"
echo "Copy and paste this command in your RunPod web terminal:"
echo ""
echo "----------------------------------------"
echo "$RUNPOD_COMMAND"
echo "----------------------------------------"
echo ""
echo "⏱️  Expected runtime: 10-30 minutes"
echo "💾 Results will be saved to /workspace/results/"
echo ""
echo "📊 Expected outputs:"
echo "   ✅ TM-score, RMSD, GDT-TS, lDDT metrics"
echo "   ✅ Runtime and GPU memory measurements"
echo "   ✅ Comprehensive report and visualizations"
echo "   ✅ PDB structure files"
