#!/bin/bash
cd /workspace/openfold

echo "🔍 Checking environment..."
nvidia-smi
echo "📊 GPU Memory:"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv

echo "🔍 Checking model weights..."
ls -la resources/openfold_params/*.pt | head -5

echo "🔍 Checking CASP data..."
ls -la demo_dataset/fasta/
ls -la demo_dataset/pdb/

echo "🚀 Starting benchmark..."
python3 benchmark_casp14_foldforever_vs_baselines.py \
    --mode full \
    --gpu \
    --sequences 10 \
    --output /workspace/results \
    --timeout 600 \
    --verbose

echo "✅ Benchmark completed!"
