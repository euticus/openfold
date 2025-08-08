#!/usr/bin/env python3
"""
RunPod A100 Setup Script for FoldForever Benchmark
Run this first in your RunPod Jupyter notebook
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """Run a command and print output"""
    print(f"🔄 {description}")
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success: {description}")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ Error: {description}")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    print("🚀 FoldForever A100 Benchmark Setup")
    print("=" * 50)
    
    # Check GPU
    print("\n🔍 GPU Information:")
    run_command("nvidia-smi", "Checking GPU status")
    
    # Install requirements
    print("\n📦 Installing Python packages...")
    run_command("pip install --upgrade pip", "Upgrading pip")
    
    # Install packages one by one for better error handling
    packages = [
        "torch torchvision torchaudio",
        "transformers>=4.21.0",
        "accelerate",
        "matplotlib seaborn",
        "pandas numpy scipy",
        "psutil biopython",
        "requests tqdm",
        "fair-esm",
        "biotite",
        "py3Dmol",
        "plotly kaleido"
    ]
    
    for package in packages:
        run_command(f"pip install {package}", f"Installing {package}")
    
    # Verify PyTorch GPU
    print("\n🔥 Verifying PyTorch GPU access...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("❌ PyTorch not installed properly")
    
    # Create workspace
    workspace = "/workspace/foldforever_benchmark"
    os.makedirs(workspace, exist_ok=True)
    os.chdir(workspace)
    print(f"\n📁 Created workspace: {workspace}")
    
    print("\n✅ Setup complete!")
    print("\n🎯 Next steps:")
    print("1. Upload benchmark_casp14_foldforever_vs_baselines.py to this directory")
    print("2. Run: python benchmark_casp14_foldforever_vs_baselines.py --mode full --gpu --sequences 30")
    print(f"3. Results will be saved in: {workspace}/results")

if __name__ == "__main__":
    main()
