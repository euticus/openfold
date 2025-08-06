#!/usr/bin/env python3
"""
🚀 Execute Production Benchmark on RunPod
Runs the comprehensive benchmark using real weights and CASP data
"""

import subprocess
import time
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def execute_runpod_command(command: str, timeout: int = 600) -> dict:
    """Execute command on RunPod via curl."""
    
    # For now, let's create a script that can be uploaded and run
    # This is a simplified approach - in production you'd use proper SSH/API
    
    logger.info(f"📤 Preparing to execute: {command}")
    
    # Create a script to run on RunPod
    runpod_script = f'''#!/bin/bash
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
{command}

echo "✅ Benchmark completed!"
'''
    
    # Save the script
    script_path = Path("runpod_execute.sh")
    with open(script_path, 'w') as f:
        f.write(runpod_script)
    
    logger.info(f"💾 Created execution script: {script_path}")
    
    return {
        "status": "script_created",
        "script_path": str(script_path),
        "command": command
    }

def run_production_benchmark():
    """Run the production benchmark."""
    
    logger.info("🚀 Starting Production Benchmark Execution")
    
    # The benchmark command to run on RunPod
    benchmark_cmd = """python3 benchmark_casp14_foldforever_vs_baselines.py \\
    --mode full \\
    --gpu \\
    --sequences 10 \\
    --output /workspace/results \\
    --timeout 600 \\
    --verbose"""
    
    # Execute the command
    result = execute_runpod_command(benchmark_cmd)
    
    logger.info(f"📋 Execution prepared: {result}")
    
    # Instructions for manual execution
    instructions = f"""
🚀 PRODUCTION BENCHMARK EXECUTION INSTRUCTIONS

1. **Upload the benchmark script to RunPod:**
   In your RunPod web terminal, run:
   ```bash
   cd /workspace
   
   # Kill any existing servers
   pkill -f python
   pkill -f uvicorn
   
   # Check environment
   nvidia-smi
   ls -la openfold/resources/openfold_params/*.pt | head -5
   ls -la openfold/demo_dataset/fasta/
   
   # Run the production benchmark
   cd openfold
   python3 benchmark_casp14_foldforever_vs_baselines.py \\
       --mode full \\
       --gpu \\
       --sequences 10 \\
       --output /workspace/results \\
       --timeout 600 \\
       --verbose
   ```

2. **Expected Output:**
   - Real OpenFold model loading with actual weights
   - ESMFold model initialization
   - CASP14 target processing (T1024, T1026, T1027, H1025)
   - TM-score, RMSD, GDT-TS calculations
   - Performance metrics (runtime, GPU memory)
   - Results saved to /workspace/results/

3. **Results Location:**
   - `/workspace/results/benchmark_report.csv`
   - `/workspace/results/benchmark_report.md`
   - `/workspace/results/plots/`
   - `/workspace/results/structures/`

4. **Download Results:**
   After completion, download the results:
   ```bash
   # In RunPod terminal
   cd /workspace/results
   tar -czf production_benchmark_results.tar.gz *
   
   # Then download via RunPod file manager or:
   # Use the RunPod API to download the results
   ```

🎯 **This will run the REAL production benchmark with:**
   ✅ Actual OpenFold weights (openfold_model_1_ptm.pt + others)
   ✅ Real ESMFold from HuggingFace
   ✅ Actual CASP14 targets with reference structures
   ✅ Proper structural metrics (TM-score, RMSD, GDT-TS)
   ✅ GPU performance measurements on A100/H100
   ✅ Comprehensive results analysis and visualization
"""
    
    print(instructions)
    
    # Also save instructions to file
    with open("production_benchmark_instructions.md", 'w') as f:
        f.write(instructions)
    
    logger.info("📄 Instructions saved to production_benchmark_instructions.md")
    
    return {
        "status": "instructions_generated",
        "instructions_file": "production_benchmark_instructions.md",
        "script_file": result.get("script_path")
    }

def check_runpod_status():
    """Check RunPod server status."""
    try:
        import requests
        response = requests.get("https://5ocnemvgivdwzq-8888.proxy.runpod.net/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"✅ RunPod server healthy: {health_data}")
            return True
        else:
            logger.error(f"❌ RunPod server unhealthy: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ RunPod connection failed: {e}")
        return False

def main():
    """Main execution."""
    print("🚀 Production Benchmark Executor")
    print("=" * 50)
    
    # Check RunPod status
    if check_runpod_status():
        print("✅ RunPod server is accessible")
    else:
        print("⚠️  RunPod server status unknown - proceeding with instructions")
    
    # Generate execution plan
    result = run_production_benchmark()
    
    print(f"\n🎉 Execution plan ready!")
    print(f"📄 See: {result['instructions_file']}")
    
    # Provide direct command for copy-paste
    print("\n" + "="*60)
    print("🚀 DIRECT COMMAND FOR RUNPOD TERMINAL:")
    print("="*60)
    print("""
cd /workspace/openfold && python3 benchmark_casp14_foldforever_vs_baselines.py --mode full --gpu --sequences 10 --output /workspace/results --timeout 600 --verbose
""")
    print("="*60)

if __name__ == "__main__":
    main()
