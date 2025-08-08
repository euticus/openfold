#!/usr/bin/env python3
"""
Test RunPod Connection and Run Comprehensive Benchmark
"""

import requests
import json
import time

# Server URL - Update this based on your setup
# For local testing: "http://localhost:8000"
# For RunPod: "https://[POD_ID]-8000.proxy.runpod.net"
RUNPOD_URL = "https://5ocnemvgivdwzq-8000.proxy.runpod.net"  # Your RunPod URL

def test_connection():
    """Test basic connection to RunPod."""
    print(f"🔍 Testing connection to {RUNPOD_URL}")
    
    try:
        # Test basic endpoint
        response = requests.get(f"{RUNPOD_URL}/", timeout=30)
        print(f"✅ Basic connection: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Test health endpoint
        response = requests.get(f"{RUNPOD_URL}/health", timeout=30)
        print(f"✅ Health check: {response.status_code}")
        health_data = response.json()
        print(f"   GPU: {health_data.get('gpu_name', 'Unknown')}")
        print(f"   GPU Available: {health_data.get('gpu_available', False)}")
        print(f"   Model Weights: {health_data.get('model_weights_found', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def run_comprehensive_benchmark():
    """Run comprehensive benchmark with ALL model weights."""
    print(f"\n🚀 Starting Comprehensive Benchmark")
    print("=" * 50)
    
    # CASP14 test sequences
    test_sequences = [
        "MKLLVLGLGAGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS",  # T1024
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # T1030 (shorter)
        "MKWVTFISLLFLFSSAYS"  # Very short test
    ]
    
    # ALL available model weights (based on your directory listing)
    all_models = [
        "openfold_model_1_ptm",
        "finetuning_ptm_1", 
        "finetuning_ptm_2",
        "finetuning_no_templ_ptm_1",
        "finetuning_no_templ_1",
        "finetuning_no_templ_2", 
        "finetuning_2",
        "finetuning_3",
        "finetuning_4",
        "finetuning_5",
        "initial_training",
        "openfold_trained_weights"
    ]
    
    print(f"🧬 Testing {len(all_models)} models on {len(test_sequences)} sequences")
    print(f"🔬 Total experiments: {len(all_models) * len(test_sequences)}")
    
    # Start benchmark
    benchmark_request = {
        "sequences": test_sequences,
        "models": all_models,
        "job_id": f"comprehensive_{int(time.time())}"
    }
    
    try:
        print(f"\n📋 Starting benchmark job...")
        response = requests.post(
            f"{RUNPOD_URL}/benchmark", 
            json=benchmark_request,
            timeout=60
        )
        response.raise_for_status()
        
        job_info = response.json()
        job_id = job_info["job_id"]
        print(f"✅ Job started: {job_id}")
        
        # Monitor progress
        print(f"\n⏳ Monitoring progress...")
        start_time = time.time()
        
        while True:
            try:
                status_response = requests.get(f"{RUNPOD_URL}/benchmark/{job_id}", timeout=30)
                status_response.raise_for_status()
                status = status_response.json()
                
                elapsed = time.time() - start_time
                print(f"📊 [{elapsed:.0f}s] Status: {status['status']} - Progress: {status.get('progress', 0):.1f}%")
                
                if status["status"] == "completed":
                    print(f"🎉 Benchmark completed!")
                    
                    # Get results
                    results_response = requests.get(f"{RUNPOD_URL}/benchmark/{job_id}/results", timeout=30)
                    results_response.raise_for_status()
                    results = results_response.json()
                    
                    # Analyze results
                    analyze_results(results["results"])
                    
                    # Save results
                    with open("comprehensive_runpod_results.json", "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"💾 Results saved to: comprehensive_runpod_results.json")
                    
                    break
                    
                elif status["status"] == "failed":
                    print(f"❌ Benchmark failed: {status.get('error', 'Unknown error')}")
                    break
                    
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"⚠️  Error checking status: {e}")
                time.sleep(10)
        
    except Exception as e:
        print(f"❌ Benchmark failed to start: {e}")

def analyze_results(results):
    """Analyze comprehensive benchmark results."""
    print(f"\n📊 COMPREHENSIVE BENCHMARK RESULTS")
    print("=" * 50)
    
    successful = [r for r in results if r.get("status") == "success"]
    total = len(results)
    
    print(f"🔬 Total experiments: {total}")
    print(f"✅ Successful: {len(successful)}")
    print(f"📈 Success rate: {len(successful)/total*100:.1f}%")
    
    if successful:
        # Model performance
        model_stats = {}
        for result in successful:
            model = result["model"]
            if model not in model_stats:
                model_stats[model] = []
            model_stats[model].append(result["runtime_seconds"])
        
        print(f"\n🧬 MODEL PERFORMANCE:")
        for model, runtimes in model_stats.items():
            avg_runtime = sum(runtimes) / len(runtimes)
            print(f"   {model}: {avg_runtime:.2f}s avg ({len(runtimes)} runs)")
        
        # Overall stats
        total_runtime = sum(r["runtime_seconds"] for r in successful)
        avg_runtime = total_runtime / len(successful)
        avg_memory = sum(r.get("gpu_memory_mb", 0) for r in successful) / len(successful)
        
        print(f"\n⚡ OVERALL PERFORMANCE:")
        print(f"   Total runtime: {total_runtime:.1f}s")
        print(f"   Average per experiment: {avg_runtime:.2f}s")
        print(f"   Average GPU memory: {avg_memory:.1f}MB")
        print(f"   Throughput: {len(successful)/total_runtime:.2f} experiments/second")

if __name__ == "__main__":
    print("🚀 RunPod OdinFold Comprehensive Benchmark Test")
    print("=" * 60)
    
    # Test connection first
    if test_connection():
        print(f"\n✅ Connection successful!")
        
        # Ask user if they want to run full benchmark
        response = input(f"\n🤔 Run comprehensive benchmark with ALL 12 models? (y/n): ")
        if response.lower() in ['y', 'yes']:
            run_comprehensive_benchmark()
        else:
            print(f"👍 Connection test complete. Run again with 'y' to start benchmark.")
    else:
        print(f"\n❌ Connection failed. Check RunPod status.")
        print(f"💡 Try accessing: {RUNPOD_URL}/health in your browser")
