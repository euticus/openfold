#!/usr/bin/env python3
"""
RunPod OdinFold Benchmark Client
Optimized for calling RunPod GPU instances from local machine
"""

import json
import time
import requests
import argparse
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RunPodBenchmarkClient:
    """Client optimized for RunPod GPU instances."""
    
    def __init__(self, runpod_urls: List[str]):
        self.runpod_urls = runpod_urls
        self.session = requests.Session()
        self.session.timeout = 60  # Longer timeout for RunPod
    
    def check_runpod_health(self, runpod_url: str) -> Dict:
        """Check RunPod instance health."""
        try:
            response = self.session.get(f"{runpod_url}/health")
            response.raise_for_status()
            health_data = response.json()
            
            # Log RunPod-specific info
            if health_data.get("gpu_available"):
                logger.info(f"🔥 RunPod {runpod_url}: {health_data.get('gpu_name', 'Unknown GPU')}")
                gpu_memory_gb = health_data.get('gpu_memory_total', 0) / 1e9
                logger.info(f"   💾 GPU Memory: {gpu_memory_gb:.1f}GB")
            
            return health_data
        except Exception as e:
            logger.error(f"❌ RunPod {runpod_url} unhealthy: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def run_comprehensive_benchmark(self, 
                                   models: List[str] = None,
                                   include_casp: bool = True) -> Dict:
        """Run comprehensive benchmark across all RunPod instances."""
        
        if models is None:
            # Use ALL available model weights
            models = [
                "openfold_model_1_ptm",
                "openfold_finetuning_ptm_1", 
                "openfold_finetuning_ptm_2",
                "openfold_finetuning_no_templ_ptm_1",
                "openfold_finetuning_2",
                "openfold_finetuning_3",
                "openfold_finetuning_4",
                "openfold_finetuning_5"
            ]
        
        logger.info(f"🚀 Starting comprehensive RunPod benchmark")
        logger.info(f"🧬 Models to test: {len(models)}")
        logger.info(f"🖥️  RunPod instances: {len(self.runpod_urls)}")
        
        # Check all RunPod instances
        healthy_pods = []
        for runpod_url in self.runpod_urls:
            health = self.check_runpod_health(runpod_url)
            if health.get("status") == "healthy":
                healthy_pods.append(runpod_url)
            else:
                logger.warning(f"⚠️  RunPod {runpod_url} unhealthy, skipping")
        
        if not healthy_pods:
            raise Exception("No healthy RunPod instances available")
        
        logger.info(f"✅ {len(healthy_pods)} healthy RunPod instances found")
        
        # Get CASP sequences for benchmarking
        if include_casp:
            sequences = self.get_casp_sequences()
        else:
            sequences = [
                "MKWVTFISLLFLFSSAYS",  # Short test
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"  # Medium test
            ]
        
        logger.info(f"🧪 Testing {len(sequences)} sequences")
        
        # Distribute work across RunPod instances
        all_results = []
        jobs = []
        
        # Create jobs for each model on each RunPod
        for i, runpod_url in enumerate(healthy_pods):
            # Distribute models across pods
            models_per_pod = len(models) // len(healthy_pods)
            start_idx = i * models_per_pod
            if i == len(healthy_pods) - 1:  # Last pod gets remaining models
                pod_models = models[start_idx:]
            else:
                pod_models = models[start_idx:start_idx + models_per_pod]
            
            if pod_models:
                job_id = f"runpod_comprehensive_{int(time.time())}_{i}"
                logger.info(f"📋 Starting job {job_id} on RunPod {runpod_url}")
                logger.info(f"   🧬 Models: {pod_models}")
                
                job_status = self.start_benchmark(runpod_url, sequences, pod_models, job_id)
                jobs.append({
                    "runpod_url": runpod_url,
                    "job_id": job_id,
                    "models": pod_models,
                    "sequences_count": len(sequences)
                })
        
        # Wait for all jobs to complete
        for job in jobs:
            logger.info(f"⏳ Waiting for RunPod job {job['job_id']}")
            results = self.wait_for_completion(job["runpod_url"], job["job_id"], timeout=1800)  # 30 min timeout
            
            if results.get("status") != "timeout":
                all_results.extend(results.get("results", []))
        
        # Generate comprehensive analysis
        analysis = self.analyze_comprehensive_results(all_results, models)
        
        combined_results = {
            "benchmark_type": "comprehensive_runpod",
            "total_models": len(models),
            "total_sequences": len(sequences),
            "total_runpods": len(healthy_pods),
            "total_experiments": len(all_results),
            "results": all_results,
            "analysis": analysis,
            "timestamp": time.time()
        }
        
        logger.info(f"🎯 Comprehensive RunPod benchmark completed!")
        logger.info(f"📊 Total experiments: {len(all_results)}")
        
        return combined_results
    
    def get_casp_sequences(self) -> List[str]:
        """Get CASP14 sequences for benchmarking."""
        casp_sequences = [
            # CASP14 targets with known sequences
            "MKLLVLGLGAGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS",  # T1024
            "MAAAAAAGAGPEMVRGQVFDVGPRYTNLSYIGEGAYGMVCSAYDNVNKVRVAIKKISPFEHQTYCQRTLREIKILLRFRHENIIGINDIIRAPTIEQMKDVYIVQDLMETDLYKLLKTQHLSNDHICYFLYQILRGLKYIHSANVLHRDLKPSNLLLNTTCDLKICDFGLARVADPDHDHTGFLTEYVATRWYRAPEIMLNSKGYTKSIDIWSVGCILAEMLSNRPIFPGKHYLDQLNHILGILGSPSQEDLNCIINLKARNYLLSLPHKNKVPWNRLFPNADSKALDLLDKMLTFNPHKRIEVEQALAHPYLEQYYDPSDEPIAEAPFKFDMELDDLPKEKLKELIFEETARFQPGYRS",  # T1027
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # T1030
            "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS",  # T1040
        ]
        return casp_sequences
    
    def analyze_comprehensive_results(self, results: List[Dict], models: List[str]) -> Dict:
        """Analyze comprehensive benchmark results."""
        if not results:
            return {}
        
        successful = [r for r in results if r.get("status") == "success"]
        
        # Model performance comparison
        model_performance = {}
        for model in models:
            model_results = [r for r in successful if r.get("model") == model]
            if model_results:
                avg_runtime = sum(r.get("runtime_seconds", 0) for r in model_results) / len(model_results)
                avg_memory = sum(r.get("gpu_memory_mb", 0) for r in model_results) / len(model_results)
                
                model_performance[model] = {
                    "experiments": len(model_results),
                    "avg_runtime_seconds": avg_runtime,
                    "avg_gpu_memory_mb": avg_memory,
                    "success_rate": len(model_results) / len([r for r in results if r.get("model") == model])
                }
        
        # Sequence length analysis
        length_analysis = {}
        for result in successful:
            length = result.get("sequence_length", 0)
            length_bin = f"{(length // 50) * 50}-{(length // 50 + 1) * 50}"
            if length_bin not in length_analysis:
                length_analysis[length_bin] = []
            length_analysis[length_bin].append(result.get("runtime_seconds", 0))
        
        # Overall statistics
        total_runtime = sum(r.get("runtime_seconds", 0) for r in successful)
        total_memory = sum(r.get("gpu_memory_mb", 0) for r in successful)
        
        return {
            "total_experiments": len(results),
            "successful_experiments": len(successful),
            "overall_success_rate": len(successful) / len(results) if results else 0,
            "total_runtime_seconds": total_runtime,
            "average_runtime_seconds": total_runtime / len(successful) if successful else 0,
            "average_gpu_memory_mb": total_memory / len(successful) if successful else 0,
            "model_performance": model_performance,
            "length_analysis": {k: sum(v)/len(v) for k, v in length_analysis.items()},
            "throughput_sequences_per_second": len(successful) / total_runtime if total_runtime > 0 else 0
        }
    
    def start_benchmark(self, runpod_url: str, sequences: List[str], 
                       models: List[str], job_id: str = None) -> Dict:
        """Start benchmark on RunPod instance."""
        payload = {
            "sequences": sequences,
            "models": models,
            "job_id": job_id
        }
        
        try:
            response = self.session.post(f"{runpod_url}/benchmark", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Failed to start benchmark on {runpod_url}: {e}")
            raise
    
    def wait_for_completion(self, runpod_url: str, job_id: str, 
                           poll_interval: int = 10, timeout: int = 1800) -> Dict:
        """Wait for RunPod job completion."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{runpod_url}/benchmark/{job_id}")
                response.raise_for_status()
                status = response.json()
                
                logger.info(f"📊 RunPod job {job_id}: {status['status']} ({status.get('progress', 0):.1f}%)")
                
                if status["status"] == "completed":
                    logger.info(f"✅ RunPod job {job_id} completed!")
                    # Get detailed results
                    response = self.session.get(f"{runpod_url}/benchmark/{job_id}/results")
                    response.raise_for_status()
                    return response.json()
                elif status["status"] == "failed":
                    logger.error(f"❌ RunPod job {job_id} failed: {status.get('error', 'Unknown error')}")
                    return status
                
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.warning(f"⚠️  Error checking job status: {e}")
                time.sleep(poll_interval)
        
        logger.error(f"⏰ RunPod job {job_id} timed out after {timeout}s")
        return {"status": "timeout"}

def main():
    parser = argparse.ArgumentParser(description="RunPod OdinFold Benchmark Client")
    parser.add_argument("--runpods", nargs="+", required=True,
                       help="RunPod URLs (e.g., https://abc123-8000.proxy.runpod.net)")
    parser.add_argument("--models", nargs="+", 
                       help="Specific models to test (default: all available)")
    parser.add_argument("--include-casp", action="store_true", default=True,
                       help="Include CASP14 sequences")
    parser.add_argument("--output", default="runpod_comprehensive_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Create client
    client = RunPodBenchmarkClient(args.runpods)
    
    try:
        # Run comprehensive benchmark
        results = client.run_comprehensive_benchmark(
            models=args.models,
            include_casp=args.include_casp
        )
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        analysis = results["analysis"]
        print("\n🎯 COMPREHENSIVE RUNPOD BENCHMARK RESULTS")
        print("=" * 50)
        print(f"🧬 Total models tested: {results['total_models']}")
        print(f"🧪 Total sequences: {results['total_sequences']}")
        print(f"🖥️  RunPod instances: {results['total_runpods']}")
        print(f"🔬 Total experiments: {results['total_experiments']}")
        print(f"✅ Success rate: {analysis.get('overall_success_rate', 0):.1%}")
        print(f"⏱️  Total runtime: {analysis.get('total_runtime_seconds', 0):.1f}s")
        print(f"🚀 Throughput: {analysis.get('throughput_sequences_per_second', 0):.2f} seq/s")
        print(f"💾 Results saved to: {args.output}")
        
        # Model performance summary
        print(f"\n📊 MODEL PERFORMANCE SUMMARY:")
        for model, perf in analysis.get('model_performance', {}).items():
            print(f"   {model}: {perf['avg_runtime_seconds']:.2f}s avg, {perf['success_rate']:.1%} success")
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
