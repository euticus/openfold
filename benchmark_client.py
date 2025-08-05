#!/usr/bin/env python3
"""
OdinFold Benchmark Client
Call GPU servers from your local machine
"""

import json
import time
import requests
import argparse
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkClient:
    """Client for calling remote benchmark servers."""
    
    def __init__(self, server_urls: List[str]):
        self.server_urls = server_urls
        self.session = requests.Session()
        self.session.timeout = 30
    
    def check_server_health(self, server_url: str) -> Dict:
        """Check if server is healthy."""
        try:
            response = self.session.get(f"{server_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Server {server_url} unhealthy: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def start_benchmark(self, server_url: str, sequences: List[str], 
                       models: List[str] = ["odinfold"], job_id: str = None) -> Dict:
        """Start benchmark on remote server."""
        payload = {
            "sequences": sequences,
            "models": models,
            "job_id": job_id
        }
        
        try:
            response = self.session.post(f"{server_url}/benchmark", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Failed to start benchmark on {server_url}: {e}")
            raise
    
    def get_job_status(self, server_url: str, job_id: str) -> Dict:
        """Get job status from server."""
        try:
            response = self.session.get(f"{server_url}/benchmark/{job_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Failed to get status for job {job_id}: {e}")
            raise
    
    def get_job_results(self, server_url: str, job_id: str) -> Dict:
        """Get detailed job results."""
        try:
            response = self.session.get(f"{server_url}/benchmark/{job_id}/results")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Failed to get results for job {job_id}: {e}")
            raise
    
    def wait_for_completion(self, server_url: str, job_id: str, 
                           poll_interval: int = 5, timeout: int = 3600) -> Dict:
        """Wait for job completion with progress updates."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_job_status(server_url, job_id)
            
            logger.info(f"📊 Job {job_id}: {status['status']} ({status.get('progress', 0):.1f}%)")
            
            if status["status"] == "completed":
                logger.info(f"✅ Job {job_id} completed successfully!")
                return self.get_job_results(server_url, job_id)
            elif status["status"] == "failed":
                logger.error(f"❌ Job {job_id} failed: {status.get('error', 'Unknown error')}")
                return status
            
            time.sleep(poll_interval)
        
        logger.error(f"⏰ Job {job_id} timed out after {timeout}s")
        return {"status": "timeout"}
    
    def run_distributed_benchmark(self, sequences: List[str], 
                                 models: List[str] = ["odinfold"]) -> Dict:
        """Run benchmark across multiple servers."""
        logger.info(f"🚀 Starting distributed benchmark on {len(self.server_urls)} servers")
        
        # Check server health
        healthy_servers = []
        for server_url in self.server_urls:
            health = self.check_server_health(server_url)
            if health.get("status") == "healthy":
                healthy_servers.append(server_url)
                logger.info(f"✅ Server {server_url}: {health.get('gpu_name', 'Unknown GPU')}")
            else:
                logger.warning(f"⚠️  Server {server_url} unhealthy, skipping")
        
        if not healthy_servers:
            raise Exception("No healthy servers available")
        
        # Distribute sequences across servers
        jobs = []
        sequences_per_server = len(sequences) // len(healthy_servers)
        
        for i, server_url in enumerate(healthy_servers):
            start_idx = i * sequences_per_server
            if i == len(healthy_servers) - 1:  # Last server gets remaining sequences
                server_sequences = sequences[start_idx:]
            else:
                server_sequences = sequences[start_idx:start_idx + sequences_per_server]
            
            if server_sequences:
                job_id = f"distributed_job_{int(time.time())}_{i}"
                logger.info(f"📋 Starting job {job_id} on {server_url} with {len(server_sequences)} sequences")
                
                job_status = self.start_benchmark(server_url, server_sequences, models, job_id)
                jobs.append({
                    "server_url": server_url,
                    "job_id": job_id,
                    "sequences_count": len(server_sequences)
                })
        
        # Wait for all jobs to complete
        all_results = []
        for job in jobs:
            logger.info(f"⏳ Waiting for job {job['job_id']} on {job['server_url']}")
            results = self.wait_for_completion(job["server_url"], job["job_id"])
            
            if results.get("status") != "timeout":
                all_results.extend(results.get("results", []))
        
        # Combine results
        combined_results = {
            "total_sequences": len(sequences),
            "total_servers": len(healthy_servers),
            "successful_jobs": len([r for r in all_results if r.get("status") == "success"]),
            "results": all_results,
            "summary": self.generate_summary(all_results)
        }
        
        logger.info(f"🎯 Distributed benchmark completed!")
        logger.info(f"📊 Processed {len(sequences)} sequences on {len(healthy_servers)} servers")
        
        return combined_results
    
    def generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary from all results."""
        if not results:
            return {}
        
        successful = [r for r in results if r.get("status") == "success"]
        
        if not successful:
            return {"success_rate": 0}
        
        total_runtime = sum(r.get("runtime_seconds", 0) for r in successful)
        total_memory = sum(r.get("gpu_memory_mb", 0) for r in successful)
        
        return {
            "total_results": len(results),
            "successful_results": len(successful),
            "success_rate": len(successful) / len(results),
            "average_runtime_seconds": total_runtime / len(successful),
            "average_gpu_memory_mb": total_memory / len(successful),
            "total_runtime_seconds": total_runtime,
            "sequences_per_second": len(successful) / total_runtime if total_runtime > 0 else 0
        }

def main():
    parser = argparse.ArgumentParser(description="OdinFold Benchmark Client")
    parser.add_argument("--servers", nargs="+", required=True, 
                       help="List of server URLs (e.g., http://gpu1:8000 http://gpu2:8000)")
    parser.add_argument("--sequences", nargs="+", 
                       default=["MKWVTFISLLFLFSSAYS", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
                       help="Protein sequences to benchmark")
    parser.add_argument("--models", nargs="+", default=["odinfold"],
                       help="Models to benchmark")
    parser.add_argument("--output", default="benchmark_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Create client
    client = BenchmarkClient(args.servers)
    
    # Run benchmark
    try:
        results = client.run_distributed_benchmark(args.sequences, args.models)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        summary = results["summary"]
        print("\n🎯 BENCHMARK RESULTS SUMMARY")
        print("=" * 40)
        print(f"📊 Total sequences: {results['total_sequences']}")
        print(f"🖥️  Servers used: {results['total_servers']}")
        print(f"✅ Success rate: {summary.get('success_rate', 0):.1%}")
        print(f"⏱️  Average runtime: {summary.get('average_runtime_seconds', 0):.2f}s")
        print(f"💾 Average GPU memory: {summary.get('average_gpu_memory_mb', 0):.1f}MB")
        print(f"🚀 Throughput: {summary.get('sequences_per_second', 0):.2f} seq/s")
        print(f"💾 Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
