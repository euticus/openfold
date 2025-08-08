#!/usr/bin/env python3
"""
RunPod OdinFold Benchmark Server
Optimized for A100 GPU with all model weights
"""

import os
import sys
import json
import time
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
import uuid

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add current directory to path for imports
sys.path.append('/workspace')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
app = FastAPI(title="OdinFold RunPod Benchmark Server", version="1.0.0")
benchmark_jobs = {}  # Store running jobs
model_cache = {}     # Cache loaded models

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class BenchmarkRequest(BaseModel):
    sequences: List[str]
    models: List[str]
    job_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: str
    model_weights_found: int
    timestamp: str

# Check GPU and model availability
def check_gpu_status():
    """Check GPU availability and model weights."""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
        
        # Check for model weights
        model_dirs = [
            "/workspace/openfold_params",
            "/workspace/weights", 
            "/workspace/models",
            "/root/openfold_params"
        ]
        
        model_count = 0
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                model_count += len([f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))])
        
        return {
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
            "model_weights_found": model_count
        }
    except Exception as e:
        logger.error(f"Error checking GPU status: {e}")
        return {
            "gpu_available": False,
            "gpu_name": "Error",
            "model_weights_found": 0
        }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "OdinFold RunPod Benchmark Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/health", "/benchmark", "/benchmark/{job_id}", "/benchmark/{job_id}/results"]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_status = check_gpu_status()
    
    return HealthResponse(
        status="healthy",
        gpu_available=gpu_status["gpu_available"],
        gpu_name=gpu_status["gpu_name"],
        model_weights_found=gpu_status["model_weights_found"],
        timestamp=datetime.now().isoformat()
    )

@app.post("/benchmark")
async def start_benchmark(request: BenchmarkRequest, background_tasks: BackgroundTasks):
    """Start a comprehensive benchmark job."""
    job_id = request.job_id or str(uuid.uuid4())
    
    # Initialize job
    benchmark_jobs[job_id] = {
        "status": "starting",
        "progress": 0.0,
        "total_experiments": len(request.sequences) * len(request.models),
        "completed_experiments": 0,
        "results": [],
        "start_time": time.time(),
        "sequences": request.sequences,
        "models": request.models
    }
    
    # Start background task
    background_tasks.add_task(run_benchmark_job, job_id, request.sequences, request.models)
    
    return {
        "job_id": job_id,
        "status": "started",
        "total_experiments": len(request.sequences) * len(request.models),
        "message": f"Benchmark started with {len(request.models)} models on {len(request.sequences)} sequences"
    }

@app.get("/benchmark/{job_id}")
async def get_benchmark_status(job_id: str):
    """Get benchmark job status."""
    if job_id not in benchmark_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = benchmark_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "completed_experiments": job["completed_experiments"],
        "total_experiments": job["total_experiments"],
        "runtime_seconds": time.time() - job["start_time"]
    }

@app.get("/benchmark/{job_id}/results")
async def get_benchmark_results(job_id: str):
    """Get benchmark job results."""
    if job_id not in benchmark_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = benchmark_jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "total_runtime": time.time() - job["start_time"],
        "results": job["results"]
    }

async def run_benchmark_job(job_id: str, sequences: List[str], models: List[str]):
    """Run the actual benchmark job."""
    try:
        job = benchmark_jobs[job_id]
        job["status"] = "running"
        
        logger.info(f"Starting benchmark job {job_id} with {len(models)} models and {len(sequences)} sequences")
        
        total_experiments = len(sequences) * len(models)
        completed = 0
        
        for seq_idx, sequence in enumerate(sequences):
            for model_idx, model in enumerate(models):
                try:
                    # Simulate protein folding (replace with actual OpenFold inference)
                    start_time = time.time()
                    
                    # Mock folding result - replace with actual OpenFold call
                    await asyncio.sleep(2)  # Simulate computation time
                    
                    runtime = time.time() - start_time
                    
                    result = {
                        "sequence_index": seq_idx,
                        "sequence_length": len(sequence),
                        "model": model,
                        "status": "success",
                        "runtime_seconds": runtime,
                        "gpu_memory_mb": 8000,  # Mock value
                        "confidence_score": 0.85,  # Mock value
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    job["results"].append(result)
                    completed += 1
                    job["completed_experiments"] = completed
                    job["progress"] = (completed / total_experiments) * 100
                    
                    logger.info(f"Job {job_id}: Completed {completed}/{total_experiments} ({job['progress']:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"Error in experiment {completed+1}: {e}")
                    result = {
                        "sequence_index": seq_idx,
                        "model": model,
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    job["results"].append(result)
                    completed += 1
                    job["completed_experiments"] = completed
                    job["progress"] = (completed / total_experiments) * 100
        
        job["status"] = "completed"
        logger.info(f"Benchmark job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Benchmark job {job_id} failed: {e}")
        job["status"] = "failed"
        job["error"] = str(e)

if __name__ == "__main__":
    # Get RunPod info
    pod_id = os.environ.get('RUNPOD_POD_ID', 'unknown')
    
    print("🚀 OdinFold RunPod Benchmark Server Starting...")
    print("=" * 60)
    print(f"🔗 Local URL: http://localhost:8000")
    print(f"🌐 External URL: https://{pod_id}-8000.proxy.runpod.net")
    print(f"📋 Health check: /health")
    print(f"🧬 Benchmark endpoint: /benchmark")
    print("=" * 60)
    
    # Check GPU status on startup
    gpu_status = check_gpu_status()
    print(f"🔥 GPU Available: {gpu_status['gpu_available']}")
    print(f"🎯 GPU Name: {gpu_status['gpu_name']}")
    print(f"📦 Model Weights Found: {gpu_status['model_weights_found']}")
    print("=" * 60)
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
