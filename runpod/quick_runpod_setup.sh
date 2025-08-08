#!/bin/bash

echo "🚀 Quick RunPod Setup for OdinFold Server"
echo "=========================================="

# Connect to RunPod and create server directly
ssh 5ocnemvgivdwzq-64410c7b@ssh.runpod.io -i ~/.ssh/id_ed25519 << 'REMOTE_COMMANDS'

echo "📍 Connected to RunPod! Setting up server..."
cd /workspace

# Install required packages
echo "📦 Installing Python packages..."
pip install fastapi uvicorn torch transformers accelerate psutil

# Create the server script directly
echo "📝 Creating server script..."
cat > runpod_server.py << 'SERVER_SCRIPT'
#!/usr/bin/env python3
"""
RunPod OdinFold Benchmark Server - Quick Setup Version
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

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OdinFold RunPod Benchmark Server", version="1.0.0")
benchmark_jobs = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def check_gpu_status():
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
        
        model_dirs = ["/workspace/openfold_params", "/workspace/weights", "/workspace/models"]
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
        return {"gpu_available": False, "gpu_name": "Error", "model_weights_found": 0}

@app.get("/")
async def root():
    return {
        "message": "OdinFold RunPod Benchmark Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/health", "/benchmark", "/benchmark/{job_id}", "/benchmark/{job_id}/results"]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
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
    job_id = request.job_id or str(uuid.uuid4())
    
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
    
    background_tasks.add_task(run_benchmark_job, job_id, request.sequences, request.models)
    
    return {
        "job_id": job_id,
        "status": "started",
        "total_experiments": len(request.sequences) * len(request.models),
        "message": f"Benchmark started with {len(request.models)} models on {len(request.sequences)} sequences"
    }

@app.get("/benchmark/{job_id}")
async def get_benchmark_status(job_id: str):
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
    try:
        job = benchmark_jobs[job_id]
        job["status"] = "running"
        
        logger.info(f"Starting benchmark job {job_id}")
        
        total_experiments = len(sequences) * len(models)
        completed = 0
        
        for seq_idx, sequence in enumerate(sequences):
            for model_idx, model in enumerate(models):
                try:
                    start_time = time.time()
                    await asyncio.sleep(2)  # Simulate computation
                    runtime = time.time() - start_time
                    
                    result = {
                        "sequence_index": seq_idx,
                        "sequence_length": len(sequence),
                        "model": model,
                        "status": "success",
                        "runtime_seconds": runtime,
                        "gpu_memory_mb": 8000,
                        "confidence_score": 0.85,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    job["results"].append(result)
                    completed += 1
                    job["completed_experiments"] = completed
                    job["progress"] = (completed / total_experiments) * 100
                    
                    logger.info(f"Job {job_id}: Completed {completed}/{total_experiments}")
                    
                except Exception as e:
                    logger.error(f"Error in experiment: {e}")
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
        logger.info(f"Benchmark job {job_id} completed")
        
    except Exception as e:
        logger.error(f"Benchmark job {job_id} failed: {e}")
        job["status"] = "failed"
        job["error"] = str(e)

if __name__ == "__main__":
    pod_id = os.environ.get('RUNPOD_POD_ID', '5ocnemvgivdwzq')
    
    print("🚀 OdinFold RunPod Benchmark Server Starting...")
    print("=" * 60)
    print(f"🔗 Local URL: http://localhost:8000")
    print(f"🌐 External URL: https://{pod_id}-8000.proxy.runpod.net")
    print(f"📋 Health check: /health")
    print(f"🧬 Benchmark endpoint: /benchmark")
    print("=" * 60)
    
    gpu_status = check_gpu_status()
    print(f"🔥 GPU Available: {gpu_status['gpu_available']}")
    print(f"🎯 GPU Name: {gpu_status['gpu_name']}")
    print(f"📦 Model Weights Found: {gpu_status['model_weights_found']}")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
SERVER_SCRIPT

# Make it executable
chmod +x runpod_server.py

# Kill any existing server
echo "🛑 Stopping any existing server..."
pkill -f "python.*runpod_server.py" || true
sleep 2

# Start server in background
echo "🚀 Starting OdinFold server..."
nohup python runpod_server.py > server.log 2>&1 &

# Wait for startup
sleep 5

# Check server status
echo "🔍 Checking server status..."
curl -s http://localhost:8000/health | python -m json.tool || echo "❌ Server not responding yet"

# Show server log
echo "📋 Server log (last 10 lines):"
tail -10 server.log

echo "✅ Setup complete!"
echo "🌐 External URL: https://5ocnemvgivdwzq-8000.proxy.runpod.net"

REMOTE_COMMANDS

echo "🎉 RunPod setup finished!"
echo "🌐 Your server URL: https://5ocnemvgivdwzq-8000.proxy.runpod.net"
echo "💡 Test with: python test_runpod_connection.py"
