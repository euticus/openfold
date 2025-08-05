#!/usr/bin/env python3
"""
Deploy OdinFold Benchmark Server
Run this on your GPU servers, call from local machine
"""

import os
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import benchmark runner
from production_benchmark_setup import ProductionBenchmarkRunner

app = FastAPI(title="OdinFold Benchmark Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global benchmark runner
benchmark_runner = None
active_jobs = {}

class BenchmarkRequest(BaseModel):
    """Benchmark request model."""
    sequences: List[str]
    models: List[str] = ["odinfold"]
    job_id: str = None
    config_overrides: Dict = {}

class BenchmarkStatus(BaseModel):
    """Benchmark status model."""
    job_id: str
    status: str  # "queued", "running", "completed", "failed"
    progress: float = 0.0
    results: List[Dict] = []
    error: str = None
    started_at: float = None
    completed_at: float = None

@app.on_event("startup")
async def startup_event():
    """Initialize benchmark runner on startup."""
    global benchmark_runner
    logger.info("🚀 Starting OdinFold Benchmark Server")
    
    try:
        benchmark_runner = ProductionBenchmarkRunner()
        logger.info("✅ Benchmark runner initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize benchmark runner: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "OdinFold Benchmark Server",
        "version": "1.0.0",
        "status": "running",
        "gpu_available": torch.cuda.is_available() if 'torch' in globals() else False
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    import torch
    
    health_info = {
        "status": "healthy",
        "timestamp": time.time(),
        "gpu_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        health_info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
            "gpu_memory_allocated": torch.cuda.memory_allocated(),
        })
    
    return health_info

@app.post("/benchmark", response_model=BenchmarkStatus)
async def start_benchmark(request: BenchmarkRequest, background_tasks: BackgroundTasks):
    """Start a benchmark job."""
    job_id = request.job_id or f"job_{int(time.time() * 1000)}"
    
    if job_id in active_jobs:
        raise HTTPException(status_code=400, detail=f"Job {job_id} already exists")
    
    # Initialize job status
    job_status = BenchmarkStatus(
        job_id=job_id,
        status="queued",
        started_at=time.time()
    )
    active_jobs[job_id] = job_status
    
    # Start benchmark in background
    background_tasks.add_task(run_benchmark_job, job_id, request)
    
    logger.info(f"📋 Started benchmark job {job_id}")
    return job_status

@app.get("/benchmark/{job_id}", response_model=BenchmarkStatus)
async def get_benchmark_status(job_id: str):
    """Get benchmark job status."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return active_jobs[job_id]

@app.get("/benchmark/{job_id}/results")
async def get_benchmark_results(job_id: str):
    """Get detailed benchmark results."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job_status = active_jobs[job_id]
    if job_status.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job_id} not completed")
    
    return {
        "job_id": job_id,
        "results": job_status.results,
        "summary": generate_results_summary(job_status.results)
    }

@app.get("/jobs")
async def list_jobs():
    """List all benchmark jobs."""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job.status,
                "progress": job.progress,
                "started_at": job.started_at,
                "completed_at": job.completed_at
            }
            for job_id, job in active_jobs.items()
        ]
    }

async def run_benchmark_job(job_id: str, request: BenchmarkRequest):
    """Run benchmark job in background."""
    job_status = active_jobs[job_id]
    
    try:
        job_status.status = "running"
        logger.info(f"🔬 Running benchmark job {job_id}")
        
        # Update config with overrides
        if request.config_overrides:
            benchmark_runner.config.update(request.config_overrides)
        
        # Run benchmark for each sequence
        results = []
        total_sequences = len(request.sequences)
        
        for i, sequence in enumerate(request.sequences):
            logger.info(f"Processing sequence {i+1}/{total_sequences}")
            
            # Update progress
            job_status.progress = (i / total_sequences) * 100
            
            # Benchmark sequence with each model
            for model_name in request.models:
                if model_name == "odinfold":
                    model = benchmark_runner.load_odinfold_model()
                    result = benchmark_runner.benchmark_single_sequence(
                        model_name, model, sequence, i
                    )
                    results.append(result)
        
        # Job completed successfully
        job_status.status = "completed"
        job_status.progress = 100.0
        job_status.results = results
        job_status.completed_at = time.time()
        
        logger.info(f"✅ Completed benchmark job {job_id}")
        
    except Exception as e:
        logger.error(f"❌ Benchmark job {job_id} failed: {e}")
        job_status.status = "failed"
        job_status.error = str(e)
        job_status.completed_at = time.time()

def generate_results_summary(results: List[Dict]) -> Dict:
    """Generate summary statistics from results."""
    if not results:
        return {}
    
    # Calculate averages
    total_runtime = sum(r.get("runtime_seconds", 0) for r in results)
    total_memory = sum(r.get("gpu_memory_mb", 0) for r in results)
    successful_runs = len([r for r in results if r.get("status") == "success"])
    
    return {
        "total_sequences": len(results),
        "successful_runs": successful_runs,
        "success_rate": successful_runs / len(results) if results else 0,
        "average_runtime_seconds": total_runtime / len(results) if results else 0,
        "average_gpu_memory_mb": total_memory / len(results) if results else 0,
        "total_runtime_seconds": total_runtime,
    }

# CLI for running the server
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OdinFold Benchmark Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    logger.info(f"🌐 Starting server on {args.host}:{args.port}")
    
    uvicorn.run(
        "deploy_benchmark_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False
    )
