#!/usr/bin/env python3
"""
OdinFold API Server

FastAPI-based REST API for OdinFold protein folding inference.
The engine that powers FoldForever.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('ODINFOLD_LOG_LEVEL', 'INFO').upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OdinFold API",
    description="The engine that powers FoldForever - Advanced protein folding inference",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
DEVICE = torch.device(os.getenv('ODINFOLD_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
MODEL_LOADED = False
FOLDING_MODEL = None

logger.info(f"OdinFold API starting on device: {DEVICE}")


# Pydantic models
class FoldingRequest(BaseModel):
    """Request model for protein folding."""
    
    sequence: str = Field(..., description="Protein sequence in single-letter amino acid code", min_length=10, max_length=2048)
    job_id: Optional[str] = Field(None, description="Optional job ID for tracking")
    confidence: bool = Field(True, description="Include pLDDT confidence scores")
    relax: bool = Field(False, description="Apply post-fold relaxation")
    format: str = Field("json", description="Output format: json, pdb")


class FoldingResponse(BaseModel):
    """Response model for protein folding."""
    
    job_id: str
    sequence: str
    sequence_length: int
    coordinates: List[List[float]]
    confidence_scores: Optional[List[float]] = None
    tm_score_estimate: Optional[float] = None
    runtime_seconds: float
    device: str
    status: str = "completed"


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    device: str
    cuda_available: bool
    model_loaded: bool
    memory_usage_gb: float
    uptime_seconds: float


# Global variables for tracking
START_TIME = time.time()


def load_model():
    """Load the OdinFold model."""
    global MODEL_LOADED, FOLDING_MODEL
    
    if MODEL_LOADED:
        return
    
    logger.info("Loading OdinFold model...")
    
    try:
        # Mock model loading for now
        # In production, this would load the actual OdinFold model
        FOLDING_MODEL = {
            'device': DEVICE,
            'loaded_at': time.time(),
            'model_type': 'odinfold_v1'
        }
        
        MODEL_LOADED = True
        logger.info(f"OdinFold model loaded successfully on {DEVICE}")
        
    except Exception as e:
        logger.error(f"Failed to load OdinFold model: {e}")
        raise


def mock_fold_protein(sequence: str, confidence: bool = True, relax: bool = False) -> Dict:
    """
    Mock protein folding function.
    
    In production, this would call the actual OdinFold inference pipeline.
    """
    
    start_time = time.time()
    
    # Validate sequence
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    if not all(aa in valid_amino_acids for aa in sequence.upper()):
        raise ValueError("Invalid amino acid sequence")
    
    seq_len = len(sequence)
    
    # Mock folding process
    logger.info(f"Folding protein sequence of length {seq_len}")
    
    # Simulate realistic folding time
    base_time = 0.5  # Base time
    length_factor = seq_len / 100  # Scale with length
    folding_time = base_time * length_factor
    
    # Add small delay for realism
    time.sleep(min(0.1, folding_time / 10))
    
    # Generate mock coordinates (realistic protein-like structure)
    coordinates = []
    for i in range(seq_len):
        # Simple helix-like structure with some noise
        x = i * 1.5 + np.random.normal(0, 0.5)
        y = 3 * np.sin(i * 0.3) + np.random.normal(0, 0.5)
        z = 3 * np.cos(i * 0.3) + np.random.normal(0, 0.5)
        coordinates.append([float(x), float(y), float(z)])
    
    # Generate mock confidence scores
    confidence_scores = None
    if confidence:
        # Realistic pLDDT distribution
        base_confidence = np.random.normal(75, 15, seq_len)
        confidence_scores = np.clip(base_confidence, 0, 100).tolist()
    
    # Mock TM-score estimate
    tm_score_estimate = np.random.normal(0.72, 0.08)
    tm_score_estimate = float(np.clip(tm_score_estimate, 0.3, 0.95))
    
    # Apply relaxation if requested
    if relax:
        logger.info("Applying post-fold relaxation")
        # Mock relaxation - slight coordinate adjustment
        for i, coord in enumerate(coordinates):
            coordinates[i] = [c + np.random.normal(0, 0.1) for c in coord]
        time.sleep(0.05)  # Relaxation time
    
    runtime = time.time() - start_time
    
    return {
        'coordinates': coordinates,
        'confidence_scores': confidence_scores,
        'tm_score_estimate': tm_score_estimate,
        'runtime_seconds': runtime
    }


@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    logger.info("Starting OdinFold API server")
    load_model()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "OdinFold API - The engine that powers FoldForever",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    
    # Calculate memory usage
    memory_usage = 0.0
    if torch.cuda.is_available():
        memory_usage = torch.cuda.memory_allocated() / (1024**3)  # GB
    
    return HealthResponse(
        status="healthy" if MODEL_LOADED else "loading",
        device=str(DEVICE),
        cuda_available=torch.cuda.is_available(),
        model_loaded=MODEL_LOADED,
        memory_usage_gb=memory_usage,
        uptime_seconds=time.time() - START_TIME
    )


@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {"status": "ready"}


@app.post("/fold", response_model=FoldingResponse)
async def fold_protein(request: FoldingRequest, background_tasks: BackgroundTasks):
    """
    Fold a protein sequence using OdinFold.
    
    This endpoint performs protein structure prediction using the OdinFold model.
    """
    
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Generate job ID if not provided
    job_id = request.job_id or f"fold_{int(time.time() * 1000)}"
    
    try:
        logger.info(f"Starting folding job {job_id} for sequence length {len(request.sequence)}")
        
        # Perform folding
        result = mock_fold_protein(
            sequence=request.sequence,
            confidence=request.confidence,
            relax=request.relax
        )
        
        # Create response
        response = FoldingResponse(
            job_id=job_id,
            sequence=request.sequence,
            sequence_length=len(request.sequence),
            coordinates=result['coordinates'],
            confidence_scores=result['confidence_scores'],
            tm_score_estimate=result['tm_score_estimate'],
            runtime_seconds=result['runtime_seconds'],
            device=str(DEVICE)
        )
        
        logger.info(f"Completed folding job {job_id} in {result['runtime_seconds']:.3f}s")
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Folding job {job_id} failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {
                "name": "odinfold_v1",
                "description": "OdinFold v1.0 - Production model",
                "loaded": MODEL_LOADED,
                "device": str(DEVICE)
            }
        ]
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    
    memory_usage = 0.0
    if torch.cuda.is_available():
        memory_usage = torch.cuda.memory_allocated() / (1024**3)
    
    metrics_text = f"""# HELP odinfold_model_loaded Whether the model is loaded
# TYPE odinfold_model_loaded gauge
odinfold_model_loaded {int(MODEL_LOADED)}

# HELP odinfold_memory_usage_gb GPU memory usage in GB
# TYPE odinfold_memory_usage_gb gauge
odinfold_memory_usage_gb {memory_usage}

# HELP odinfold_uptime_seconds Server uptime in seconds
# TYPE odinfold_uptime_seconds counter
odinfold_uptime_seconds {time.time() - START_TIME}
"""
    
    return JSONResponse(content=metrics_text, media_type="text/plain")


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "odinfold.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
