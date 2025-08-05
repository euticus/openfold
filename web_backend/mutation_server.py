"""
High-Performance Async Mutation Scanning Web Backend

FastAPI-based async web service for high-throughput mutation scanning.
Replaces slow Python mutation scanner with optimized async processing.
"""

import asyncio
import time
import logging
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import redis
from celery import Celery
import psutil

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OdinFold Mutation Scanning API",
    description="High-performance async mutation scanning service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Celery for background tasks
celery_app = Celery(
    "mutation_scanner",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

# Global model and executor
model = None
executor = ThreadPoolExecutor(max_workers=4)


# Pydantic models for API
class MutationRequest(BaseModel):
    """Single mutation request."""
    position: int = Field(..., ge=0, description="0-based position in sequence")
    from_aa: str = Field(..., min_length=1, max_length=1, description="Original amino acid")
    to_aa: str = Field(..., min_length=1, max_length=1, description="New amino acid")


class MutationScanRequest(BaseModel):
    """Mutation scanning request."""
    sequence: str = Field(..., min_length=1, max_length=2048, description="Protein sequence")
    mutations: List[MutationRequest] = Field(..., min_items=1, max_items=1000, description="List of mutations to analyze")
    batch_size: Optional[int] = Field(32, ge=1, le=128, description="Batch size for processing")
    include_confidence: bool = Field(True, description="Include confidence scores")
    include_structural_impact: bool = Field(False, description="Include structural impact analysis")


class MutationResult(BaseModel):
    """Single mutation result."""
    position: int
    from_aa: str
    to_aa: str
    ddg_kcal_mol: float
    confidence: Optional[float] = None
    structural_impact: Optional[str] = None
    processing_time_ms: Optional[float] = None


class MutationScanResponse(BaseModel):
    """Mutation scanning response."""
    sequence: str
    sequence_length: int
    num_mutations: int
    results: List[MutationResult]
    summary: Dict[str, Any]
    processing_time_ms: float
    server_info: Dict[str, Any]


class BatchMutationRequest(BaseModel):
    """Batch mutation request for multiple proteins."""
    proteins: List[MutationScanRequest] = Field(..., min_items=1, max_items=100)
    priority: str = Field("normal", regex="^(low|normal|high)$")


class SystemStatus(BaseModel):
    """System status response."""
    status: str
    uptime_seconds: float
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_usage_percent: Optional[float] = None
    active_requests: int
    queue_size: int
    model_loaded: bool


# Global state
server_start_time = time.time()
active_requests = 0
request_queue_size = 0


@dataclass
class MutationPredictor:
    """High-performance mutation effect predictor."""
    
    def __init__(self, model_path: str = "models/odinfold.pt", device: str = "auto"):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    async def load_model(self):
        """Load the mutation prediction model."""
        if self.model is None:
            logger.info(f"Loading model from {self.model_path}")
            # Mock model loading - replace with actual model
            await asyncio.sleep(0.1)  # Simulate loading time
            self.model = "mock_model"  # Replace with actual model
            logger.info("Model loaded successfully")
    
    async def predict_mutations(self, 
                               sequence: str, 
                               mutations: List[MutationRequest],
                               batch_size: int = 32,
                               include_confidence: bool = True) -> List[MutationResult]:
        """Predict mutation effects asynchronously."""
        
        if self.model is None:
            await self.load_model()
        
        results = []
        
        # Process mutations in batches
        for i in range(0, len(mutations), batch_size):
            batch_mutations = mutations[i:i + batch_size]
            
            # Process batch asynchronously
            batch_results = await self._process_mutation_batch(
                sequence, batch_mutations, include_confidence
            )
            results.extend(batch_results)
        
        return results
    
    async def _process_mutation_batch(self, 
                                    sequence: str, 
                                    mutations: List[MutationRequest],
                                    include_confidence: bool) -> List[MutationResult]:
        """Process a batch of mutations."""
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def _compute_batch():
            batch_results = []
            
            for mutation in mutations:
                start_time = time.time()
                
                # Validate mutation
                if mutation.position >= len(sequence):
                    raise ValueError(f"Position {mutation.position} out of range for sequence length {len(sequence)}")
                
                if sequence[mutation.position] != mutation.from_aa:
                    raise ValueError(f"Sequence has {sequence[mutation.position]} at position {mutation.position}, not {mutation.from_aa}")
                
                # Mock ΔΔG prediction (replace with actual model inference)
                ddg = self._mock_ddg_prediction(sequence, mutation)
                confidence = self._mock_confidence_prediction() if include_confidence else None
                
                processing_time = (time.time() - start_time) * 1000
                
                result = MutationResult(
                    position=mutation.position,
                    from_aa=mutation.from_aa,
                    to_aa=mutation.to_aa,
                    ddg_kcal_mol=ddg,
                    confidence=confidence,
                    processing_time_ms=processing_time
                )
                
                batch_results.append(result)
            
            return batch_results
        
        return await loop.run_in_executor(executor, _compute_batch)
    
    def _mock_ddg_prediction(self, sequence: str, mutation: MutationRequest) -> float:
        """Mock ΔΔG prediction - replace with actual model."""
        # Simple mock based on amino acid properties
        hydrophobicity = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        from_hydro = hydrophobicity.get(mutation.from_aa, 0)
        to_hydro = hydrophobicity.get(mutation.to_aa, 0)
        
        # Mock ΔΔG based on hydrophobicity change + noise
        ddg = (to_hydro - from_hydro) * 0.5 + np.random.normal(0, 1.0)
        
        return round(ddg, 3)
    
    def _mock_confidence_prediction(self) -> float:
        """Mock confidence prediction."""
        return round(np.random.uniform(0.7, 0.95), 3)


# Initialize global predictor
predictor = MutationPredictor()


# API endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the server."""
    logger.info("Starting OdinFold Mutation Scanning API")
    await predictor.load_model()
    logger.info("Server ready")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "service": "OdinFold Mutation Scanning API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=SystemStatus)
async def health_check():
    """Health check endpoint."""
    
    # Get system metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    gpu_usage = None
    gpu_memory = None
    
    if GPUTIL_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_usage = gpu.load * 100
                gpu_memory = (gpu.memoryUsed / gpu.memoryTotal) * 100
        except:
            pass
    
    return SystemStatus(
        status="healthy",
        uptime_seconds=time.time() - server_start_time,
        cpu_usage_percent=cpu_percent,
        memory_usage_percent=memory.percent,
        gpu_usage_percent=gpu_usage,
        gpu_memory_usage_percent=gpu_memory,
        active_requests=active_requests,
        queue_size=request_queue_size,
        model_loaded=predictor.model is not None
    )


@app.post("/api/mutations/scan", response_model=MutationScanResponse)
async def scan_mutations(request: MutationScanRequest):
    """Scan mutations for a single protein."""
    
    global active_requests
    active_requests += 1
    
    try:
        start_time = time.time()
        
        logger.info(f"Processing mutation scan: {len(request.mutations)} mutations for sequence length {len(request.sequence)}")
        
        # Predict mutations
        results = await predictor.predict_mutations(
            sequence=request.sequence,
            mutations=request.mutations,
            batch_size=request.batch_size,
            include_confidence=request.include_confidence
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate summary statistics
        ddg_values = [r.ddg_kcal_mol for r in results]
        summary = {
            "mean_ddg": round(np.mean(ddg_values), 3),
            "std_ddg": round(np.std(ddg_values), 3),
            "min_ddg": round(np.min(ddg_values), 3),
            "max_ddg": round(np.max(ddg_values), 3),
            "stabilizing_mutations": sum(1 for ddg in ddg_values if ddg < -1.0),
            "destabilizing_mutations": sum(1 for ddg in ddg_values if ddg > 1.0),
            "neutral_mutations": sum(1 for ddg in ddg_values if -1.0 <= ddg <= 1.0)
        }
        
        # Server info
        server_info = {
            "processing_time_ms": round(processing_time, 2),
            "mutations_per_second": round(len(request.mutations) / (processing_time / 1000), 1),
            "batch_size_used": request.batch_size,
            "device": str(predictor.device)
        }
        
        response = MutationScanResponse(
            sequence=request.sequence,
            sequence_length=len(request.sequence),
            num_mutations=len(request.mutations),
            results=results,
            summary=summary,
            processing_time_ms=processing_time,
            server_info=server_info
        )
        
        logger.info(f"Completed mutation scan in {processing_time:.1f}ms ({len(request.mutations)/(processing_time/1000):.1f} mutations/s)")
        
        return response
        
    except Exception as e:
        logger.error(f"Mutation scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        active_requests -= 1


@app.post("/api/mutations/batch")
async def batch_scan_mutations(request: BatchMutationRequest, background_tasks: BackgroundTasks):
    """Submit batch mutation scanning job."""
    
    # Create background task for batch processing
    task_id = f"batch_{int(time.time())}"
    
    background_tasks.add_task(
        process_batch_mutations,
        task_id,
        request.proteins,
        request.priority
    )
    
    return {
        "task_id": task_id,
        "status": "submitted",
        "num_proteins": len(request.proteins),
        "estimated_time_minutes": len(request.proteins) * 2  # Rough estimate
    }


async def process_batch_mutations(task_id: str, proteins: List[MutationScanRequest], priority: str):
    """Process batch mutations in background."""
    
    logger.info(f"Starting batch task {task_id} with {len(proteins)} proteins")
    
    results = []
    
    for i, protein_request in enumerate(proteins):
        try:
            logger.info(f"Processing protein {i+1}/{len(proteins)} in batch {task_id}")
            
            # Process single protein
            protein_results = await predictor.predict_mutations(
                sequence=protein_request.sequence,
                mutations=protein_request.mutations,
                batch_size=protein_request.batch_size,
                include_confidence=protein_request.include_confidence
            )
            
            results.append({
                "protein_index": i,
                "sequence_length": len(protein_request.sequence),
                "num_mutations": len(protein_request.mutations),
                "results": [r.dict() for r in protein_results]
            })
            
        except Exception as e:
            logger.error(f"Failed to process protein {i} in batch {task_id}: {e}")
            results.append({
                "protein_index": i,
                "error": str(e)
            })
    
    # Store results (in production, use Redis or database)
    logger.info(f"Completed batch task {task_id}")


@app.get("/api/mutations/batch/{task_id}")
async def get_batch_status(task_id: str):
    """Get batch processing status."""
    
    # Mock status - in production, check actual task status
    return {
        "task_id": task_id,
        "status": "completed",  # or "running", "failed"
        "progress": 100,
        "results_available": True
    }


if __name__ == "__main__":
    uvicorn.run(
        "mutation_server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        loop="asyncio",
        log_level="info"
    )
