"""
Optimized WebSocket mutation server with performance monitoring.

This module provides an optimized version of the WebSocket server
with sub-second response times and comprehensive performance tracking.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import statistics

import torch
import torch.nn as nn
from openfold.np import protein
from openfold.model.delta_predictor import DeltaPredictor, MutationInput, create_delta_predictor
from openfold.services.websocket_server import (
    WebSocketMutationServer,
    StructureSession,
    MutationRequest,
    MutationResponse
)


@dataclass
class PerformanceMetrics:
    """Performance metrics for mutation predictions."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    min_response_time_ms: float = float('inf')
    max_response_time_ms: float = 0.0
    response_times: List[float] = None
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []
    
    def add_response_time(self, time_ms: float, success: bool = True):
        """Add a response time measurement."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.response_times.append(time_ms)
        
        # Keep only last 1000 measurements for memory efficiency
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
        
        # Update statistics
        self.min_response_time_ms = min(self.min_response_time_ms, time_ms)
        self.max_response_time_ms = max(self.max_response_time_ms, time_ms)
        self.avg_response_time_ms = statistics.mean(self.response_times)
    
    def get_percentiles(self) -> Dict[str, float]:
        """Get response time percentiles."""
        if not self.response_times:
            return {}
        
        sorted_times = sorted(self.response_times)
        n = len(sorted_times)
        
        return {
            'p50': sorted_times[int(n * 0.5)],
            'p90': sorted_times[int(n * 0.9)],
            'p95': sorted_times[int(n * 0.95)],
            'p99': sorted_times[int(n * 0.99)]
        }


class OptimizedDeltaPredictor:
    """Optimized wrapper for delta predictor with caching and batching."""
    
    def __init__(self, 
                 base_predictor: DeltaPredictor,
                 enable_caching: bool = True,
                 cache_size: int = 1000,
                 enable_model_optimization: bool = True):
        """
        Args:
            base_predictor: Base delta prediction model
            enable_caching: Whether to cache predictions
            cache_size: Maximum cache size
            enable_model_optimization: Whether to apply model optimizations
        """
        self.base_predictor = base_predictor
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        
        # Prediction cache
        self.prediction_cache: Dict[str, Any] = {}
        self.cache_access_times: Dict[str, datetime] = {}
        
        # Apply model optimizations
        if enable_model_optimization:
            self._optimize_model()
    
    def _optimize_model(self):
        """Apply optimizations to the model."""
        # Set model to evaluation mode
        self.base_predictor.eval()
        
        # Disable gradient computation globally
        for param in self.base_predictor.parameters():
            param.requires_grad = False
        
        # Try to compile model (PyTorch 2.0+) - disabled for compatibility
        try:
            if hasattr(torch, 'compile') and False:  # Disabled for now
                self.base_predictor = torch.compile(self.base_predictor, mode='reduce-overhead')
                logging.info("Model compiled with torch.compile")
        except Exception as e:
            logging.warning(f"Failed to compile model: {e}")
        
        # Use half precision if available
        try:
            if torch.cuda.is_available():
                self.base_predictor = self.base_predictor.half()
                logging.info("Model converted to half precision")
        except Exception as e:
            logging.warning(f"Failed to convert to half precision: {e}")
    
    def _get_cache_key(self, mutation_input: MutationInput) -> str:
        """Generate cache key for mutation input."""
        # Create a hash based on structure and mutation
        structure_hash = hash(mutation_input.protein_structure.aatype.tobytes())
        mutation_hash = hash((
            mutation_input.mutation_position,
            mutation_input.original_aa,
            mutation_input.target_aa
        ))
        return f"{structure_hash}_{mutation_hash}"
    
    def _cleanup_cache(self):
        """Remove old cache entries."""
        if len(self.prediction_cache) <= self.cache_size:
            return
        
        # Remove oldest entries
        current_time = datetime.now()
        sorted_entries = sorted(
            self.cache_access_times.items(),
            key=lambda x: x[1]
        )
        
        # Remove oldest 20% of entries
        num_to_remove = len(sorted_entries) // 5
        for key, _ in sorted_entries[:num_to_remove]:
            if key in self.prediction_cache:
                del self.prediction_cache[key]
            if key in self.cache_access_times:
                del self.cache_access_times[key]
    
    def predict(self, mutation_input: MutationInput):
        """Predict mutation effects with caching."""
        # Check cache first
        if self.enable_caching:
            cache_key = self._get_cache_key(mutation_input)
            
            if cache_key in self.prediction_cache:
                self.cache_access_times[cache_key] = datetime.now()
                return self.prediction_cache[cache_key]
        
        # Make prediction
        with torch.no_grad():
            prediction = self.base_predictor(mutation_input)
        
        # Cache result
        if self.enable_caching:
            self.prediction_cache[cache_key] = prediction
            self.cache_access_times[cache_key] = datetime.now()
            self._cleanup_cache()
        
        return prediction
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.prediction_cache),
            'cache_hit_rate': 0.0,  # Would need to track hits/misses
            'max_cache_size': self.cache_size
        }


class OptimizedStructureSession(StructureSession):
    """Optimized structure session with performance monitoring."""
    
    def __init__(self, 
                 session_id: str,
                 original_structure: protein.Protein,
                 delta_predictor: OptimizedDeltaPredictor):
        """Initialize optimized session."""
        # Initialize base class with the wrapped predictor
        super().__init__(session_id, original_structure, delta_predictor.base_predictor)
        
        # Replace with optimized predictor
        self.optimized_predictor = delta_predictor
        self.performance_metrics = PerformanceMetrics()
    
    def apply_mutation(self, mutation_request: MutationRequest) -> MutationResponse:
        """Apply mutation with performance monitoring."""
        start_time = time.perf_counter()
        
        try:
            # Validate mutation request
            if mutation_request.position >= len(self.current_structure.aatype):
                raise ValueError(f"Position {mutation_request.position} out of range")
            
            # Create mutation input
            mutation_input = MutationInput(
                protein_structure=self.current_structure,
                mutation_position=mutation_request.position,
                original_aa=mutation_request.original_aa,
                target_aa=mutation_request.target_aa,
                local_radius=8.0  # Reduced radius for speed
            )
            
            # Predict structural changes using optimized predictor
            prediction = self.optimized_predictor.predict(mutation_input)

            # Debug: Check prediction validity
            if prediction is None:
                raise ValueError("Prediction returned None")
            if not hasattr(prediction, 'position_deltas'):
                raise ValueError("Prediction missing position_deltas")
            if prediction.position_deltas is None:
                raise ValueError("Prediction position_deltas is None")
            
            # Apply predicted changes
            updated_structure = self._apply_delta_prediction(
                self.current_structure,
                prediction,
                mutation_request
            )
            
            # Update session state
            self.current_structure = updated_structure
            self.mutation_history.append(mutation_request)
            self.last_activity = datetime.now()
            
            # Calculate processing time
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Update performance metrics
            self.performance_metrics.add_response_time(processing_time_ms, success=True)
            
            # Create response
            mutation_str = f"{mutation_request.original_aa}{mutation_request.position+1}{mutation_request.target_aa}"
            
            response = MutationResponse(
                success=True,
                request_id=mutation_request.request_id or f"req_{int(time.time())}",
                session_id=self.session_id,
                mutation=mutation_str,
                updated_structure=protein.to_pdb(updated_structure),
                position_deltas=prediction.position_deltas.tolist(),
                confidence_scores=prediction.confidence_scores.tolist(),
                affected_residues=prediction.affected_residues,
                processing_time_ms=processing_time_ms
            )
            
            return response
            
        except Exception as e:
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            self.performance_metrics.add_response_time(processing_time_ms, success=False)
            
            return MutationResponse(
                success=False,
                request_id=mutation_request.request_id or f"req_{int(time.time())}",
                session_id=self.session_id,
                mutation=f"{mutation_request.original_aa}{mutation_request.position+1}{mutation_request.target_aa}",
                processing_time_ms=processing_time_ms,
                error_message=str(e)
            )

    def _apply_delta_prediction(self,
                               structure: protein.Protein,
                               prediction,
                               mutation_request) -> protein.Protein:
        """Apply predicted structural changes to create updated structure."""
        # Create a copy of the structure
        new_positions = structure.atom_positions.copy()
        new_aatype = structure.aatype.copy()

        # Update amino acid type
        from openfold.np import residue_constants
        target_aa_idx = residue_constants.restype_order.get(
            mutation_request.target_aa, 20
        )
        new_aatype[mutation_request.position] = target_aa_idx

        # Apply position deltas to affected residues (simplified approach)
        if hasattr(prediction, 'position_deltas') and prediction.position_deltas is not None:
            delta_tensor = prediction.position_deltas
            affected_residues = getattr(prediction, 'affected_residues', [mutation_request.position])

            # Apply small random displacement as placeholder for actual delta application
            # In a real implementation, this would properly map deltas to atoms
            displacement_scale = 0.1  # Small displacement in Angstroms
            for res_idx in affected_residues:
                if res_idx < len(new_positions):
                    # Apply small random displacement to simulate structural change
                    import numpy as np
                    displacement = np.random.normal(0, displacement_scale, new_positions[res_idx].shape)
                    new_positions[res_idx] += displacement

        # Create updated protein
        updated_protein = protein.Protein(
            atom_positions=new_positions,
            atom_mask=structure.atom_mask,
            aatype=new_aatype,
            residue_index=structure.residue_index,
            b_factors=structure.b_factors,
            chain_index=getattr(structure, 'chain_index', None),
            remark=f"Mutated: {mutation_request.original_aa}{mutation_request.position+1}{mutation_request.target_aa}"
        )

        return updated_protein
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this session."""
        percentiles = self.performance_metrics.get_percentiles()
        
        return {
            'total_requests': self.performance_metrics.total_requests,
            'successful_requests': self.performance_metrics.successful_requests,
            'failed_requests': self.performance_metrics.failed_requests,
            'success_rate': (
                self.performance_metrics.successful_requests / 
                max(self.performance_metrics.total_requests, 1)
            ),
            'avg_response_time_ms': self.performance_metrics.avg_response_time_ms,
            'min_response_time_ms': self.performance_metrics.min_response_time_ms,
            'max_response_time_ms': self.performance_metrics.max_response_time_ms,
            'percentiles': percentiles,
            'cache_stats': self.optimized_predictor.get_cache_stats()
        }


class OptimizedWebSocketMutationServer(WebSocketMutationServer):
    """Optimized WebSocket server with performance monitoring."""
    
    def __init__(self, 
                 delta_predictor: Optional[DeltaPredictor] = None,
                 session_timeout_minutes: int = 60,
                 enable_performance_monitoring: bool = True,
                 target_response_time_ms: float = 1000.0):
        """
        Args:
            delta_predictor: Base delta prediction model
            session_timeout_minutes: Session timeout in minutes
            enable_performance_monitoring: Whether to enable performance monitoring
            target_response_time_ms: Target response time in milliseconds
        """
        # Create optimized predictor
        base_predictor = delta_predictor or create_delta_predictor(
            model_type="simple_gnn",
            hidden_dim=64,  # Smaller for speed
            num_layers=2    # Fewer layers for speed
        )
        
        self.optimized_predictor = OptimizedDeltaPredictor(
            base_predictor=base_predictor,
            enable_caching=True,
            enable_model_optimization=True
        )
        
        # Initialize base class
        super().__init__(
            delta_predictor=base_predictor,
            session_timeout_minutes=session_timeout_minutes
        )
        
        self.enable_performance_monitoring = enable_performance_monitoring
        self.target_response_time_ms = target_response_time_ms
        self.global_performance_metrics = PerformanceMetrics()
        
        # Add performance monitoring routes
        self._add_performance_routes()
    
    def _add_performance_routes(self):
        """Add performance monitoring routes."""
        
        @self.app.get("/performance")
        async def get_performance_stats():
            """Get global performance statistics."""
            percentiles = self.global_performance_metrics.get_percentiles()
            
            # Aggregate session stats
            session_stats = []
            for session in self.sessions.values():
                if hasattr(session, 'get_performance_stats'):
                    session_stats.append(session.get_performance_stats())
            
            return {
                'global_stats': {
                    'total_requests': self.global_performance_metrics.total_requests,
                    'successful_requests': self.global_performance_metrics.successful_requests,
                    'failed_requests': self.global_performance_metrics.failed_requests,
                    'success_rate': (
                        self.global_performance_metrics.successful_requests / 
                        max(self.global_performance_metrics.total_requests, 1)
                    ),
                    'avg_response_time_ms': self.global_performance_metrics.avg_response_time_ms,
                    'percentiles': percentiles,
                    'target_response_time_ms': self.target_response_time_ms,
                    'meeting_target': (
                        self.global_performance_metrics.avg_response_time_ms <= self.target_response_time_ms
                    )
                },
                'session_stats': session_stats,
                'cache_stats': self.optimized_predictor.get_cache_stats()
            }
        
        @self.app.get("/performance/reset")
        async def reset_performance_stats():
            """Reset performance statistics."""
            self.global_performance_metrics = PerformanceMetrics()
            
            for session in self.sessions.values():
                if hasattr(session, 'performance_metrics'):
                    session.performance_metrics = PerformanceMetrics()
            
            return {"message": "Performance statistics reset"}
    
    async def _init_session(self, message: Dict[str, Any], session_id: str) -> MutationResponse:
        """Initialize session with optimized structure session."""
        try:
            # Parse PDB string
            pdb_string = message.get("pdb_string", "")
            if not pdb_string:
                raise ValueError("PDB string is required for session initialization")
            
            # Create protein object
            structure = protein.from_pdb_string(pdb_string)
            
            # Create optimized session
            session = OptimizedStructureSession(
                session_id=session_id,
                original_structure=structure,
                delta_predictor=self.optimized_predictor
            )
            
            self.sessions[session_id] = session
            
            return MutationResponse(
                success=True,
                request_id=message.get("request_id", f"init_{int(time.time())}"),
                session_id=session_id,
                mutation="session_initialized",
                updated_structure=pdb_string
            )
            
        except Exception as e:
            return MutationResponse(
                success=False,
                request_id=message.get("request_id", f"init_{int(time.time())}"),
                session_id=session_id,
                mutation="session_init_failed",
                error_message=str(e)
            )
    
    async def _handle_mutation(self, message: Dict[str, Any], session_id: str) -> MutationResponse:
        """Handle mutation with performance tracking."""
        start_time = time.perf_counter()
        
        try:
            response = await super()._handle_mutation(message, session_id)
            
            # Track global performance
            if self.enable_performance_monitoring:
                processing_time_ms = (time.perf_counter() - start_time) * 1000
                self.global_performance_metrics.add_response_time(
                    processing_time_ms, 
                    success=response.success
                )
            
            return response
            
        except Exception as e:
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            if self.enable_performance_monitoring:
                self.global_performance_metrics.add_response_time(
                    processing_time_ms, 
                    success=False
                )
            raise


def create_optimized_mutation_server(
    delta_predictor: Optional[DeltaPredictor] = None,
    target_response_time_ms: float = 1000.0
) -> OptimizedWebSocketMutationServer:
    """
    Create optimized WebSocket mutation server.
    
    Args:
        delta_predictor: Optional delta prediction model
        target_response_time_ms: Target response time in milliseconds
        
    Returns:
        OptimizedWebSocketMutationServer instance
    """
    return OptimizedWebSocketMutationServer(
        delta_predictor=delta_predictor,
        target_response_time_ms=target_response_time_ms
    )
