"""
Mutation server with integrated MD refinement.

This module combines the real-time mutation system with MD-based
structure refinement to produce high-quality mutated structures.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict

import torch
import numpy as np
from openfold.np import protein
from openfold.model.delta_predictor import DeltaPredictor, MutationInput
# Import MD refinement with fallback
try:
    from openfold.utils.md_refinement import (
        MDRefinementPipeline,
        EnhancedAmberRefinement,
        refine_openfold_output
    )
    MD_REFINEMENT_AVAILABLE = True
except ImportError:
    MD_REFINEMENT_AVAILABLE = False
    logging.warning("MD refinement not available (missing dependencies)")
from openfold.services.optimized_mutation_server import (
    OptimizedWebSocketMutationServer,
    OptimizedStructureSession,
    MutationRequest,
    MutationResponse,
    PerformanceMetrics
)


@dataclass
class RefinementConfig:
    """Configuration for post-mutation refinement."""
    enable_refinement: bool = True
    refinement_method: str = "amber"  # "amber", "openmm", "torchmd", or "auto"
    max_refinement_time_ms: float = 5000.0  # 5 second timeout
    refinement_steps: int = 100  # Number of refinement steps
    enable_clash_detection: bool = True
    clash_threshold: float = 2.0  # Angstrom threshold for clash detection
    energy_minimization: bool = True
    constraint_strength: float = 1.0  # Constraint strength for refinement


@dataclass
class RefinedMutationResponse(MutationResponse):
    """Extended mutation response with refinement information."""
    refinement_applied: bool = False
    refinement_method: Optional[str] = None
    refinement_time_ms: float = 0.0
    initial_energy: Optional[float] = None
    final_energy: Optional[float] = None
    energy_improvement: Optional[float] = None
    clashes_detected: int = 0
    clashes_resolved: int = 0
    refinement_rmsd: Optional[float] = None
    refinement_error: Optional[str] = None


class StructureQualityAnalyzer:
    """Analyzes structure quality and detects issues."""
    
    def __init__(self, clash_threshold: float = 2.0):
        """
        Args:
            clash_threshold: Distance threshold for clash detection (Angstroms)
        """
        self.clash_threshold = clash_threshold
    
    def detect_clashes(self, structure: protein.Protein) -> List[Dict[str, Any]]:
        """
        Detect atomic clashes in the structure.
        
        Args:
            structure: Protein structure to analyze
            
        Returns:
            List of clash information dictionaries
        """
        clashes = []
        positions = structure.atom_positions
        atom_mask = structure.atom_mask
        
        # Check all atom pairs for clashes
        for res_i in range(len(positions)):
            for atom_i in range(positions.shape[1]):
                if atom_mask[res_i, atom_i] == 0:
                    continue
                
                pos_i = positions[res_i, atom_i]
                
                for res_j in range(res_i, len(positions)):
                    start_atom = atom_i + 1 if res_j == res_i else 0
                    
                    for atom_j in range(start_atom, positions.shape[1]):
                        if atom_mask[res_j, atom_j] == 0:
                            continue
                        
                        pos_j = positions[res_j, atom_j]
                        distance = np.linalg.norm(pos_i - pos_j)
                        
                        # Skip bonded atoms (very close)
                        if distance < 0.5:
                            continue
                        
                        if distance < self.clash_threshold:
                            clashes.append({
                                'residue_1': res_i,
                                'atom_1': atom_i,
                                'residue_2': res_j,
                                'atom_2': atom_j,
                                'distance': distance,
                                'severity': self.clash_threshold - distance
                            })
        
        return clashes
    
    def calculate_structure_energy(self, structure: protein.Protein) -> float:
        """
        Calculate approximate structure energy.
        
        Args:
            structure: Protein structure
            
        Returns:
            Approximate energy value
        """
        # Simple energy approximation based on distances
        positions = structure.atom_positions
        atom_mask = structure.atom_mask
        
        total_energy = 0.0
        
        for res_i in range(len(positions)):
            for atom_i in range(positions.shape[1]):
                if atom_mask[res_i, atom_i] == 0:
                    continue
                
                pos_i = positions[res_i, atom_i]
                
                for res_j in range(res_i + 1, len(positions)):
                    for atom_j in range(positions.shape[1]):
                        if atom_mask[res_j, atom_j] == 0:
                            continue
                        
                        pos_j = positions[res_j, atom_j]
                        distance = np.linalg.norm(pos_i - pos_j)
                        
                        # Simple Lennard-Jones-like potential
                        if distance > 0.5:  # Avoid division by zero
                            # Attractive term (simplified)
                            attractive = -1.0 / (distance ** 6)
                            # Repulsive term (simplified)
                            repulsive = 1.0 / (distance ** 12)
                            total_energy += attractive + repulsive
        
        return total_energy
    
    def assess_structure_quality(self, structure: protein.Protein) -> Dict[str, Any]:
        """
        Comprehensive structure quality assessment.
        
        Args:
            structure: Protein structure to assess
            
        Returns:
            Quality assessment dictionary
        """
        clashes = self.detect_clashes(structure)
        energy = self.calculate_structure_energy(structure)
        
        # Calculate quality metrics
        clash_count = len(clashes)
        severe_clashes = sum(1 for clash in clashes if clash['severity'] > 1.0)
        
        # Overall quality score (0-1, higher is better)
        quality_score = max(0.0, 1.0 - (clash_count * 0.1) - (severe_clashes * 0.2))
        
        return {
            'clash_count': clash_count,
            'severe_clashes': severe_clashes,
            'energy': energy,
            'quality_score': quality_score,
            'clashes': clashes,
            'needs_refinement': clash_count > 0 or quality_score < 0.8
        }


class RefinedStructureSession(OptimizedStructureSession):
    """Structure session with integrated MD refinement."""
    
    def __init__(self, 
                 session_id: str,
                 original_structure: protein.Protein,
                 delta_predictor,
                 refinement_config: RefinementConfig = None):
        """
        Args:
            session_id: Unique session identifier
            original_structure: Initial protein structure
            delta_predictor: Optimized delta prediction model
            refinement_config: Configuration for refinement
        """
        super().__init__(session_id, original_structure, delta_predictor)
        
        self.refinement_config = refinement_config or RefinementConfig()
        self.quality_analyzer = StructureQualityAnalyzer(
            clash_threshold=self.refinement_config.clash_threshold
        )
        
        # Initialize refinement pipeline
        self.refinement_pipeline = None
        if self.refinement_config.enable_refinement and MD_REFINEMENT_AVAILABLE:
            try:
                self.refinement_pipeline = MDRefinementPipeline(
                    methods=[self.refinement_config.refinement_method],
                    use_gpu=False,  # Use CPU for compatibility
                    fallback_on_failure=True
                )
                logging.info(f"MD refinement pipeline initialized with {self.refinement_config.refinement_method}")
            except Exception as e:
                logging.warning(f"Failed to initialize MD refinement: {e}")
                self.refinement_config.enable_refinement = False
        elif self.refinement_config.enable_refinement and not MD_REFINEMENT_AVAILABLE:
            logging.warning("MD refinement requested but not available")
            self.refinement_config.enable_refinement = False
    
    def apply_mutation(self, mutation_request: MutationRequest) -> RefinedMutationResponse:
        """Apply mutation with optional refinement."""
        start_time = time.perf_counter()
        
        try:
            # Apply base mutation
            base_response = super().apply_mutation(mutation_request)
            
            if not base_response.success:
                # Convert to refined response
                return RefinedMutationResponse(
                    success=False,
                    request_id=base_response.request_id,
                    session_id=base_response.session_id,
                    mutation=base_response.mutation,
                    processing_time_ms=base_response.processing_time_ms,
                    error_message=base_response.error_message,
                    refinement_applied=False
                )
            
            # Analyze structure quality
            quality_assessment = self.quality_analyzer.assess_structure_quality(
                self.current_structure
            )
            
            initial_energy = quality_assessment['energy']
            clashes_detected = quality_assessment['clash_count']
            
            # Apply refinement if needed and enabled
            refinement_applied = False
            refinement_method = None
            refinement_time_ms = 0.0
            final_energy = initial_energy
            refinement_rmsd = None
            refinement_error = None
            clashes_resolved = 0
            
            if (self.refinement_config.enable_refinement and 
                quality_assessment['needs_refinement'] and 
                self.refinement_pipeline is not None):
                
                refinement_start = time.perf_counter()
                
                try:
                    # Apply refinement (or simulate if not available)
                    if self.refinement_pipeline:
                        refined_pdb, refinement_info = self.refinement_pipeline.refine_structure(
                            self.current_structure
                        )
                    else:
                        # Simulate refinement for demonstration
                        refined_pdb = protein.to_pdb(self.current_structure)
                        refinement_info = {
                            'method': 'simulated',
                            'rmsd': 0.1,
                            'energy_change': -5.0
                        }
                    
                    if refined_pdb and not refinement_info.get('error'):
                        # Parse refined structure
                        refined_structure = protein.from_pdb_string(refined_pdb)
                        
                        # Assess refined structure quality
                        refined_quality = self.quality_analyzer.assess_structure_quality(
                            refined_structure
                        )
                        
                        # Check if refinement improved the structure
                        if refined_quality['quality_score'] > quality_assessment['quality_score']:
                            self.current_structure = refined_structure
                            refinement_applied = True
                            refinement_method = refinement_info.get('method', 'unknown')
                            final_energy = refined_quality['energy']
                            refinement_rmsd = refinement_info.get('rmsd')
                            clashes_resolved = max(0, clashes_detected - refined_quality['clash_count'])
                        else:
                            logging.info("Refinement did not improve structure quality, keeping original")
                    
                except Exception as e:
                    refinement_error = str(e)
                    logging.warning(f"Refinement failed: {e}")
                
                refinement_time_ms = (time.perf_counter() - refinement_start) * 1000
            
            # Calculate total processing time
            total_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Update performance metrics
            self.performance_metrics.add_response_time(total_time_ms, success=True)
            
            # Create refined response
            response = RefinedMutationResponse(
                success=True,
                request_id=base_response.request_id,
                session_id=self.session_id,
                mutation=base_response.mutation,
                updated_structure=protein.to_pdb(self.current_structure),
                position_deltas=base_response.position_deltas,
                confidence_scores=base_response.confidence_scores,
                affected_residues=base_response.affected_residues,
                processing_time_ms=total_time_ms,
                refinement_applied=refinement_applied,
                refinement_method=refinement_method,
                refinement_time_ms=refinement_time_ms,
                initial_energy=initial_energy,
                final_energy=final_energy,
                energy_improvement=initial_energy - final_energy if refinement_applied else None,
                clashes_detected=clashes_detected,
                clashes_resolved=clashes_resolved,
                refinement_rmsd=refinement_rmsd,
                refinement_error=refinement_error
            )
            
            return response
            
        except Exception as e:
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            self.performance_metrics.add_response_time(processing_time_ms, success=False)
            
            return RefinedMutationResponse(
                success=False,
                request_id=mutation_request.request_id or f"req_{int(time.time())}",
                session_id=self.session_id,
                mutation=f"{mutation_request.original_aa}{mutation_request.position+1}{mutation_request.target_aa}",
                processing_time_ms=processing_time_ms,
                error_message=str(e),
                refinement_applied=False
            )


class MutationRefinementServer(OptimizedWebSocketMutationServer):
    """WebSocket server with integrated mutation and refinement."""
    
    def __init__(self, 
                 delta_predictor: Optional[DeltaPredictor] = None,
                 refinement_config: RefinementConfig = None,
                 session_timeout_minutes: int = 60):
        """
        Args:
            delta_predictor: Delta prediction model
            refinement_config: Refinement configuration
            session_timeout_minutes: Session timeout in minutes
        """
        super().__init__(
            delta_predictor=delta_predictor,
            session_timeout_minutes=session_timeout_minutes,
            target_response_time_ms=2000.0  # Allow more time for refinement
        )
        
        self.refinement_config = refinement_config or RefinementConfig()
        
        # Add refinement-specific routes
        self._add_refinement_routes()
    
    def _add_refinement_routes(self):
        """Add refinement-specific API routes."""
        
        @self.app.get("/refinement/config")
        async def get_refinement_config():
            """Get current refinement configuration."""
            return asdict(self.refinement_config)
        
        @self.app.post("/refinement/config")
        async def update_refinement_config(config: dict):
            """Update refinement configuration."""
            try:
                # Update configuration
                for key, value in config.items():
                    if hasattr(self.refinement_config, key):
                        setattr(self.refinement_config, key, value)
                
                return {"message": "Refinement configuration updated", "config": asdict(self.refinement_config)}
            except Exception as e:
                return {"error": str(e)}
        
        @self.app.get("/refinement/stats")
        async def get_refinement_stats():
            """Get refinement statistics across all sessions."""
            stats = {
                'total_mutations': 0,
                'refinements_applied': 0,
                'avg_refinement_time_ms': 0.0,
                'energy_improvements': 0,
                'clashes_resolved': 0
            }
            
            refinement_times = []
            
            for session in self.sessions.values():
                if hasattr(session, 'mutation_history'):
                    for mutation in session.mutation_history:
                        stats['total_mutations'] += 1
                        # In a real implementation, we'd track refinement stats per mutation
            
            return stats
    
    async def _init_session(self, message: Dict[str, Any], session_id: str) -> MutationResponse:
        """Initialize session with refinement capabilities."""
        try:
            # Parse PDB string
            pdb_string = message.get("pdb_string", "")
            if not pdb_string:
                raise ValueError("PDB string is required for session initialization")
            
            # Create protein object
            structure = protein.from_pdb_string(pdb_string)
            
            # Create refined session
            session = RefinedStructureSession(
                session_id=session_id,
                original_structure=structure,
                delta_predictor=self.optimized_predictor,
                refinement_config=self.refinement_config
            )
            
            self.sessions[session_id] = session
            
            return MutationResponse(
                success=True,
                request_id=message.get("request_id", f"init_{int(time.time())}"),
                session_id=session_id,
                mutation="session_initialized_with_refinement",
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


def create_mutation_refinement_server(
    delta_predictor: Optional[DeltaPredictor] = None,
    refinement_config: RefinementConfig = None
) -> MutationRefinementServer:
    """
    Create mutation server with integrated refinement.
    
    Args:
        delta_predictor: Optional delta prediction model
        refinement_config: Optional refinement configuration
        
    Returns:
        MutationRefinementServer instance
    """
    return MutationRefinementServer(
        delta_predictor=delta_predictor,
        refinement_config=refinement_config
    )
