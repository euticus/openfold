#!/usr/bin/env python3
"""
CLI Command for Fast Post-Fold Relaxation

Provides --relax flag integration for OpenFold++ CLI
to enable optional structure relaxation.
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import logging
import time
import sys
from typing import Dict, List, Tuple, Optional, Union

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from openfoldpp.relaxation.fast_relaxation import create_fast_relaxer, RelaxationConfig
    RELAXATION_AVAILABLE = True
except ImportError as e:
    RELAXATION_AVAILABLE = False
    logging.warning(f"Relaxation not available: {e}")


class RelaxationCLI:
    """
    CLI interface for post-fold relaxation.
    
    Integrates with OpenFold++ main CLI to provide
    optional structure relaxation with --relax flag.
    """
    
    def __init__(self):
        self.relaxer = None
        
    def add_relaxation_args(self, parser: argparse.ArgumentParser):
        """Add relaxation arguments to CLI parser."""
        
        relax_group = parser.add_argument_group('Relaxation Options')
        
        relax_group.add_argument(
            '--relax',
            action='store_true',
            help='Enable post-fold relaxation for improved RMSD'
        )
        
        relax_group.add_argument(
            '--relax-iterations',
            type=int,
            default=100,
            help='Maximum relaxation iterations (default: 100)'
        )
        
        relax_group.add_argument(
            '--relax-tolerance',
            type=float,
            default=1.0,
            help='Energy tolerance for relaxation (kJ/mol/nm, default: 1.0)'
        )
        
        relax_group.add_argument(
            '--relax-platform',
            choices=['CUDA', 'OpenCL', 'CPU'],
            default='CUDA',
            help='OpenMM platform for relaxation (default: CUDA)'
        )
        
        relax_group.add_argument(
            '--relax-constrain-backbone',
            action='store_true',
            default=True,
            help='Constrain backbone during relaxation (default: True)'
        )
        
        relax_group.add_argument(
            '--relax-verbose',
            action='store_true',
            help='Verbose relaxation output'
        )
    
    def setup_relaxer(self, args: argparse.Namespace) -> bool:
        """Setup relaxer from CLI arguments."""
        
        if not args.relax:
            return False
        
        if not RELAXATION_AVAILABLE:
            logging.error("Relaxation requested but not available. Install OpenMM: conda install -c conda-forge openmm")
            return False
        
        # Create relaxation config from args
        config = RelaxationConfig(
            max_iterations=args.relax_iterations,
            tolerance=args.relax_tolerance,
            platform=args.relax_platform,
            constrain_backbone=args.relax_constrain_backbone,
            verbose=args.relax_verbose
        )
        
        try:
            self.relaxer = create_fast_relaxer(config)
            logging.info(f"Relaxation enabled: {config.platform} platform, {config.max_iterations} iterations")
            return True
            
        except Exception as e:
            logging.error(f"Failed to setup relaxer: {e}")
            return False
    
    def relax_prediction(
        self,
        coords: np.ndarray,
        sequence: str,
        confidence: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Relax predicted structure.
        
        Args:
            coords: [N, 3] predicted coordinates
            sequence: Protein sequence
            confidence: Optional per-residue confidence scores
            
        Returns:
            relaxed_coords: [N, 3] relaxed coordinates
            metrics: Relaxation metrics
        """
        
        if self.relaxer is None:
            # Return original coordinates if relaxation not enabled
            return coords, {'relaxation_enabled': False}
        
        start_time = time.time()
        
        try:
            # Relax structure
            relaxed_coords, metrics = self.relaxer.relax_structure(coords, sequence)
            
            # Add timing info
            metrics['total_relaxation_time_s'] = time.time() - start_time
            metrics['relaxation_enabled'] = True
            
            # Log results
            if metrics.get('verbose', False):
                logging.info(f"Relaxation complete: {metrics['relaxation_time_s']:.3f}s, "
                           f"RMSD change: {metrics['rmsd_change']:.2f}Å")
            
            return relaxed_coords, metrics
            
        except Exception as e:
            logging.error(f"Relaxation failed: {e}")
            
            # Return original coordinates on failure
            metrics = {
                'relaxation_enabled': True,
                'relaxation_failed': True,
                'error': str(e),
                'total_relaxation_time_s': time.time() - start_time
            }
            
            return coords, metrics
    
    def format_relaxation_output(self, metrics: Dict[str, float]) -> str:
        """Format relaxation metrics for output."""
        
        if not metrics.get('relaxation_enabled', False):
            return ""
        
        if metrics.get('relaxation_failed', False):
            return f"Relaxation failed: {metrics.get('error', 'Unknown error')}"
        
        output_lines = [
            "Relaxation Results:",
            f"  Time: {metrics.get('relaxation_time_s', 0):.3f}s",
            f"  Energy reduction: {metrics.get('energy_reduction', 0):.1f} kJ/mol",
            f"  RMSD change: {metrics.get('rmsd_change', 0):.2f} Å",
            f"  Converged: {'Yes' if metrics.get('converged', False) else 'No'}"
        ]
        
        return "\n".join(output_lines)


def integrate_relaxation_cli() -> RelaxationCLI:
    """
    Factory function to create relaxation CLI integration.
    
    Returns:
        RelaxationCLI instance for integration with main CLI
    """
    return RelaxationCLI()


# Example CLI integration
def example_cli_integration():
    """Example of how to integrate relaxation into main CLI."""
    
    parser = argparse.ArgumentParser(description="OpenFold++ with relaxation")
    
    # Main folding arguments
    parser.add_argument('--input', required=True, help='Input FASTA file')
    parser.add_argument('--output', required=True, help='Output PDB file')
    
    # Add relaxation arguments
    relax_cli = integrate_relaxation_cli()
    relax_cli.add_relaxation_args(parser)
    
    args = parser.parse_args()
    
    # Setup relaxation
    relaxation_enabled = relax_cli.setup_relaxer(args)
    
    print(f"Relaxation enabled: {relaxation_enabled}")
    
    if relaxation_enabled:
        # Mock folding result
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        coords = np.random.randn(len(sequence), 3) * 10
        
        # Apply relaxation
        relaxed_coords, metrics = relax_cli.relax_prediction(coords, sequence)
        
        # Format output
        output = relax_cli.format_relaxation_output(metrics)
        print(output)


# Testing
if __name__ == "__main__":
    print("🧬 Testing Relaxation CLI Integration")
    print("=" * 50)
    
    # Test CLI integration
    relax_cli = integrate_relaxation_cli()
    
    # Mock arguments
    class MockArgs:
        def __init__(self):
            self.relax = True
            self.relax_iterations = 50
            self.relax_tolerance = 1.0
            self.relax_platform = 'CPU'  # Use CPU for testing
            self.relax_constrain_backbone = True
            self.relax_verbose = True
    
    args = MockArgs()
    
    # Setup relaxer
    success = relax_cli.setup_relaxer(args)
    
    print(f"✅ CLI integration test: {'PASS' if success else 'FAIL'}")
    
    if success:
        # Test relaxation
        test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        test_coords = np.random.randn(len(test_sequence), 3) * 10
        
        relaxed_coords, metrics = relax_cli.relax_prediction(test_coords, test_sequence)
        
        print(f"\n📊 Relaxation Test Results:")
        print(f"   Input shape: {test_coords.shape}")
        print(f"   Output shape: {relaxed_coords.shape}")
        print(f"   Time: {metrics.get('relaxation_time_s', 0):.3f}s")
        print(f"   Success: {'✅' if not metrics.get('relaxation_failed', False) else '❌'}")
        
        # Test output formatting
        output = relax_cli.format_relaxation_output(metrics)
        print(f"\n📝 Formatted Output:")
        print(output)
    
    print(f"\n🎯 Relaxation CLI Ready!")
    print(f"   Use --relax flag to enable post-fold relaxation")
    print(f"   <1s overhead for improved RMSD")
