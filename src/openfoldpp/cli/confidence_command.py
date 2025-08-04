#!/usr/bin/env python3
"""
CLI Command for pLDDT Confidence Estimation

Provides --confidence flag integration for OpenFold++ CLI
to enable per-residue confidence prediction.
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import logging
import time
import json
import sys
from typing import Dict, List, Tuple, Optional, Union

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from openfoldpp.modules.confidence_head import create_confidence_head, ConfidenceConfig
    CONFIDENCE_AVAILABLE = True
except ImportError as e:
    CONFIDENCE_AVAILABLE = False
    logging.warning(f"Confidence estimation not available: {e}")


class ConfidenceCLI:
    """
    CLI interface for pLDDT confidence estimation.
    
    Integrates with OpenFold++ main CLI to provide
    optional per-residue confidence scores with --confidence flag.
    """
    
    def __init__(self):
        self.confidence_head = None
        self.config = None
        
    def add_confidence_args(self, parser: argparse.ArgumentParser):
        """Add confidence arguments to CLI parser."""
        
        conf_group = parser.add_argument_group('Confidence Options')
        
        conf_group.add_argument(
            '--confidence',
            action='store_true',
            help='Enable per-residue confidence prediction (pLDDT scores)'
        )
        
        conf_group.add_argument(
            '--confidence-bins',
            type=int,
            default=50,
            help='Number of distance bins for pLDDT calculation (default: 50)'
        )
        
        conf_group.add_argument(
            '--confidence-output',
            choices=['scores', 'categories', 'both'],
            default='both',
            help='Confidence output format (default: both)'
        )
        
        conf_group.add_argument(
            '--confidence-threshold',
            type=float,
            default=70.0,
            help='Confidence threshold for highlighting low-confidence regions (default: 70.0)'
        )
        
        conf_group.add_argument(
            '--confidence-format',
            choices=['json', 'csv', 'pdb'],
            default='json',
            help='Output format for confidence scores (default: json)'
        )
    
    def setup_confidence_head(self, args: argparse.Namespace) -> bool:
        """Setup confidence head from CLI arguments."""
        
        if not args.confidence:
            return False
        
        if not CONFIDENCE_AVAILABLE:
            logging.error("Confidence prediction requested but not available.")
            return False
        
        # Create confidence config from args
        self.config = ConfidenceConfig(
            input_dim=256,  # Standard single representation dimension
            hidden_dim=128,
            num_layers=3,
            num_bins=args.confidence_bins,
            dropout=0.1
        )
        
        try:
            self.confidence_head = create_confidence_head(self.config)
            self.confidence_head.eval()  # Set to evaluation mode
            
            logging.info(f"Confidence estimation enabled: {self.config.num_bins} bins")
            return True
            
        except Exception as e:
            logging.error(f"Failed to setup confidence head: {e}")
            return False
    
    def predict_confidence(
        self,
        single_repr: torch.Tensor,
        coordinates: torch.Tensor,
        sequence: str,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        Predict confidence scores for a structure.
        
        Args:
            single_repr: [seq_len, input_dim] single representation
            coordinates: [seq_len, 3] predicted coordinates
            sequence: Protein sequence string
            mask: Optional sequence mask
            
        Returns:
            Dictionary with confidence predictions and analysis
        """
        
        if self.confidence_head is None:
            return {'confidence_enabled': False}
        
        start_time = time.time()
        
        try:
            # Add batch dimension if needed
            if single_repr.dim() == 2:
                single_repr = single_repr.unsqueeze(0)
            if coordinates.dim() == 2:
                coordinates = coordinates.unsqueeze(0)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)
            
            # Predict confidence
            with torch.no_grad():
                outputs = self.confidence_head(single_repr, coordinates, mask)
            
            # Extract results
            plddt_scores = outputs['plddt'].squeeze(0)  # Remove batch dimension
            
            # Get confidence categories
            categories = self.confidence_head.get_confidence_categories(plddt_scores.unsqueeze(0))
            categories = {k: v.squeeze(0) for k, v in categories.items()}
            
            # Calculate statistics
            stats = self._calculate_confidence_stats(plddt_scores, sequence)
            
            results = {
                'confidence_enabled': True,
                'plddt_scores': plddt_scores,
                'categories': categories,
                'statistics': stats,
                'prediction_time_s': time.time() - start_time,
                'sequence_length': len(sequence)
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Confidence prediction failed: {e}")
            
            return {
                'confidence_enabled': True,
                'confidence_failed': True,
                'error': str(e),
                'prediction_time_s': time.time() - start_time
            }
    
    def _calculate_confidence_stats(
        self,
        plddt_scores: torch.Tensor,
        sequence: str
    ) -> Dict[str, float]:
        """Calculate confidence statistics."""
        
        scores_np = plddt_scores.cpu().numpy()
        
        stats = {
            'mean_plddt': float(np.mean(scores_np)),
            'median_plddt': float(np.median(scores_np)),
            'min_plddt': float(np.min(scores_np)),
            'max_plddt': float(np.max(scores_np)),
            'std_plddt': float(np.std(scores_np)),
            'very_high_confidence_pct': float(np.mean(scores_np >= 90) * 100),
            'confident_pct': float(np.mean(scores_np >= 70) * 100),
            'low_confidence_pct': float(np.mean((scores_np >= 50) & (scores_np < 70)) * 100),
            'very_low_confidence_pct': float(np.mean(scores_np < 50) * 100),
            'sequence_length': len(sequence)
        }
        
        return stats
    
    def format_confidence_output(
        self,
        results: Dict,
        args: argparse.Namespace,
        sequence: str
    ) -> str:
        """Format confidence results for output."""
        
        if not results.get('confidence_enabled', False):
            return ""
        
        if results.get('confidence_failed', False):
            return f"Confidence prediction failed: {results.get('error', 'Unknown error')}"
        
        output_lines = []
        
        # Header
        output_lines.append("Confidence Prediction Results (pLDDT):")
        output_lines.append("=" * 40)
        
        # Statistics
        stats = results['statistics']
        output_lines.extend([
            f"Sequence Length: {stats['sequence_length']}",
            f"Mean pLDDT: {stats['mean_plddt']:.1f}",
            f"Median pLDDT: {stats['median_plddt']:.1f}",
            f"Min pLDDT: {stats['min_plddt']:.1f}",
            f"Max pLDDT: {stats['max_plddt']:.1f}",
            ""
        ])
        
        # Confidence distribution
        output_lines.extend([
            "Confidence Distribution:",
            f"  Very High (≥90): {stats['very_high_confidence_pct']:.1f}%",
            f"  Confident (≥70): {stats['confident_pct']:.1f}%",
            f"  Low (50-70): {stats['low_confidence_pct']:.1f}%",
            f"  Very Low (<50): {stats['very_low_confidence_pct']:.1f}%",
            ""
        ])
        
        # Per-residue scores (if requested)
        if args.confidence_output in ['scores', 'both']:
            output_lines.append("Per-Residue Scores:")
            
            plddt_scores = results['plddt_scores'].cpu().numpy()
            
            # Show first 20 residues as example
            for i in range(min(20, len(sequence))):
                aa = sequence[i]
                score = plddt_scores[i]
                confidence_level = self._get_confidence_level(score)
                output_lines.append(f"  {i+1:3d} {aa} {score:5.1f} ({confidence_level})")
            
            if len(sequence) > 20:
                output_lines.append(f"  ... ({len(sequence) - 20} more residues)")
            
            output_lines.append("")
        
        # Low confidence regions
        if args.confidence_threshold:
            low_conf_regions = self._find_low_confidence_regions(
                results['plddt_scores'].cpu().numpy(),
                args.confidence_threshold
            )
            
            if low_conf_regions:
                output_lines.append(f"Low Confidence Regions (<{args.confidence_threshold}):")
                for start, end, avg_score in low_conf_regions:
                    output_lines.append(f"  Residues {start+1}-{end+1}: {avg_score:.1f} pLDDT")
                output_lines.append("")
        
        # Timing
        output_lines.append(f"Prediction time: {results['prediction_time_s']:.3f}s")
        
        return "\n".join(output_lines)
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level description for a score."""
        
        if score >= 90:
            return "Very High"
        elif score >= 70:
            return "Confident"
        elif score >= 50:
            return "Low"
        else:
            return "Very Low"
    
    def _find_low_confidence_regions(
        self,
        scores: np.ndarray,
        threshold: float,
        min_length: int = 3
    ) -> List[Tuple[int, int, float]]:
        """Find contiguous low confidence regions."""
        
        low_conf_mask = scores < threshold
        regions = []
        
        start = None
        for i, is_low in enumerate(low_conf_mask):
            if is_low and start is None:
                start = i
            elif not is_low and start is not None:
                if i - start >= min_length:
                    avg_score = np.mean(scores[start:i])
                    regions.append((start, i-1, avg_score))
                start = None
        
        # Handle region at end
        if start is not None and len(scores) - start >= min_length:
            avg_score = np.mean(scores[start:])
            regions.append((start, len(scores)-1, avg_score))
        
        return regions
    
    def save_confidence_results(
        self,
        results: Dict,
        output_path: Path,
        format_type: str = 'json',
        sequence: str = ""
    ):
        """Save confidence results to file."""
        
        if not results.get('confidence_enabled', False):
            return
        
        if format_type == 'json':
            # Convert tensors to lists for JSON serialization
            json_results = {
                'statistics': results['statistics'],
                'plddt_scores': results['plddt_scores'].cpu().tolist(),
                'sequence': sequence,
                'prediction_time_s': results['prediction_time_s']
            }
            
            with open(output_path.with_suffix('.confidence.json'), 'w') as f:
                json.dump(json_results, f, indent=2)
        
        elif format_type == 'csv':
            import csv
            
            with open(output_path.with_suffix('.confidence.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Residue', 'AA', 'pLDDT', 'Confidence_Level'])
                
                plddt_scores = results['plddt_scores'].cpu().numpy()
                for i, (aa, score) in enumerate(zip(sequence, plddt_scores)):
                    level = self._get_confidence_level(score)
                    writer.writerow([i+1, aa, f"{score:.1f}", level])
        
        logging.info(f"Confidence results saved: {output_path}")


def integrate_confidence_cli() -> ConfidenceCLI:
    """
    Factory function to create confidence CLI integration.
    
    Returns:
        ConfidenceCLI instance for integration with main CLI
    """
    return ConfidenceCLI()


# Testing
if __name__ == "__main__":
    print("🎯 Testing Confidence CLI Integration")
    print("=" * 50)
    
    # Test CLI integration
    conf_cli = integrate_confidence_cli()
    
    # Mock arguments
    class MockArgs:
        def __init__(self):
            self.confidence = True
            self.confidence_bins = 50
            self.confidence_output = 'both'
            self.confidence_threshold = 70.0
            self.confidence_format = 'json'
    
    args = MockArgs()
    
    # Setup confidence head
    success = conf_cli.setup_confidence_head(args)
    
    print(f"✅ CLI integration test: {'PASS' if success else 'FAIL'}")
    
    if success:
        # Test confidence prediction
        test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        single_repr = torch.randn(len(test_sequence), 256)
        coordinates = torch.randn(len(test_sequence), 3) * 10
        
        results = conf_cli.predict_confidence(single_repr, coordinates, test_sequence)
        
        print(f"\n📊 Confidence Test Results:")
        print(f"   Sequence length: {len(test_sequence)}")
        print(f"   Mean pLDDT: {results['statistics']['mean_plddt']:.1f}")
        print(f"   Confident residues: {results['statistics']['confident_pct']:.1f}%")
        print(f"   Prediction time: {results['prediction_time_s']:.3f}s")
        
        # Test output formatting
        output = conf_cli.format_confidence_output(results, args, test_sequence)
        print(f"\n📝 Sample Output (first 10 lines):")
        for line in output.split('\n')[:10]:
            print(f"   {line}")
    
    print(f"\n🎯 Confidence CLI Ready!")
    print(f"   Use --confidence flag to enable pLDDT prediction")
    print(f"   Per-residue confidence scores (0-100)")
