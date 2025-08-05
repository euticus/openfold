"""
Mutation Scanner for OdinFold

High-throughput mutation scanning and analysis for protein stability
and function prediction.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import pandas as pd

from .ddg_predictor import DDGPredictor, amino_acid_to_index, index_to_amino_acid

logger = logging.getLogger(__name__)


@dataclass
class MutationEffect:
    """Data class for storing mutation effect predictions."""

    position: int
    wt_aa: str
    mut_aa: str
    ddg_pred: float
    uncertainty: float
    confidence: float
    effect_category: str = ""

    def __post_init__(self):
        """Categorize mutation effect based on ΔΔG."""
        if self.ddg_pred > 2.0:
            self.effect_category = "Destabilizing"
        elif self.ddg_pred > 0.5:
            self.effect_category = "Mildly Destabilizing"
        elif self.ddg_pred > -0.5:
            self.effect_category = "Neutral"
        elif self.ddg_pred > -2.0:
            self.effect_category = "Mildly Stabilizing"
        else:
            self.effect_category = "Stabilizing"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'position': self.position,
            'wt_aa': self.wt_aa,
            'mut_aa': self.mut_aa,
            'mutation': f"{self.wt_aa}{self.position+1}{self.mut_aa}",
            'ddg_pred': self.ddg_pred,
            'uncertainty': self.uncertainty,
            'confidence': self.confidence,
            'effect_category': self.effect_category
        }


class MutationScanner:
    """
    High-throughput mutation scanner for protein stability analysis.
    
    Performs comprehensive mutation scanning using the ΔΔG predictor
    to identify stabilizing and destabilizing mutations.
    """
    
    def __init__(self, ddg_predictor: DDGPredictor):
        self.ddg_predictor = ddg_predictor
        
        # Amino acid groups for analysis
        self.aa_groups = {
            'hydrophobic': ['A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V'],
            'polar': ['S', 'T', 'N', 'Q'],
            'charged_positive': ['K', 'R', 'H'],
            'charged_negative': ['D', 'E'],
            'special': ['C', 'G', 'P']
        }
        
        # Conservation scores (simplified)
        self.conservation_weights = {
            'C': 0.9,  # Cysteine highly conserved
            'W': 0.8,  # Tryptophan conserved
            'P': 0.7,  # Proline structurally important
            'G': 0.6,  # Glycine flexible
            'default': 0.5
        }
    
    def scan_all_mutations(self,
                          structure_features: torch.Tensor,
                          sequence: str,
                          positions: Optional[List[int]] = None,
                          exclude_identity: bool = True) -> List[MutationEffect]:
        """
        Scan all possible single mutations.
        
        Args:
            structure_features: Structure features [seq_len, structure_dim]
            sequence: Wild-type sequence
            positions: Positions to scan (default: all)
            exclude_identity: Whether to exclude identity mutations
            
        Returns:
            List of mutation effects
        """
        
        if positions is None:
            positions = list(range(len(sequence)))
        
        # Convert sequence to indices
        wt_sequence = [amino_acid_to_index(aa) for aa in sequence]
        
        # Scan mutations
        scan_results = self.ddg_predictor.scan_mutations(
            structure_features, wt_sequence, positions
        )
        
        # Convert to MutationEffect objects
        mutation_effects = []
        mutations = scan_results['mutations']
        ddg_preds = scan_results['ddg_predictions'].cpu().numpy()
        uncertainties = scan_results['uncertainties'].cpu().numpy()
        confidences = scan_results['confidences'].cpu().numpy()
        
        for i, (wt_idx, mut_idx, pos) in enumerate(mutations):
            wt_aa = index_to_amino_acid(wt_idx)
            mut_aa = index_to_amino_acid(mut_idx)
            
            if exclude_identity and wt_aa == mut_aa:
                continue
            
            effect = MutationEffect(
                position=pos,
                wt_aa=wt_aa,
                mut_aa=mut_aa,
                ddg_pred=float(ddg_preds[i]),
                uncertainty=float(uncertainties[i]),
                confidence=float(confidences[i])
            )
            
            mutation_effects.append(effect)
        
        return mutation_effects
    
    def scan_position(self,
                     structure_features: torch.Tensor,
                     sequence: str,
                     position: int,
                     target_aa: Optional[List[str]] = None) -> List[MutationEffect]:
        """
        Scan all mutations at a specific position.
        
        Args:
            structure_features: Structure features [seq_len, structure_dim]
            sequence: Wild-type sequence
            position: Position to scan
            target_aa: Target amino acids (default: all 20)
            
        Returns:
            List of mutation effects at the position
        """
        
        if target_aa is None:
            target_aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                        'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        
        wt_aa = sequence[position]
        mutation_effects = []
        
        for mut_aa in target_aa:
            if mut_aa == wt_aa:
                continue  # Skip identity
            
            result = self.ddg_predictor.predict_single(
                structure_features,
                amino_acid_to_index(wt_aa),
                amino_acid_to_index(mut_aa),
                position
            )
            
            effect = MutationEffect(
                position=position,
                wt_aa=wt_aa,
                mut_aa=mut_aa,
                ddg_pred=result['ddg_pred'],
                uncertainty=result['uncertainty'],
                confidence=result['confidence']
            )
            
            mutation_effects.append(effect)
        
        # Sort by ΔΔG (most stabilizing first)
        mutation_effects.sort(key=lambda x: x.ddg_pred)
        
        return mutation_effects
    
    def find_stabilizing_mutations(self,
                                  structure_features: torch.Tensor,
                                  sequence: str,
                                  ddg_threshold: float = -0.5,
                                  confidence_threshold: float = 0.7,
                                  top_k: Optional[int] = None) -> List[MutationEffect]:
        """
        Find mutations that stabilize the protein.
        
        Args:
            structure_features: Structure features
            sequence: Wild-type sequence
            ddg_threshold: ΔΔG threshold for stabilization
            confidence_threshold: Minimum confidence threshold
            top_k: Return top k stabilizing mutations
            
        Returns:
            List of stabilizing mutations
        """
        
        all_mutations = self.scan_all_mutations(structure_features, sequence)
        
        # Filter stabilizing mutations
        stabilizing = [
            mut for mut in all_mutations
            if mut.ddg_pred < ddg_threshold and mut.confidence > confidence_threshold
        ]
        
        # Sort by ΔΔG (most stabilizing first)
        stabilizing.sort(key=lambda x: x.ddg_pred)
        
        if top_k is not None:
            stabilizing = stabilizing[:top_k]
        
        return stabilizing
    
    def find_destabilizing_mutations(self,
                                   structure_features: torch.Tensor,
                                   sequence: str,
                                   ddg_threshold: float = 2.0,
                                   confidence_threshold: float = 0.7,
                                   top_k: Optional[int] = None) -> List[MutationEffect]:
        """
        Find mutations that destabilize the protein.
        
        Args:
            structure_features: Structure features
            sequence: Wild-type sequence
            ddg_threshold: ΔΔG threshold for destabilization
            confidence_threshold: Minimum confidence threshold
            top_k: Return top k destabilizing mutations
            
        Returns:
            List of destabilizing mutations
        """
        
        all_mutations = self.scan_all_mutations(structure_features, sequence)
        
        # Filter destabilizing mutations
        destabilizing = [
            mut for mut in all_mutations
            if mut.ddg_pred > ddg_threshold and mut.confidence > confidence_threshold
        ]
        
        # Sort by ΔΔG (most destabilizing first)
        destabilizing.sort(key=lambda x: x.ddg_pred, reverse=True)
        
        if top_k is not None:
            destabilizing = destabilizing[:top_k]
        
        return destabilizing
    
    def analyze_position_sensitivity(self,
                                   structure_features: torch.Tensor,
                                   sequence: str) -> Dict[int, Dict]:
        """
        Analyze mutation sensitivity for each position.
        
        Args:
            structure_features: Structure features
            sequence: Wild-type sequence
            
        Returns:
            Dictionary with position sensitivity analysis
        """
        
        position_analysis = {}
        
        for position in range(len(sequence)):
            mutations = self.scan_position(structure_features, sequence, position)
            
            if not mutations:
                continue
            
            ddg_values = [mut.ddg_pred for mut in mutations]
            
            analysis = {
                'wt_aa': sequence[position],
                'num_mutations': len(mutations),
                'mean_ddg': np.mean(ddg_values),
                'std_ddg': np.std(ddg_values),
                'min_ddg': np.min(ddg_values),
                'max_ddg': np.max(ddg_values),
                'sensitivity_score': np.std(ddg_values),  # Higher std = more sensitive
                'conservation_weight': self.conservation_weights.get(
                    sequence[position], self.conservation_weights['default']
                ),
                'best_mutation': mutations[0].to_dict(),  # Most stabilizing
                'worst_mutation': mutations[-1].to_dict()  # Most destabilizing
            }
            
            position_analysis[position] = analysis
        
        return position_analysis
    
    def generate_mutation_report(self,
                               structure_features: torch.Tensor,
                               sequence: str,
                               protein_name: str = "Unknown") -> Dict:
        """
        Generate comprehensive mutation analysis report.
        
        Args:
            structure_features: Structure features
            sequence: Wild-type sequence
            protein_name: Name of the protein
            
        Returns:
            Comprehensive mutation report
        """
        
        logger.info(f"Generating mutation report for {protein_name}")
        
        # Scan all mutations
        all_mutations = self.scan_all_mutations(structure_features, sequence)
        
        # Find key mutations
        stabilizing = self.find_stabilizing_mutations(
            structure_features, sequence, top_k=10
        )
        destabilizing = self.find_destabilizing_mutations(
            structure_features, sequence, top_k=10
        )
        
        # Position sensitivity analysis
        position_analysis = self.analyze_position_sensitivity(
            structure_features, sequence
        )
        
        # Overall statistics
        ddg_values = [mut.ddg_pred for mut in all_mutations]
        confidence_values = [mut.confidence for mut in all_mutations]
        
        # Categorize mutations
        categories = {}
        for mut in all_mutations:
            cat = mut.effect_category
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        
        report = {
            'protein_name': protein_name,
            'sequence_length': len(sequence),
            'total_mutations_scanned': len(all_mutations),
            'statistics': {
                'mean_ddg': np.mean(ddg_values),
                'std_ddg': np.std(ddg_values),
                'mean_confidence': np.mean(confidence_values),
                'min_ddg': np.min(ddg_values),
                'max_ddg': np.max(ddg_values)
            },
            'mutation_categories': categories,
            'top_stabilizing': [mut.to_dict() for mut in stabilizing],
            'top_destabilizing': [mut.to_dict() for mut in destabilizing],
            'position_analysis': position_analysis,
            'recommendations': self._generate_recommendations(
                stabilizing, destabilizing, position_analysis
            )
        }
        
        return report
    
    def _generate_recommendations(self,
                                stabilizing: List[MutationEffect],
                                destabilizing: List[MutationEffect],
                                position_analysis: Dict) -> Dict:
        """Generate mutation recommendations."""
        
        recommendations = {
            'engineering_targets': [],
            'conservation_warnings': [],
            'stability_improvements': []
        }
        
        # Engineering targets (highly stabilizing mutations)
        for mut in stabilizing[:5]:
            if mut.ddg_pred < -1.0 and mut.confidence > 0.8:
                recommendations['engineering_targets'].append({
                    'mutation': f"{mut.wt_aa}{mut.position+1}{mut.mut_aa}",
                    'ddg_improvement': -mut.ddg_pred,
                    'confidence': mut.confidence
                })
        
        # Conservation warnings (sensitive positions)
        for pos, analysis in position_analysis.items():
            if analysis['sensitivity_score'] > 2.0 and analysis['conservation_weight'] > 0.7:
                recommendations['conservation_warnings'].append({
                    'position': pos + 1,
                    'wt_aa': analysis['wt_aa'],
                    'sensitivity_score': analysis['sensitivity_score'],
                    'reason': 'Highly sensitive and conserved position'
                })
        
        # Stability improvements
        if stabilizing:
            total_improvement = sum(-mut.ddg_pred for mut in stabilizing[:3])
            recommendations['stability_improvements'].append({
                'strategy': 'Top 3 stabilizing mutations',
                'estimated_improvement': total_improvement,
                'mutations': [f"{mut.wt_aa}{mut.position+1}{mut.mut_aa}" for mut in stabilizing[:3]]
            })
        
        return recommendations
    
    def export_to_dataframe(self, mutations: List[MutationEffect]) -> pd.DataFrame:
        """Export mutation effects to pandas DataFrame."""
        
        data = [mut.to_dict() for mut in mutations]
        return pd.DataFrame(data)
