"""
Pose Analysis for OdinFold Docking

Analysis and visualization of molecular docking poses,
binding affinity prediction, and interaction analysis.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DockingPoseAnalyzer:
    """
    Analyzer for molecular docking poses and binding interactions.
    
    Provides comprehensive analysis of docking results including
    binding affinity prediction, interaction mapping, and pose quality assessment.
    """
    
    def __init__(self):
        self.interaction_cutoffs = {
            'hydrogen_bond': 3.5,
            'hydrophobic': 4.0,
            'electrostatic': 5.0,
            'van_der_waals': 4.5
        }
    
    def analyze_pose(self, protein_coords: torch.Tensor,
                    ligand_coords: torch.Tensor,
                    protein_sequence: str,
                    docking_score: float,
                    pose_id: int = 1) -> Dict:
        """
        Comprehensive analysis of a single docking pose.
        
        Args:
            protein_coords: Protein coordinates [seq_len, 3]
            ligand_coords: Ligand coordinates [num_atoms, 3]
            protein_sequence: Amino acid sequence
            docking_score: Docking score (e.g., Vina score)
            pose_id: Pose identifier
            
        Returns:
            Comprehensive pose analysis
        """
        
        logger.info(f"Analyzing docking pose {pose_id}")
        
        # Basic geometric analysis
        geometric_analysis = self._analyze_geometry(protein_coords, ligand_coords)
        
        # Interaction analysis
        interactions = self._analyze_interactions(
            protein_coords, ligand_coords, protein_sequence
        )
        
        # Binding site analysis
        binding_site = self._analyze_binding_site(
            protein_coords, ligand_coords, protein_sequence
        )
        
        # Pose quality assessment
        quality_metrics = self._assess_pose_quality(
            protein_coords, ligand_coords, docking_score
        )
        
        # Predicted binding affinity
        predicted_affinity = self._predict_binding_affinity(
            interactions, geometric_analysis, docking_score
        )
        
        analysis = {
            'pose_id': pose_id,
            'docking_score': docking_score,
            'predicted_affinity': predicted_affinity,
            'geometric_analysis': geometric_analysis,
            'interactions': interactions,
            'binding_site': binding_site,
            'quality_metrics': quality_metrics,
            'summary': self._generate_summary(
                docking_score, predicted_affinity, interactions, quality_metrics
            )
        }
        
        return analysis
    
    def _analyze_geometry(self, protein_coords: torch.Tensor, 
                         ligand_coords: torch.Tensor) -> Dict:
        """Analyze geometric properties of the binding pose."""
        
        protein_np = protein_coords.detach().cpu().numpy()
        ligand_np = ligand_coords.detach().cpu().numpy()
        
        # Calculate centers
        protein_center = protein_np.mean(axis=0)
        ligand_center = ligand_np.mean(axis=0)
        
        # Calculate spans
        protein_span = protein_np.max(axis=0) - protein_np.min(axis=0)
        ligand_span = ligand_np.max(axis=0) - ligand_np.min(axis=0)
        
        # Distance between centers
        center_distance = np.linalg.norm(protein_center - ligand_center)
        
        # Minimum distance between protein and ligand
        distances = np.linalg.norm(
            protein_np[:, np.newaxis, :] - ligand_np[np.newaxis, :, :], 
            axis=2
        )
        min_distance = distances.min()
        
        # Contact surface area (approximate)
        contact_cutoff = 4.0
        contacts = distances < contact_cutoff
        contact_area = contacts.sum() * 1.0  # Approximate area
        
        return {
            'protein_center': protein_center.tolist(),
            'ligand_center': ligand_center.tolist(),
            'protein_span': protein_span.tolist(),
            'ligand_span': ligand_span.tolist(),
            'center_distance': float(center_distance),
            'min_distance': float(min_distance),
            'contact_area': float(contact_area),
            'ligand_volume': float(np.prod(ligand_span))
        }
    
    def _analyze_interactions(self, protein_coords: torch.Tensor,
                            ligand_coords: torch.Tensor,
                            protein_sequence: str) -> Dict:
        """Analyze protein-ligand interactions."""
        
        protein_np = protein_coords.detach().cpu().numpy()
        ligand_np = ligand_coords.detach().cpu().numpy()
        
        # Calculate all pairwise distances
        distances = np.linalg.norm(
            protein_np[:, np.newaxis, :] - ligand_np[np.newaxis, :, :], 
            axis=2
        )
        
        interactions = {
            'hydrogen_bonds': [],
            'hydrophobic_contacts': [],
            'electrostatic_interactions': [],
            'van_der_waals_contacts': []
        }
        
        # Analyze each residue
        for res_idx, (res_coord, aa) in enumerate(zip(protein_np, protein_sequence)):
            min_dist_to_ligand = distances[res_idx].min()
            closest_ligand_atom = distances[res_idx].argmin()
            
            # Hydrogen bonds (polar residues)
            if aa in 'STNQHKRDE' and min_dist_to_ligand <= self.interaction_cutoffs['hydrogen_bond']:
                interactions['hydrogen_bonds'].append({
                    'residue_index': res_idx,
                    'residue_type': aa,
                    'ligand_atom': int(closest_ligand_atom),
                    'distance': float(min_dist_to_ligand)
                })
            
            # Hydrophobic contacts
            if aa in 'AILMFPWYV' and min_dist_to_ligand <= self.interaction_cutoffs['hydrophobic']:
                interactions['hydrophobic_contacts'].append({
                    'residue_index': res_idx,
                    'residue_type': aa,
                    'ligand_atom': int(closest_ligand_atom),
                    'distance': float(min_dist_to_ligand)
                })
            
            # Electrostatic interactions (charged residues)
            if aa in 'KRHDE' and min_dist_to_ligand <= self.interaction_cutoffs['electrostatic']:
                interactions['electrostatic_interactions'].append({
                    'residue_index': res_idx,
                    'residue_type': aa,
                    'ligand_atom': int(closest_ligand_atom),
                    'distance': float(min_dist_to_ligand)
                })
            
            # Van der Waals contacts
            if min_dist_to_ligand <= self.interaction_cutoffs['van_der_waals']:
                interactions['van_der_waals_contacts'].append({
                    'residue_index': res_idx,
                    'residue_type': aa,
                    'ligand_atom': int(closest_ligand_atom),
                    'distance': float(min_dist_to_ligand)
                })
        
        # Calculate interaction statistics
        interaction_stats = {
            'total_interactions': sum(len(interactions[key]) for key in interactions),
            'hydrogen_bond_count': len(interactions['hydrogen_bonds']),
            'hydrophobic_contact_count': len(interactions['hydrophobic_contacts']),
            'electrostatic_count': len(interactions['electrostatic_interactions']),
            'vdw_contact_count': len(interactions['van_der_waals_contacts'])
        }
        
        interactions['statistics'] = interaction_stats
        
        return interactions
    
    def _analyze_binding_site(self, protein_coords: torch.Tensor,
                            ligand_coords: torch.Tensor,
                            protein_sequence: str,
                            cutoff: float = 5.0) -> Dict:
        """Analyze binding site properties."""
        
        protein_np = protein_coords.detach().cpu().numpy()
        ligand_np = ligand_coords.detach().cpu().numpy()
        
        # Find binding site residues
        distances = np.linalg.norm(
            protein_np[:, np.newaxis, :] - ligand_np[np.newaxis, :, :], 
            axis=2
        )
        min_distances = distances.min(axis=1)
        binding_site_mask = min_distances <= cutoff
        binding_site_indices = np.where(binding_site_mask)[0]
        
        # Analyze binding site composition
        binding_site_residues = [protein_sequence[i] for i in binding_site_indices]
        
        # Calculate amino acid composition
        aa_composition = {}
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            count = binding_site_residues.count(aa)
            aa_composition[aa] = count
        
        # Binding site properties
        if len(binding_site_indices) > 0:
            binding_site_coords = protein_np[binding_site_indices]
            binding_site_center = binding_site_coords.mean(axis=0)
            binding_site_span = binding_site_coords.max(axis=0) - binding_site_coords.min(axis=0)
        else:
            binding_site_center = np.array([0.0, 0.0, 0.0])
            binding_site_span = np.array([0.0, 0.0, 0.0])
        
        # Calculate binding site properties
        hydrophobic_residues = sum(aa_composition[aa] for aa in 'AILMFPWYV')
        polar_residues = sum(aa_composition[aa] for aa in 'STNQH')
        charged_residues = sum(aa_composition[aa] for aa in 'KRHDE')
        
        total_residues = len(binding_site_residues)
        
        return {
            'binding_site_residues': binding_site_indices.tolist(),
            'binding_site_sequence': ''.join(binding_site_residues),
            'binding_site_size': total_residues,
            'binding_site_center': binding_site_center.tolist(),
            'binding_site_span': binding_site_span.tolist(),
            'aa_composition': aa_composition,
            'hydrophobic_fraction': hydrophobic_residues / max(total_residues, 1),
            'polar_fraction': polar_residues / max(total_residues, 1),
            'charged_fraction': charged_residues / max(total_residues, 1)
        }
    
    def _assess_pose_quality(self, protein_coords: torch.Tensor,
                           ligand_coords: torch.Tensor,
                           docking_score: float) -> Dict:
        """Assess the quality of the docking pose."""
        
        protein_np = protein_coords.detach().cpu().numpy()
        ligand_np = ligand_coords.detach().cpu().numpy()
        
        # Calculate quality metrics
        distances = np.linalg.norm(
            protein_np[:, np.newaxis, :] - ligand_np[np.newaxis, :, :], 
            axis=2
        )
        min_distance = distances.min()
        
        # Check for clashes (too close contacts)
        clash_cutoff = 2.0
        clashes = (distances < clash_cutoff).sum()
        
        # Check for reasonable contacts
        contact_cutoff = 4.0
        contacts = (distances < contact_cutoff).sum()
        
        # Buried surface area (approximate)
        buried_atoms = (distances < 3.5).sum()
        
        # Overall quality score
        quality_score = 0.0
        
        # Penalize clashes
        quality_score -= clashes * 0.5
        
        # Reward contacts
        quality_score += min(contacts * 0.1, 2.0)
        
        # Reward good docking score
        if docking_score < -8.0:
            quality_score += 1.0
        elif docking_score < -6.0:
            quality_score += 0.5
        
        # Normalize to 0-1 range
        quality_score = max(0.0, min(1.0, (quality_score + 2.0) / 4.0))
        
        return {
            'quality_score': float(quality_score),
            'min_distance': float(min_distance),
            'clash_count': int(clashes),
            'contact_count': int(contacts),
            'buried_atoms': int(buried_atoms),
            'has_clashes': clashes > 0,
            'well_contacted': contacts > 10
        }
    
    def _predict_binding_affinity(self, interactions: Dict,
                                geometric_analysis: Dict,
                                docking_score: float) -> Dict:
        """Predict binding affinity using simple empirical model."""
        
        # Extract features
        hbond_count = interactions['statistics']['hydrogen_bond_count']
        hydrophobic_count = interactions['statistics']['hydrophobic_contact_count']
        electrostatic_count = interactions['statistics']['electrostatic_count']
        contact_area = geometric_analysis['contact_area']
        
        # Simple empirical model (placeholder)
        # In practice, you'd use a trained ML model
        
        affinity_score = 0.0
        
        # Hydrogen bonds contribute significantly
        affinity_score += hbond_count * 1.5
        
        # Hydrophobic contacts
        affinity_score += hydrophobic_count * 0.8
        
        # Electrostatic interactions
        affinity_score += electrostatic_count * 1.2
        
        # Contact area
        affinity_score += contact_area * 0.01
        
        # Convert to approximate binding affinity (kcal/mol)
        predicted_kd = -affinity_score - 2.0  # Rough conversion
        
        # Confidence based on number of interactions
        total_interactions = interactions['statistics']['total_interactions']
        confidence = min(1.0, total_interactions / 20.0)
        
        return {
            'predicted_affinity_kcal_mol': float(predicted_kd),
            'affinity_score': float(affinity_score),
            'confidence': float(confidence),
            'docking_score': docking_score,
            'method': 'empirical_model'
        }
    
    def _generate_summary(self, docking_score: float, predicted_affinity: Dict,
                         interactions: Dict, quality_metrics: Dict) -> Dict:
        """Generate a summary of the pose analysis."""
        
        # Determine binding strength
        if docking_score < -10.0:
            binding_strength = "Very Strong"
        elif docking_score < -8.0:
            binding_strength = "Strong"
        elif docking_score < -6.0:
            binding_strength = "Moderate"
        else:
            binding_strength = "Weak"
        
        # Determine pose quality
        quality_score = quality_metrics['quality_score']
        if quality_score > 0.8:
            pose_quality = "Excellent"
        elif quality_score > 0.6:
            pose_quality = "Good"
        elif quality_score > 0.4:
            pose_quality = "Fair"
        else:
            pose_quality = "Poor"
        
        # Key interactions
        key_interactions = []
        if interactions['statistics']['hydrogen_bond_count'] > 0:
            key_interactions.append(f"{interactions['statistics']['hydrogen_bond_count']} hydrogen bonds")
        if interactions['statistics']['hydrophobic_contact_count'] > 0:
            key_interactions.append(f"{interactions['statistics']['hydrophobic_contact_count']} hydrophobic contacts")
        if interactions['statistics']['electrostatic_count'] > 0:
            key_interactions.append(f"{interactions['statistics']['electrostatic_count']} electrostatic interactions")
        
        return {
            'binding_strength': binding_strength,
            'pose_quality': pose_quality,
            'key_interactions': key_interactions,
            'total_interactions': interactions['statistics']['total_interactions'],
            'recommended': quality_score > 0.6 and docking_score < -6.0
        }


def calculate_binding_affinity(docking_results: List[Dict]) -> Dict:
    """
    Calculate consensus binding affinity from multiple docking poses.
    
    Args:
        docking_results: List of docking result dictionaries
        
    Returns:
        Consensus binding affinity analysis
    """
    
    if not docking_results:
        return {'error': 'No docking results provided'}
    
    # Extract scores
    vina_scores = [r.get('best_score', 0) for r in docking_results if r.get('success', False)]
    cnn_scores = [r.get('best_cnn_score', 0) for r in docking_results if r.get('best_cnn_score') is not None]
    
    if not vina_scores:
        return {'error': 'No successful docking results'}
    
    # Calculate consensus metrics
    consensus_vina = np.mean(vina_scores)
    best_vina = min(vina_scores)
    std_vina = np.std(vina_scores)
    
    consensus_cnn = np.mean(cnn_scores) if cnn_scores else None
    best_cnn = max(cnn_scores) if cnn_scores else None
    
    # Estimate binding affinity
    # Simple conversion: Vina score ≈ binding affinity
    estimated_affinity = best_vina
    
    # Confidence based on consistency
    confidence = 1.0 / (1.0 + std_vina)
    
    return {
        'consensus_vina_score': float(consensus_vina),
        'best_vina_score': float(best_vina),
        'vina_std': float(std_vina),
        'consensus_cnn_score': float(consensus_cnn) if consensus_cnn else None,
        'best_cnn_score': float(best_cnn) if best_cnn else None,
        'estimated_affinity_kcal_mol': float(estimated_affinity),
        'confidence': float(confidence),
        'num_poses': len(vina_scores)
    }
