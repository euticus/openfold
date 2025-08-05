"""
Multimer Fold Runner for OdinFold

High-level interface for folding multi-chain protein complexes
with proper chain handling and validation.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import logging

from .chain_processing import MultimerDataProcessor, parse_multimer_fasta
from .attention_utils import MultimerAttentionMask, InterChainAttention


logger = logging.getLogger(__name__)


class MultimerFoldRunner:
    """
    High-level runner for multimer protein folding.
    
    Handles complete pipeline from multi-FASTA input to folded complex
    with proper chain separation and interface prediction.
    """
    
    def __init__(self, model, device: str = "cuda", d_model: int = 256):
        self.model = model
        self.device = torch.device(device)
        self.d_model = d_model
        
        # Initialize processors
        self.data_processor = MultimerDataProcessor(d_model=d_model)
        self.attention_mask = MultimerAttentionMask(mask_inter_chain=False)
        
        # Move model to device
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
        
        logger.info(f"MultimerFoldRunner initialized on {self.device}")
    
    def fold_multimer_from_sequences(self, sequences: List[str], 
                                   chain_ids: Optional[List[str]] = None,
                                   **kwargs) -> Dict:
        """
        Fold a multimer from a list of sequences.
        
        Args:
            sequences: List of amino acid sequences
            chain_ids: Optional chain identifiers
            **kwargs: Additional folding parameters
            
        Returns:
            Folding results with chain-separated coordinates
        """
        
        logger.info(f"Folding multimer with {len(sequences)} chains")
        
        # Validate input
        validation = self.data_processor.validate_multimer_input(sequences)
        if not validation['overall_valid']:
            raise ValueError(f"Invalid multimer input: {validation['details']}")
        
        # Process multimer data
        multimer_data = self.data_processor.process_multimer_input(sequences, chain_ids)
        
        # Run folding
        results = self._run_multimer_folding(multimer_data, **kwargs)
        
        # Post-process results
        processed_results = self._post_process_multimer_results(results, multimer_data)
        
        logger.info("Multimer folding completed successfully")
        
        return processed_results
    
    def fold_multimer_from_fasta(self, fasta_path: Union[str, Path], **kwargs) -> Dict:
        """
        Fold a multimer from a multi-FASTA file.
        
        Args:
            fasta_path: Path to multi-FASTA file
            **kwargs: Additional folding parameters
            
        Returns:
            Folding results with chain-separated coordinates
        """
        
        fasta_path = Path(fasta_path)
        
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        
        # Parse FASTA file
        with open(fasta_path, 'r') as f:
            fasta_content = f.read()
        
        chains = parse_multimer_fasta(fasta_content)
        
        if len(chains) < 2:
            raise ValueError("Multi-FASTA file must contain at least 2 chains")
        
        # Extract sequences and chain IDs
        chain_ids, sequences = zip(*chains)
        
        logger.info(f"Loaded {len(sequences)} chains from {fasta_path}")
        
        return self.fold_multimer_from_sequences(list(sequences), list(chain_ids), **kwargs)
    
    def _run_multimer_folding(self, multimer_data: Dict, **kwargs) -> Dict:
        """
        Run the actual multimer folding process.
        
        Args:
            multimer_data: Processed multimer data
            **kwargs: Additional parameters
            
        Returns:
            Raw folding results
        """
        
        # Extract data
        sequence = multimer_data['concatenated_sequence']
        chain_id_tensor = multimer_data['chain_id_tensor'].to(self.device)
        chain_break_mask = multimer_data['chain_break_mask'].to(self.device)
        
        # Create attention mask
        attention_mask = self.attention_mask.create_multimer_mask(
            chain_id_tensor, chain_break_mask
        )
        
        # Mock folding process (replace with actual model inference)
        results = self._mock_multimer_folding(
            sequence, chain_id_tensor, attention_mask, **kwargs
        )
        
        return results
    
    def _mock_multimer_folding(self, sequence: str, chain_id_tensor: torch.Tensor,
                             attention_mask: torch.Tensor, **kwargs) -> Dict:
        """
        Mock multimer folding for testing.
        
        In production, this would call the actual OdinFold model.
        """
        
        seq_len = len(sequence)
        
        # Generate mock coordinates
        coordinates = torch.zeros(seq_len, 3, device=self.device)
        
        # Create realistic multimer structure
        chain_ids = chain_id_tensor.cpu().numpy()
        unique_chains = np.unique(chain_ids)
        
        for i, chain_id in enumerate(unique_chains):
            chain_mask = chain_ids == chain_id
            chain_positions = np.where(chain_mask)[0]
            
            # Position each chain in 3D space
            chain_center = np.array([i * 20.0, 0.0, 0.0])  # Separate chains
            
            for j, pos in enumerate(chain_positions):
                # Create helix-like structure for each chain
                x = chain_center[0] + j * 1.5 + np.random.normal(0, 0.5)
                y = chain_center[1] + 3 * np.sin(j * 0.3) + np.random.normal(0, 0.5)
                z = chain_center[2] + 3 * np.cos(j * 0.3) + np.random.normal(0, 0.5)
                
                coordinates[pos] = torch.tensor([x, y, z], device=self.device)
        
        # Generate mock confidence scores
        confidence_scores = torch.rand(seq_len, device=self.device) * 100
        
        # Mock interface prediction
        interface_residues = self._predict_interface_residues(coordinates, chain_id_tensor)
        
        # Calculate mock metrics
        tm_score = np.random.normal(0.75, 0.1)  # Higher for multimers
        tm_score = float(np.clip(tm_score, 0.4, 0.95))
        
        return {
            'coordinates': coordinates,
            'confidence_scores': confidence_scores,
            'interface_residues': interface_residues,
            'tm_score_estimate': tm_score,
            'chain_id_tensor': chain_id_tensor,
            'attention_mask': attention_mask
        }
    
    def _predict_interface_residues(self, coordinates: torch.Tensor, 
                                  chain_id_tensor: torch.Tensor,
                                  cutoff: float = 8.0) -> Dict[str, List[int]]:
        """
        Predict interface residues between chains.
        
        Args:
            coordinates: 3D coordinates [seq_len, 3]
            chain_id_tensor: Chain IDs [seq_len]
            cutoff: Distance cutoff for interface definition
            
        Returns:
            Dictionary mapping chain pairs to interface residue indices
        """
        
        # Compute pairwise distances
        distances = torch.cdist(coordinates, coordinates)
        
        # Find inter-chain contacts
        chain_matrix = chain_id_tensor.unsqueeze(0) != chain_id_tensor.unsqueeze(1)
        inter_chain_contacts = chain_matrix & (distances < cutoff)
        
        # Extract interface residues
        interface_residues = {}
        chain_ids = chain_id_tensor.cpu().numpy()
        unique_chains = np.unique(chain_ids)
        
        for i, chain_a in enumerate(unique_chains):
            for j, chain_b in enumerate(unique_chains):
                if i >= j:  # Avoid duplicates
                    continue
                
                chain_a_mask = chain_ids == chain_a
                chain_b_mask = chain_ids == chain_b
                
                # Find contacts between these chains
                contacts = inter_chain_contacts.cpu().numpy()
                
                chain_a_interface = []
                chain_b_interface = []
                
                for pos_a in np.where(chain_a_mask)[0]:
                    if np.any(contacts[pos_a, chain_b_mask]):
                        chain_a_interface.append(int(pos_a))
                
                for pos_b in np.where(chain_b_mask)[0]:
                    if np.any(contacts[pos_b, chain_a_mask]):
                        chain_b_interface.append(int(pos_b))
                
                if chain_a_interface or chain_b_interface:
                    interface_key = f"chain_{chain_a}_chain_{chain_b}"
                    interface_residues[interface_key] = {
                        f'chain_{chain_a}': chain_a_interface,
                        f'chain_{chain_b}': chain_b_interface
                    }
        
        return interface_residues
    
    def _post_process_multimer_results(self, results: Dict, multimer_data: Dict) -> Dict:
        """
        Post-process multimer folding results.
        
        Args:
            results: Raw folding results
            multimer_data: Original multimer data
            
        Returns:
            Processed results with chain separation
        """
        
        # Split coordinates by chain
        chain_coordinates = self.data_processor.chain_processor.split_coordinates(
            results['coordinates'], multimer_data['chain_infos']
        )
        
        # Split confidence scores by chain
        chain_confidences = {}
        if results['confidence_scores'] is not None:
            for chain_info in multimer_data['chain_infos']:
                start_idx = chain_info.start_pos
                end_idx = chain_info.end_pos + 1
                chain_confidences[chain_info.chain_id] = results['confidence_scores'][start_idx:end_idx]
        
        # Calculate chain-specific metrics
        chain_metrics = {}
        for chain_info in multimer_data['chain_infos']:
            chain_coords = chain_coordinates[chain_info.chain_id]
            chain_conf = chain_confidences.get(chain_info.chain_id)
            
            chain_metrics[chain_info.chain_id] = {
                'length': chain_info.length,
                'mean_confidence': float(chain_conf.mean()) if chain_conf is not None else None,
                'rmsd_estimate': float(torch.std(chain_coords).mean()),  # Mock RMSD
                'secondary_structure': self._predict_secondary_structure(chain_coords)
            }
        
        # Compile final results
        processed_results = {
            'multimer_info': {
                'num_chains': multimer_data['num_chains'],
                'total_length': multimer_data['total_length'],
                'chain_ids': [info.chain_id for info in multimer_data['chain_infos']],
                'chain_lengths': [info.length for info in multimer_data['chain_infos']]
            },
            'coordinates': {
                'full_complex': results['coordinates'].cpu().numpy(),
                'by_chain': {k: v.cpu().numpy() for k, v in chain_coordinates.items()}
            },
            'confidence_scores': {
                'full_complex': results['confidence_scores'].cpu().numpy() if results['confidence_scores'] is not None else None,
                'by_chain': {k: v.cpu().numpy() for k, v in chain_confidences.items()}
            },
            'interface_analysis': results['interface_residues'],
            'chain_metrics': chain_metrics,
            'overall_metrics': {
                'tm_score_estimate': results['tm_score_estimate'],
                'interface_quality': self._assess_interface_quality(results['interface_residues']),
                'complex_compactness': self._calculate_compactness(results['coordinates'])
            }
        }
        
        return processed_results
    
    def _predict_secondary_structure(self, coordinates: torch.Tensor) -> Dict[str, float]:
        """Mock secondary structure prediction."""
        
        # Simple mock based on coordinate patterns
        return {
            'helix_content': float(np.random.uniform(0.2, 0.6)),
            'sheet_content': float(np.random.uniform(0.1, 0.4)),
            'loop_content': float(np.random.uniform(0.2, 0.5))
        }
    
    def _assess_interface_quality(self, interface_residues: Dict) -> float:
        """Assess the quality of predicted interfaces."""
        
        if not interface_residues:
            return 0.0
        
        # Mock interface quality score
        total_interface_residues = sum(
            len(interface['chain_0']) + len(interface['chain_1'])
            for interface in interface_residues.values()
            if isinstance(interface, dict)
        )
        
        # Normalize by expected interface size
        quality_score = min(1.0, total_interface_residues / 20.0)
        
        return float(quality_score)
    
    def _calculate_compactness(self, coordinates: torch.Tensor) -> float:
        """Calculate complex compactness metric."""
        
        # Calculate radius of gyration
        center_of_mass = coordinates.mean(dim=0)
        distances_from_com = torch.norm(coordinates - center_of_mass, dim=1)
        radius_of_gyration = torch.sqrt(torch.mean(distances_from_com ** 2))
        
        # Normalize by sequence length
        compactness = 1.0 / (1.0 + radius_of_gyration / len(coordinates))
        
        return float(compactness)
