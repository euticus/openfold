"""
Chain Processing for OdinFold Multimer

Handles chain break logic, positional encodings, and sequence concatenation
for multi-chain protein complex folding.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ChainInfo:
    """Information about a single chain in a multimer."""
    
    chain_id: str
    sequence: str
    start_pos: int
    end_pos: int
    length: int


class ChainBreakProcessor:
    """
    Processes multiple protein chains for multimer folding.
    
    Handles chain concatenation, break tokens, and chain ID assignment.
    """
    
    def __init__(self, chain_break_token: str = "|"):
        self.chain_break_token = chain_break_token
        self.chain_break_token_id = 21  # Standard amino acid vocab is 0-20
    
    def process_chains(self, sequences: List[str], chain_ids: Optional[List[str]] = None) -> Dict:
        """
        Process multiple chains into a single concatenated sequence.
        
        Args:
            sequences: List of amino acid sequences
            chain_ids: Optional list of chain identifiers (A, B, C, etc.)
            
        Returns:
            Dictionary containing processed multimer information
        """
        
        if chain_ids is None:
            chain_ids = [chr(65 + i) for i in range(len(sequences))]  # A, B, C, ...
        
        if len(sequences) != len(chain_ids):
            raise ValueError("Number of sequences must match number of chain IDs")
        
        # Concatenate sequences with chain breaks
        concatenated_sequence = ""
        chain_infos = []
        current_pos = 0
        
        for i, (seq, chain_id) in enumerate(zip(sequences, chain_ids)):
            # Validate sequence
            valid_amino_acids = set('ABCDEFGHIKLMNPQRSTVWYZ')  # Include B, Z for extended alphabet
            if not all(aa in valid_amino_acids for aa in seq.upper()):
                raise ValueError(f"Invalid amino acids in chain {chain_id}")
            
            # Add chain info
            chain_info = ChainInfo(
                chain_id=chain_id,
                sequence=seq,
                start_pos=current_pos,
                end_pos=current_pos + len(seq) - 1,
                length=len(seq)
            )
            chain_infos.append(chain_info)
            
            # Add sequence
            concatenated_sequence += seq
            current_pos += len(seq)
            
            # Add chain break token (except for last chain)
            if i < len(sequences) - 1:
                concatenated_sequence += self.chain_break_token
                current_pos += 1
        
        # Create chain ID tensor
        chain_id_tensor = self._create_chain_id_tensor(chain_infos, len(concatenated_sequence))
        
        # Create chain break mask
        chain_break_mask = self._create_chain_break_mask(concatenated_sequence)
        
        return {
            'concatenated_sequence': concatenated_sequence,
            'chain_infos': chain_infos,
            'chain_id_tensor': chain_id_tensor,
            'chain_break_mask': chain_break_mask,
            'num_chains': len(sequences),
            'total_length': len(concatenated_sequence)
        }
    
    def _create_chain_id_tensor(self, chain_infos: List[ChainInfo], total_length: int) -> torch.Tensor:
        """Create tensor indicating which chain each position belongs to."""
        
        chain_ids = torch.zeros(total_length, dtype=torch.long)
        
        for i, chain_info in enumerate(chain_infos):
            chain_ids[chain_info.start_pos:chain_info.end_pos + 1] = i
        
        return chain_ids
    
    def _create_chain_break_mask(self, sequence: str) -> torch.Tensor:
        """Create mask indicating chain break positions."""
        
        mask = torch.zeros(len(sequence), dtype=torch.bool)
        
        for i, token in enumerate(sequence):
            if token == self.chain_break_token:
                mask[i] = True
        
        return mask
    
    def split_coordinates(self, coordinates: torch.Tensor, chain_infos: List[ChainInfo]) -> Dict[str, torch.Tensor]:
        """Split predicted coordinates back into individual chains."""
        
        chain_coordinates = {}
        
        for chain_info in chain_infos:
            start_idx = chain_info.start_pos
            end_idx = chain_info.end_pos + 1
            chain_coordinates[chain_info.chain_id] = coordinates[start_idx:end_idx]
        
        return chain_coordinates


class MultimerPositionalEncoding(nn.Module):
    """
    Positional encoding for multimer sequences.
    
    Handles chain-aware positional encoding with proper offsets
    and relative position encoding between chains.
    """
    
    def __init__(self, d_model: int, max_len: int = 2048, chain_offset: int = 1000):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.chain_offset = chain_offset
        
        # Standard positional encoding
        self.pos_encoding = self._create_positional_encoding(d_model, max_len)
        
        # Chain embedding
        self.chain_embedding = nn.Embedding(32, d_model)  # Support up to 32 chains
        
        # Relative position embedding for inter-chain interactions
        self.relative_pos_embedding = nn.Embedding(2 * max_len + 1, d_model)
    
    def _create_positional_encoding(self, d_model: int, max_len: int) -> torch.Tensor:
        """Create standard sinusoidal positional encoding."""
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x: torch.Tensor, chain_id_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply multimer-aware positional encoding.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            chain_id_tensor: Chain ID for each position [seq_len]
            
        Returns:
            Encoded tensor with positional and chain information
        """
        
        batch_size, seq_len, d_model = x.shape
        
        # Standard positional encoding with chain offset
        positions = torch.arange(seq_len, device=x.device)
        chain_offsets = chain_id_tensor * self.chain_offset
        adjusted_positions = positions + chain_offsets
        
        # Clamp to valid range
        adjusted_positions = torch.clamp(adjusted_positions, 0, self.max_len - 1)
        
        pos_encoding = self.pos_encoding[adjusted_positions].to(x.device)
        
        # Chain embedding
        chain_encoding = self.chain_embedding(chain_id_tensor)
        
        # Combine encodings
        encoded = x + pos_encoding.unsqueeze(0) + chain_encoding.unsqueeze(0)
        
        return encoded
    
    def get_relative_position_encoding(self, seq_len: int, chain_id_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get relative position encoding for attention bias.
        
        Returns:
            Relative position encoding [seq_len, seq_len, d_model]
        """
        
        # Create relative position matrix
        positions = torch.arange(seq_len, device=chain_id_tensor.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Adjust for different chains (large offset for inter-chain)
        chain_matrix = chain_id_tensor.unsqueeze(0) == chain_id_tensor.unsqueeze(1)
        inter_chain_mask = ~chain_matrix
        
        # Apply large offset for inter-chain positions
        rel_pos = torch.where(inter_chain_mask, rel_pos + self.max_len, rel_pos)
        
        # Clamp to valid range
        rel_pos = torch.clamp(rel_pos + self.max_len, 0, 2 * self.max_len)
        
        # Get embeddings
        rel_pos_encoding = self.relative_pos_embedding(rel_pos)
        
        return rel_pos_encoding


class MultimerDataProcessor:
    """
    High-level processor for multimer data preparation.
    
    Combines chain processing and positional encoding for complete
    multimer data preparation pipeline.
    """
    
    def __init__(self, d_model: int = 256, max_len: int = 2048):
        self.chain_processor = ChainBreakProcessor()
        self.pos_encoder = MultimerPositionalEncoding(d_model, max_len)
    
    def process_multimer_input(self, sequences: List[str], chain_ids: Optional[List[str]] = None) -> Dict:
        """
        Complete processing pipeline for multimer input.
        
        Args:
            sequences: List of amino acid sequences
            chain_ids: Optional chain identifiers
            
        Returns:
            Complete multimer processing results
        """
        
        # Process chains
        chain_data = self.chain_processor.process_chains(sequences, chain_ids)
        
        # Add processed data
        result = {
            **chain_data,
            'pos_encoder': self.pos_encoder,
            'ready_for_folding': True
        }
        
        return result
    
    def validate_multimer_input(self, sequences: List[str]) -> Dict[str, bool]:
        """
        Validate multimer input sequences.
        
        Returns:
            Validation results with detailed checks
        """
        
        validation = {
            'valid_sequences': True,
            'reasonable_lengths': True,
            'total_length_ok': True,
            'num_chains_ok': True,
            'details': {}
        }
        
        # Check individual sequences
        valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        for i, seq in enumerate(sequences):
            seq_valid = all(aa in valid_amino_acids for aa in seq.upper())
            validation['details'][f'chain_{i}_valid'] = seq_valid
            validation['details'][f'chain_{i}_length'] = len(seq)
            
            if not seq_valid:
                validation['valid_sequences'] = False
            
            if len(seq) < 10 or len(seq) > 1000:
                validation['reasonable_lengths'] = False
        
        # Check total length
        total_length = sum(len(seq) for seq in sequences) + len(sequences) - 1  # Include chain breaks
        validation['details']['total_length'] = total_length
        
        if total_length > 2048:
            validation['total_length_ok'] = False
        
        # Check number of chains
        validation['details']['num_chains'] = len(sequences)
        if len(sequences) < 2 or len(sequences) > 10:
            validation['num_chains_ok'] = False
        
        # Overall validation
        validation['overall_valid'] = all([
            validation['valid_sequences'],
            validation['reasonable_lengths'],
            validation['total_length_ok'],
            validation['num_chains_ok']
        ])
        
        return validation


def parse_multimer_fasta(fasta_content: str) -> List[Tuple[str, str]]:
    """
    Parse multi-FASTA content into chain sequences.

    Args:
        fasta_content: Multi-FASTA format string

    Returns:
        List of (chain_id, sequence) tuples
    """

    chains = []
    current_chain_id = None
    current_sequence = ""

    for line in fasta_content.strip().split('\n'):
        line = line.strip()

        if line.startswith('>'):
            # Save previous chain
            if current_chain_id is not None:
                chains.append((current_chain_id, current_sequence))

            # Start new chain
            current_chain_id = line[1:].split()[0]  # Take first word after >
            current_sequence = ""
        else:
            current_sequence += line.upper()

    # Save last chain
    if current_chain_id is not None:
        chains.append((current_chain_id, current_sequence))

    return chains
