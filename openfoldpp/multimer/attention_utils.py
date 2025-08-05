"""
Attention Utilities for OdinFold Multimer

Implements attention masking and inter-chain attention mechanisms
for multi-chain protein complex folding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import math


class MultimerAttentionMask:
    """
    Creates attention masks for multimer folding.
    
    Handles intra-chain and inter-chain attention patterns
    with configurable masking strategies.
    """
    
    def __init__(self, mask_inter_chain: bool = False, allow_chain_breaks: bool = False):
        self.mask_inter_chain = mask_inter_chain
        self.allow_chain_breaks = allow_chain_breaks
    
    def create_multimer_mask(self, chain_id_tensor: torch.Tensor, 
                           chain_break_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Create attention mask for multimer sequences.
        
        Args:
            chain_id_tensor: Chain ID for each position [seq_len]
            chain_break_mask: Mask for chain break positions [seq_len]
            
        Returns:
            Attention mask [seq_len, seq_len] where True = masked (no attention)
        """
        
        seq_len = chain_id_tensor.shape[0]
        device = chain_id_tensor.device
        
        # Base mask (all positions can attend to all)
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        # Create chain interaction matrix
        chain_matrix = chain_id_tensor.unsqueeze(0) == chain_id_tensor.unsqueeze(1)
        
        # Mask inter-chain attention if requested
        if self.mask_inter_chain:
            inter_chain_mask = ~chain_matrix
            mask = mask | inter_chain_mask
        
        # Mask chain break positions
        if chain_break_mask is not None and not self.allow_chain_breaks:
            # Positions cannot attend to chain breaks
            chain_break_expanded = chain_break_mask.unsqueeze(0).expand(seq_len, -1)
            mask = mask | chain_break_expanded
            
            # Chain breaks cannot attend to anything
            chain_break_self = chain_break_mask.unsqueeze(1).expand(-1, seq_len)
            mask = mask | chain_break_self
        
        return mask
    
    def create_triangular_mask(self, chain_id_tensor: torch.Tensor, 
                             mask_type: str = "full") -> torch.Tensor:
        """
        Create triangular attention masks for pair representations.
        
        Args:
            chain_id_tensor: Chain ID tensor
            mask_type: "intra_only", "inter_only", or "full"
            
        Returns:
            Triangular mask for pair attention
        """
        
        seq_len = chain_id_tensor.shape[0]
        device = chain_id_tensor.device
        
        # Create chain pair matrix
        chain_i = chain_id_tensor.unsqueeze(1).expand(-1, seq_len)
        chain_j = chain_id_tensor.unsqueeze(0).expand(seq_len, -1)
        
        if mask_type == "intra_only":
            # Only allow attention within same chain
            mask = chain_i != chain_j
        elif mask_type == "inter_only":
            # Only allow attention between different chains
            mask = chain_i == chain_j
        else:  # "full"
            # Allow all attention
            mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        return mask


class InterChainAttention(nn.Module):
    """
    Specialized attention mechanism for inter-chain interactions.
    
    Implements cross-attention between different protein chains
    with geometric and evolutionary biases.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        # Attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Inter-chain specific components
        self.chain_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.distance_bias = nn.Linear(1, num_heads)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, chain_id_tensor: torch.Tensor,
                distances: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of inter-chain attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            chain_id_tensor: Chain IDs [seq_len]
            distances: Optional distance matrix [seq_len, seq_len]
            mask: Optional attention mask [seq_len, seq_len]
            
        Returns:
            Attended output [batch_size, seq_len, d_model]
        """
        
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Add inter-chain bias
        chain_matrix = chain_id_tensor.unsqueeze(0) != chain_id_tensor.unsqueeze(1)
        inter_chain_bias = self.chain_bias * chain_matrix.float().unsqueeze(0).unsqueeze(0)
        scores = scores + inter_chain_bias
        
        # Add distance bias if provided
        if distances is not None:
            dist_bias = self.distance_bias(distances.unsqueeze(-1))  # [seq_len, seq_len, num_heads]
            dist_bias = dist_bias.permute(2, 0, 1)  # [num_heads, seq_len, seq_len]
            scores = scores + dist_bias.unsqueeze(0)  # [batch_size, num_heads, seq_len, seq_len]
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out_proj(out)
        
        return out


class MultimerTriangleAttention(nn.Module):
    """
    Triangle attention for multimer pair representations.
    
    Handles both intra-chain and inter-chain triangular updates
    with proper masking and geometric constraints.
    """
    
    def __init__(self, d_pair: int, num_heads: int = 4):
        super().__init__()
        
        self.d_pair = d_pair
        self.num_heads = num_heads
        self.head_dim = d_pair // num_heads
        
        # Triangle attention components
        self.triangle_attn_start = nn.MultiheadAttention(d_pair, num_heads, batch_first=True)
        self.triangle_attn_end = nn.MultiheadAttention(d_pair, num_heads, batch_first=True)
        
        # Layer norms
        self.norm_start = nn.LayerNorm(d_pair)
        self.norm_end = nn.LayerNorm(d_pair)
        
        # Chain-specific gating
        self.chain_gate = nn.Linear(d_pair, d_pair)
        
    def forward(self, pair_repr: torch.Tensor, chain_id_tensor: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply triangle attention to pair representation.
        
        Args:
            pair_repr: Pair representation [batch_size, seq_len, seq_len, d_pair]
            chain_id_tensor: Chain IDs [seq_len]
            mask: Optional mask [seq_len, seq_len]
            
        Returns:
            Updated pair representation
        """
        
        batch_size, seq_len, _, d_pair = pair_repr.shape
        
        # Triangle attention starting from i
        pair_flat_i = pair_repr.view(batch_size * seq_len, seq_len, d_pair)
        
        # Create mask for triangle attention
        if mask is not None:
            attn_mask_i = mask.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        else:
            attn_mask_i = None
        
        # Apply attention
        attn_out_i, _ = self.triangle_attn_start(
            pair_flat_i, pair_flat_i, pair_flat_i,
            attn_mask=attn_mask_i
        )
        
        attn_out_i = attn_out_i.view(batch_size, seq_len, seq_len, d_pair)
        pair_repr = self.norm_start(pair_repr + attn_out_i)
        
        # Triangle attention starting from j
        pair_flat_j = pair_repr.transpose(1, 2).contiguous().view(batch_size * seq_len, seq_len, d_pair)
        
        if mask is not None:
            attn_mask_j = mask.transpose(0, 1).unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        else:
            attn_mask_j = None
        
        attn_out_j, _ = self.triangle_attn_end(
            pair_flat_j, pair_flat_j, pair_flat_j,
            attn_mask=attn_mask_j
        )
        
        attn_out_j = attn_out_j.view(batch_size, seq_len, seq_len, d_pair).transpose(1, 2)
        pair_repr = self.norm_end(pair_repr + attn_out_j)
        
        # Apply chain-specific gating
        chain_gate = torch.sigmoid(self.chain_gate(pair_repr))
        pair_repr = pair_repr * chain_gate
        
        return pair_repr


class MultimerAttentionBias:
    """
    Computes attention biases for multimer folding.
    
    Includes evolutionary, geometric, and chain-specific biases
    to guide attention patterns in multimer structures.
    """
    
    def __init__(self, d_model: int):
        self.d_model = d_model
    
    def compute_evolutionary_bias(self, msa_features: torch.Tensor, 
                                chain_id_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute evolutionary coupling bias between residues.
        
        Args:
            msa_features: MSA-derived features [seq_len, d_msa]
            chain_id_tensor: Chain IDs [seq_len]
            
        Returns:
            Evolutionary bias matrix [seq_len, seq_len]
        """
        
        seq_len = msa_features.shape[0]
        
        # Compute coevolution scores (simplified)
        coevolution = torch.matmul(msa_features, msa_features.transpose(0, 1))
        
        # Normalize
        coevolution = coevolution / (torch.norm(msa_features, dim=1, keepdim=True) + 1e-8)
        coevolution = coevolution / (torch.norm(msa_features, dim=1).unsqueeze(0) + 1e-8)
        
        # Boost inter-chain coevolution
        chain_matrix = chain_id_tensor.unsqueeze(0) != chain_id_tensor.unsqueeze(1)
        inter_chain_boost = 1.5
        coevolution = torch.where(chain_matrix, coevolution * inter_chain_boost, coevolution)
        
        return coevolution
    
    def compute_geometric_bias(self, coordinates: torch.Tensor,
                             chain_id_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute geometric bias based on 3D coordinates.
        
        Args:
            coordinates: 3D coordinates [seq_len, 3]
            chain_id_tensor: Chain IDs [seq_len]
            
        Returns:
            Geometric bias matrix [seq_len, seq_len]
        """
        
        # Compute pairwise distances
        distances = torch.cdist(coordinates, coordinates)
        
        # Convert to bias (closer = higher attention)
        geometric_bias = 1.0 / (1.0 + distances / 10.0)  # Scale factor
        
        # Boost interface regions (inter-chain contacts)
        chain_matrix = chain_id_tensor.unsqueeze(0) != chain_id_tensor.unsqueeze(1)
        interface_mask = chain_matrix & (distances < 8.0)  # 8Å cutoff for interfaces
        
        interface_boost = 2.0
        geometric_bias = torch.where(interface_mask, geometric_bias * interface_boost, geometric_bias)
        
        return geometric_bias
    
    def combine_biases(self, *biases: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Combine multiple attention biases.
        
        Args:
            biases: Variable number of bias tensors
            weights: Optional weights for each bias
            
        Returns:
            Combined bias tensor
        """
        
        if weights is None:
            weights = torch.ones(len(biases)) / len(biases)
        
        combined = torch.zeros_like(biases[0])
        for bias, weight in zip(biases, weights):
            combined += weight * bias
        
        return combined
