#!/usr/bin/env python3
"""
Protein Language Model to EvoFormer Projection Module

This module handles the projection of ESM-2 embeddings (1280-dim) into the
EvoFormer MSA representation space (64/128-dim) to replace MSA dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math
import logging
from dataclasses import dataclass


@dataclass
class PLMProjectionConfig:
    """Configuration for PLM projection module."""
    plm_dim: int = 1280  # ESM-2 650M embedding dimension
    msa_dim: int = 256   # OpenFold MSA representation dimension
    num_heads: int = 8   # Number of attention heads for projection
    dropout: float = 0.1
    use_layer_norm: bool = True
    use_residual: bool = True
    projection_type: str = "linear"  # "linear", "mlp", "attention"
    mlp_hidden_dim: Optional[int] = None
    

class LinearProjection(nn.Module):
    """Simple linear projection from PLM to MSA space."""
    
    def __init__(self, config: PLMProjectionConfig):
        super().__init__()
        self.config = config
        
        self.projection = nn.Linear(config.plm_dim, config.msa_dim)
        
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.msa_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, plm_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project PLM embeddings to MSA space.
        
        Args:
            plm_embeddings: [batch, seq_len, plm_dim]
            
        Returns:
            msa_embeddings: [batch, seq_len, msa_dim]
        """
        # Linear projection
        msa_embeddings = self.projection(plm_embeddings)
        
        # Apply layer norm
        if self.config.use_layer_norm:
            msa_embeddings = self.layer_norm(msa_embeddings)
        
        # Apply dropout
        msa_embeddings = self.dropout(msa_embeddings)
        
        return msa_embeddings


class MLPProjection(nn.Module):
    """MLP-based projection from PLM to MSA space."""
    
    def __init__(self, config: PLMProjectionConfig):
        super().__init__()
        self.config = config
        
        hidden_dim = config.mlp_hidden_dim or (config.plm_dim + config.msa_dim) // 2
        
        self.mlp = nn.Sequential(
            nn.Linear(config.plm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.msa_dim)
        )
        
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.msa_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MLP weights."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, plm_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project PLM embeddings to MSA space using MLP.
        
        Args:
            plm_embeddings: [batch, seq_len, plm_dim]
            
        Returns:
            msa_embeddings: [batch, seq_len, msa_dim]
        """
        # MLP projection
        msa_embeddings = self.mlp(plm_embeddings)
        
        # Apply layer norm
        if self.config.use_layer_norm:
            msa_embeddings = self.layer_norm(msa_embeddings)
        
        # Apply dropout
        msa_embeddings = self.dropout(msa_embeddings)
        
        return msa_embeddings


class AttentionProjection(nn.Module):
    """Attention-based projection from PLM to MSA space."""
    
    def __init__(self, config: PLMProjectionConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention for projection
        self.attention = nn.MultiheadAttention(
            embed_dim=config.plm_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Final projection to MSA dimension
        self.final_projection = nn.Linear(config.plm_dim, config.msa_dim)
        
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.msa_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights."""
        nn.init.xavier_uniform_(self.final_projection.weight)
        nn.init.zeros_(self.final_projection.bias)
    
    def forward(self, plm_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project PLM embeddings to MSA space using attention.
        
        Args:
            plm_embeddings: [batch, seq_len, plm_dim]
            
        Returns:
            msa_embeddings: [batch, seq_len, msa_dim]
        """
        # Self-attention
        attended_embeddings, _ = self.attention(
            plm_embeddings, plm_embeddings, plm_embeddings
        )
        
        # Residual connection
        if self.config.use_residual:
            attended_embeddings = attended_embeddings + plm_embeddings
        
        # Final projection
        msa_embeddings = self.final_projection(attended_embeddings)
        
        # Apply layer norm
        if self.config.use_layer_norm:
            msa_embeddings = self.layer_norm(msa_embeddings)
        
        # Apply dropout
        msa_embeddings = self.dropout(msa_embeddings)
        
        return msa_embeddings


class PLMToMSAProjector(nn.Module):
    """
    Main projection module that converts PLM embeddings to MSA format.
    
    This module handles:
    - Dimension projection (1280 -> 256)
    - Single sequence to MSA row conversion
    - Positional encoding adaptation
    - Compatibility with OpenFold EvoFormer
    """
    
    def __init__(self, config: PLMProjectionConfig = None):
        super().__init__()
        self.config = config or PLMProjectionConfig()
        
        # Create projection module based on type
        if self.config.projection_type == "linear":
            self.projector = LinearProjection(self.config)
        elif self.config.projection_type == "mlp":
            self.projector = MLPProjection(self.config)
        elif self.config.projection_type == "attention":
            self.projector = AttentionProjection(self.config)
        else:
            raise ValueError(f"Unknown projection type: {self.config.projection_type}")
        
        # Positional encoding for MSA compatibility
        self.positional_encoding = self._create_positional_encoding()
        
        logging.info(f"PLM projector initialized: {self.config.projection_type}")
    
    def _create_positional_encoding(self) -> nn.Module:
        """Create positional encoding for MSA compatibility."""
        return PositionalEncoding(self.config.msa_dim, max_len=2048)
    
    def forward(
        self, 
        plm_embeddings: torch.Tensor,
        add_positional: bool = True
    ) -> torch.Tensor:
        """
        Convert PLM embeddings to MSA format.
        
        Args:
            plm_embeddings: [batch, seq_len, plm_dim] PLM token embeddings
            add_positional: Whether to add positional encoding
            
        Returns:
            msa_embeddings: [batch, 1, seq_len, msa_dim] Single MSA row
        """
        batch_size, seq_len, plm_dim = plm_embeddings.shape
        
        # Project to MSA dimension
        msa_embeddings = self.projector(plm_embeddings)  # [batch, seq_len, msa_dim]
        
        # Add positional encoding if requested
        if add_positional:
            msa_embeddings = self.positional_encoding(msa_embeddings)
        
        # Reshape to MSA format: [batch, num_msa=1, seq_len, msa_dim]
        msa_embeddings = msa_embeddings.unsqueeze(1)
        
        return msa_embeddings
    
    def create_fake_msa(
        self, 
        plm_embeddings: torch.Tensor,
        num_msa_rows: int = 1
    ) -> torch.Tensor:
        """
        Create fake MSA with multiple rows from single sequence.
        
        Args:
            plm_embeddings: [batch, seq_len, plm_dim]
            num_msa_rows: Number of MSA rows to create
            
        Returns:
            fake_msa: [batch, num_msa_rows, seq_len, msa_dim]
        """
        # Get base MSA representation
        base_msa = self.forward(plm_embeddings, add_positional=True)  # [batch, 1, seq_len, msa_dim]
        
        if num_msa_rows == 1:
            return base_msa
        
        # Create multiple rows with slight variations
        fake_msa_rows = []
        
        for i in range(num_msa_rows):
            if i == 0:
                # First row is the original
                fake_msa_rows.append(base_msa.squeeze(1))
            else:
                # Add noise to create diversity
                noise_scale = 0.1 * (i / num_msa_rows)  # Increasing noise
                noise = torch.randn_like(base_msa.squeeze(1)) * noise_scale
                noisy_row = base_msa.squeeze(1) + noise
                fake_msa_rows.append(noisy_row)
        
        # Stack rows
        fake_msa = torch.stack(fake_msa_rows, dim=1)  # [batch, num_msa_rows, seq_len, msa_dim]
        
        return fake_msa


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence position information."""
    
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


def create_plm_projector(
    plm_dim: int = 1280,
    msa_dim: int = 256,
    projection_type: str = "linear",
    **kwargs
) -> PLMToMSAProjector:
    """
    Factory function to create PLM projector.
    
    Args:
        plm_dim: PLM embedding dimension
        msa_dim: Target MSA dimension
        projection_type: Type of projection ("linear", "mlp", "attention")
        **kwargs: Additional config parameters
        
    Returns:
        Configured PLMToMSAProjector
    """
    config = PLMProjectionConfig(
        plm_dim=plm_dim,
        msa_dim=msa_dim,
        projection_type=projection_type,
        **kwargs
    )
    
    return PLMToMSAProjector(config)


# Example usage and testing
if __name__ == "__main__":
    # Test projection
    batch_size = 2
    seq_len = 128
    plm_dim = 1280
    msa_dim = 256
    
    # Create test data
    plm_embeddings = torch.randn(batch_size, seq_len, plm_dim)
    
    # Test different projection types
    for proj_type in ["linear", "mlp", "attention"]:
        print(f"\nTesting {proj_type} projection:")
        
        projector = create_plm_projector(
            plm_dim=plm_dim,
            msa_dim=msa_dim,
            projection_type=proj_type
        )
        
        # Test single MSA row
        msa_output = projector(plm_embeddings)
        print(f"Single MSA shape: {msa_output.shape}")
        
        # Test fake MSA
        fake_msa = projector.create_fake_msa(plm_embeddings, num_msa_rows=5)
        print(f"Fake MSA shape: {fake_msa.shape}")
    
    print("\nProjection tests complete!")
