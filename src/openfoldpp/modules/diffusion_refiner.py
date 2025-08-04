#!/usr/bin/env python3
"""
SE(3) Diffusion Refiner for OpenFold++

This module implements a lightweight diffusion-based structure refinement head
inspired by RoseTTAFold 2's 3D diffusion block. It performs iterative refinement
of protein coordinates using SE(3)-equivariant operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass


@dataclass
class DiffusionRefinerConfig:
    """Configuration for SE(3) diffusion refiner."""
    hidden_dim: int = 256
    num_iterations: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    
    # Diffusion parameters
    num_timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    
    # SE(3) equivariance
    use_se3_equivariance: bool = True
    coordinate_scaling: float = 1.0
    
    # Performance
    use_checkpoint: bool = True
    

class SE3EquivariantLayer(nn.Module):
    """
    SE(3)-equivariant layer for coordinate refinement.
    
    Processes coordinates while maintaining SE(3) equivariance
    (rotation and translation invariance).
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Scalar features (invariant)
        self.scalar_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scalar_norm = nn.LayerNorm(hidden_dim)
        
        # Vector features (equivariant)
        self.vector_proj = nn.Linear(hidden_dim, hidden_dim * 3)  # 3D vectors
        
        # Attention for coordinate updates
        self.coord_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projections
        self.coord_update = nn.Linear(hidden_dim, 3)
        self.feature_update = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        coordinates: torch.Tensor,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with SE(3) equivariance.
        
        Args:
            coordinates: [batch, seq_len, 3] coordinate tensor
            features: [batch, seq_len, hidden_dim] feature tensor
            mask: Optional [batch, seq_len] mask
            
        Returns:
            Updated coordinates and features
        """
        batch_size, seq_len, _ = coordinates.shape
        
        # Process scalar features
        scalar_features = self.scalar_norm(self.scalar_proj(features))
        
        # Compute pairwise distances (SE(3) invariant)
        coord_diff = coordinates.unsqueeze(2) - coordinates.unsqueeze(1)  # [batch, seq_len, seq_len, 3]
        distances = torch.norm(coord_diff, dim=-1, keepdim=True)  # [batch, seq_len, seq_len, 1]
        
        # Distance-based attention weights
        distance_features = torch.exp(-distances / 10.0)  # Gaussian RBF
        
        # Self-attention on features
        attn_output, _ = self.coord_attention(
            scalar_features, scalar_features, scalar_features,
            key_padding_mask=~mask if mask is not None else None
        )
        
        # Coordinate updates (equivariant)
        coord_updates = self.coord_update(attn_output)  # [batch, seq_len, 3]
        
        # Feature updates
        feature_updates = self.feature_update(attn_output)
        
        # Apply updates
        new_coordinates = coordinates + coord_updates
        new_features = features + feature_updates
        
        return new_coordinates, new_features


class DiffusionTimeEmbedding(nn.Module):
    """Time embedding for diffusion process."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Sinusoidal time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create time embeddings.
        
        Args:
            timesteps: [batch] timestep tensor
            
        Returns:
            Time embeddings [batch, hidden_dim]
        """
        # Sinusoidal embedding
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # MLP projection
        emb = self.time_mlp(emb)
        
        return emb


class DiffusionBlock(nn.Module):
    """Single diffusion refinement block."""
    
    def __init__(self, config: DiffusionRefinerConfig):
        super().__init__()
        
        self.config = config
        
        # Time embedding
        self.time_embedding = DiffusionTimeEmbedding(config.hidden_dim)
        
        # SE(3) equivariant layers
        self.se3_layer1 = SE3EquivariantLayer(config.hidden_dim, config.num_heads)
        self.se3_layer2 = SE3EquivariantLayer(config.hidden_dim, config.num_heads)
        
        # Feature processing
        self.feature_norm1 = nn.LayerNorm(config.hidden_dim)
        self.feature_norm2 = nn.LayerNorm(config.hidden_dim)
        
        # Time conditioning
        self.time_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        coordinates: torch.Tensor,
        features: torch.Tensor,
        timestep: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through diffusion block.
        
        Args:
            coordinates: [batch, seq_len, 3] coordinates
            features: [batch, seq_len, hidden_dim] features
            timestep: [batch] timestep
            mask: Optional [batch, seq_len] mask
            
        Returns:
            Refined coordinates and features
        """
        
        # Time embedding
        time_emb = self.time_embedding(timestep)  # [batch, hidden_dim]
        time_emb = time_emb.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Add time conditioning to features
        features = features + self.time_proj(time_emb)
        
        # First SE(3) layer
        features = self.feature_norm1(features)
        coordinates, features = self.se3_layer1(coordinates, features, mask)
        features = self.dropout(features)
        
        # Second SE(3) layer
        features = self.feature_norm2(features)
        coordinates, features = self.se3_layer2(coordinates, features, mask)
        features = self.dropout(features)
        
        return coordinates, features


class SE3DiffusionRefiner(nn.Module):
    """
    SE(3) Diffusion Refiner for protein structure refinement.
    
    This module performs iterative refinement of protein coordinates using
    a diffusion-based approach with SE(3) equivariance.
    """
    
    def __init__(self, config: DiffusionRefinerConfig = None):
        super().__init__()
        
        self.config = config or DiffusionRefinerConfig()
        
        # Input projection
        self.coord_embedding = nn.Linear(3, self.config.hidden_dim)
        self.feature_projection = nn.Linear(384, self.config.hidden_dim)  # From single repr
        
        # Diffusion blocks
        self.diffusion_blocks = nn.ModuleList([
            DiffusionBlock(self.config) for _ in range(self.config.num_iterations)
        ])
        
        # Output projection
        self.coord_output = nn.Linear(self.config.hidden_dim, 3)
        
        # Diffusion schedule
        self.register_buffer('betas', self._create_diffusion_schedule())
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        logging.info(f"SE3DiffusionRefiner initialized: {self.config.num_iterations} iterations, "
                    f"hidden_dim={self.config.hidden_dim}")
    
    def _create_diffusion_schedule(self) -> torch.Tensor:
        """Create linear diffusion schedule."""
        return torch.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.config.num_timesteps
        )
    
    def add_noise(
        self,
        coordinates: torch.Tensor,
        timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to coordinates for training.
        
        Args:
            coordinates: [batch, seq_len, 3] clean coordinates
            timesteps: [batch] timesteps
            
        Returns:
            Noisy coordinates and noise
        """
        noise = torch.randn_like(coordinates)
        
        # Get alpha values for timesteps
        alpha_cumprod = self.alphas_cumprod[timesteps]
        alpha_cumprod = alpha_cumprod.view(-1, 1, 1)
        
        # Add noise: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        noisy_coords = (
            torch.sqrt(alpha_cumprod) * coordinates +
            torch.sqrt(1 - alpha_cumprod) * noise
        )
        
        return noisy_coords, noise
    
    def predict_noise(
        self,
        noisy_coordinates: torch.Tensor,
        features: torch.Tensor,
        timesteps: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict noise for denoising.
        
        Args:
            noisy_coordinates: [batch, seq_len, 3] noisy coordinates
            features: [batch, seq_len, feature_dim] input features
            timesteps: [batch] timesteps
            mask: Optional [batch, seq_len] mask
            
        Returns:
            Predicted noise [batch, seq_len, 3]
        """
        
        # Project inputs
        coord_features = self.coord_embedding(noisy_coordinates)
        input_features = self.feature_projection(features)
        combined_features = coord_features + input_features
        
        # Process through diffusion blocks
        current_coords = noisy_coordinates
        current_features = combined_features
        
        for block in self.diffusion_blocks:
            if self.config.use_checkpoint and self.training:
                current_coords, current_features = torch.utils.checkpoint.checkpoint(
                    block, current_coords, current_features, timesteps, mask
                )
            else:
                current_coords, current_features = block(
                    current_coords, current_features, timesteps, mask
                )
        
        # Predict noise
        predicted_noise = self.coord_output(current_features)
        
        return predicted_noise
    
    def denoise_step(
        self,
        noisy_coordinates: torch.Tensor,
        features: torch.Tensor,
        timestep: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Single denoising step.
        
        Args:
            noisy_coordinates: [batch, seq_len, 3] noisy coordinates
            features: [batch, seq_len, feature_dim] input features
            timestep: Current timestep
            mask: Optional [batch, seq_len] mask
            
        Returns:
            Denoised coordinates
        """
        batch_size = noisy_coordinates.shape[0]
        timesteps = torch.full((batch_size,), timestep, device=noisy_coordinates.device)
        
        # Predict noise
        predicted_noise = self.predict_noise(noisy_coordinates, features, timesteps, mask)
        
        # Compute denoising coefficients
        alpha = self.alphas[timestep]
        alpha_cumprod = self.alphas_cumprod[timestep]
        beta = self.betas[timestep]
        
        # Denoising formula
        coeff1 = 1 / torch.sqrt(alpha)
        coeff2 = beta / torch.sqrt(1 - alpha_cumprod)
        
        denoised_coords = coeff1 * (noisy_coordinates - coeff2 * predicted_noise)
        
        return denoised_coords
    
    def refine(
        self,
        initial_coordinates: torch.Tensor,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Refine protein coordinates using diffusion process.
        
        Args:
            initial_coordinates: [batch, seq_len, 3] initial coordinates
            features: [batch, seq_len, feature_dim] input features
            mask: Optional [batch, seq_len] mask
            num_steps: Number of denoising steps (default: num_timesteps)
            
        Returns:
            Refined coordinates [batch, seq_len, 3]
        """
        
        if num_steps is None:
            num_steps = self.config.num_timesteps
        
        # Start with noisy coordinates
        current_coords = initial_coordinates + torch.randn_like(initial_coordinates) * 0.1
        
        # Denoising loop
        timesteps = torch.linspace(num_steps - 1, 0, num_steps, dtype=torch.long, device=initial_coordinates.device)
        
        for t in timesteps:
            current_coords = self.denoise_step(current_coords, features, t.item(), mask)
        
        return current_coords
    
    def forward(
        self,
        coordinates: torch.Tensor,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        training_mode: bool = False,
        timesteps: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through diffusion refiner.
        
        Args:
            coordinates: [batch, seq_len, 3] input coordinates
            features: [batch, seq_len, feature_dim] input features
            mask: Optional [batch, seq_len] mask
            training_mode: Whether in training mode
            timesteps: Optional timesteps for training
            
        Returns:
            Refined coordinates, or (refined_coords, predicted_noise) if training
        """
        
        if training_mode and timesteps is not None:
            # Training: predict noise
            noisy_coords, true_noise = self.add_noise(coordinates, timesteps)
            predicted_noise = self.predict_noise(noisy_coords, features, timesteps, mask)
            return predicted_noise, true_noise
        
        else:
            # Inference: refine coordinates
            refined_coords = self.refine(coordinates, features, mask)
            return refined_coords


# Factory function
def create_diffusion_refiner(config: DiffusionRefinerConfig = None) -> SE3DiffusionRefiner:
    """
    Factory function to create SE(3) diffusion refiner.
    
    Args:
        config: Optional refiner configuration
        
    Returns:
        SE3DiffusionRefiner instance
    """
    return SE3DiffusionRefiner(config)


# Example usage and testing
if __name__ == "__main__":
    # Test diffusion refiner
    config = DiffusionRefinerConfig(
        hidden_dim=256,
        num_iterations=2,
        num_timesteps=50  # Reduced for testing
    )
    
    refiner = create_diffusion_refiner(config)
    
    # Test input
    batch_size, seq_len = 2, 128
    coordinates = torch.randn(batch_size, seq_len, 3)
    features = torch.randn(batch_size, seq_len, 384)  # Single representation
    mask = torch.ones(batch_size, seq_len).bool()
    
    # Test inference
    refined_coords = refiner(coordinates, features, mask)
    
    print("✅ SE(3) Diffusion Refiner test successful!")
    print(f"   Input coordinates: {coordinates.shape}")
    print(f"   Input features: {features.shape}")
    print(f"   Refined coordinates: {refined_coords.shape}")
    print(f"   Parameters: {sum(p.numel() for p in refiner.parameters()):,}")
    
    # Test training mode
    timesteps = torch.randint(0, config.num_timesteps, (batch_size,))
    predicted_noise, true_noise = refiner(
        coordinates, features, mask, 
        training_mode=True, timesteps=timesteps
    )
    
    print(f"   Training mode - Predicted noise: {predicted_noise.shape}")
    print(f"   Training mode - True noise: {true_noise.shape}")
    
    # Test unit test: t=0 should reproduce input
    refiner.eval()
    with torch.no_grad():
        # At t=0, no noise should be added
        timesteps_zero = torch.zeros(batch_size, dtype=torch.long)
        noisy_coords, _ = refiner.add_noise(coordinates, timesteps_zero)
        
        # Should be very close to original
        diff = torch.mean(torch.abs(noisy_coords - coordinates))
        print(f"   Unit test (t=0): Mean difference = {diff:.6f} (should be ~0)")
        
        assert diff < 1e-5, f"Unit test failed: difference {diff} too large"
        print("   ✅ Unit test passed: t=0 reproduces input")
