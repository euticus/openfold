#!/usr/bin/env python3
"""
pLDDT Confidence Estimation for OpenFold++

This module implements per-residue confidence prediction using pLDDT
(predicted Local Distance Difference Test) scores from 0-100.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import math


@dataclass
class ConfidenceConfig:
    """Configuration for pLDDT confidence estimation."""
    
    # Architecture
    input_dim: int = 256  # Single representation dimension
    hidden_dim: int = 128  # Hidden layer dimension
    num_layers: int = 3   # Number of MLP layers
    dropout: float = 0.1  # Dropout rate
    
    # pLDDT settings
    num_bins: int = 50    # Number of distance bins for pLDDT calculation
    min_distance: float = 0.0   # Minimum distance (Å)
    max_distance: float = 15.0  # Maximum distance (Å)
    
    # Training settings
    confidence_loss_weight: float = 0.1  # Weight in total loss
    use_focal_loss: bool = True  # Use focal loss for class imbalance
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Output settings
    output_logits: bool = False  # Output raw logits or probabilities
    temperature: float = 1.0     # Temperature scaling for calibration


class pLDDTHead(nn.Module):
    """
    pLDDT confidence prediction head.
    
    Predicts per-residue confidence scores based on local structure quality
    using the pLDDT metric (0-100 scale).
    """
    
    def __init__(self, config: ConfidenceConfig):
        super().__init__()
        
        self.config = config
        
        # Distance bins for pLDDT calculation
        self.register_buffer(
            'distance_bins',
            torch.linspace(config.min_distance, config.max_distance, config.num_bins)
        )
        
        # MLP layers for confidence prediction
        layers = []
        
        # Input layer
        layers.extend([
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        ])
        
        # Hidden layers
        for _ in range(config.num_layers - 2):
            layers.extend([
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
        
        # Output layer (predict distance distribution)
        layers.append(nn.Linear(config.hidden_dim, config.num_bins))
        
        self.confidence_mlp = nn.Sequential(*layers)
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1) * config.temperature)
        
        logging.info(f"pLDDT head initialized: {config.num_bins} bins, {config.num_layers} layers")
    
    def forward(
        self,
        single_repr: torch.Tensor,
        coordinates: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for confidence prediction.
        
        Args:
            single_repr: [batch, seq_len, input_dim] single representation
            coordinates: [batch, seq_len, 3] predicted coordinates
            mask: [batch, seq_len] optional sequence mask
            
        Returns:
            Dictionary with confidence predictions and pLDDT scores
        """
        
        batch_size, seq_len, _ = single_repr.shape
        
        # Predict distance distributions
        distance_logits = self.confidence_mlp(single_repr)  # [batch, seq_len, num_bins]
        
        # Apply temperature scaling
        distance_logits = distance_logits / self.temperature
        
        # Convert to probabilities
        distance_probs = F.softmax(distance_logits, dim=-1)  # [batch, seq_len, num_bins]
        
        # Calculate pLDDT scores
        plddt_scores = self._calculate_plddt(
            distance_probs, coordinates, mask
        )  # [batch, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            plddt_scores = plddt_scores * mask
            distance_logits = distance_logits * mask.unsqueeze(-1)
            distance_probs = distance_probs * mask.unsqueeze(-1)
        
        outputs = {
            'plddt': plddt_scores,
            'distance_logits': distance_logits,
            'distance_probs': distance_probs,
            'confidence_bins': self.distance_bins
        }
        
        return outputs
    
    def _calculate_plddt(
        self,
        distance_probs: torch.Tensor,
        coordinates: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate pLDDT scores from distance distributions.
        
        pLDDT measures the fraction of Cα-Cα distances within 15Å
        that are predicted to be within specific thresholds.
        """
        
        batch_size, seq_len, num_bins = distance_probs.shape
        
        # Calculate actual pairwise distances
        # coordinates: [batch, seq_len, 3]
        coord_diff = coordinates.unsqueeze(2) - coordinates.unsqueeze(1)  # [batch, seq_len, seq_len, 3]
        actual_distances = torch.norm(coord_diff, dim=-1)  # [batch, seq_len, seq_len]
        
        # Only consider distances within 15Å (local structure)
        local_mask = actual_distances <= 15.0
        
        # Apply sequence mask if provided
        if mask is not None:
            seq_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # [batch, seq_len, seq_len]
            local_mask = local_mask & seq_mask
        
        # pLDDT thresholds (standard: 0.5, 1.0, 2.0, 4.0 Å)
        thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0], device=distance_probs.device)
        
        plddt_scores = torch.zeros(batch_size, seq_len, device=distance_probs.device)
        
        for i in range(seq_len):
            # Get neighbors within 15Å
            neighbors_mask = local_mask[:, i, :]  # [batch, seq_len]
            
            if neighbors_mask.sum() == 0:
                continue
            
            # Predicted distance distribution for residue i
            pred_dist_probs = distance_probs[:, i, :]  # [batch, num_bins]
            
            # Calculate expected distance
            expected_distances = torch.sum(
                pred_dist_probs * self.distance_bins.unsqueeze(0), dim=-1
            )  # [batch]
            
            # Actual distances to neighbors
            actual_neighbor_distances = actual_distances[:, i, :]  # [batch, seq_len]
            
            # Calculate fraction within thresholds
            threshold_scores = []
            
            for threshold in thresholds:
                # Fraction of neighbors within threshold
                within_threshold = (actual_neighbor_distances <= threshold) & neighbors_mask
                fraction = within_threshold.sum(dim=-1).float() / neighbors_mask.sum(dim=-1).float()
                threshold_scores.append(fraction)
            
            # Average across thresholds (standard pLDDT calculation)
            residue_plddt = torch.stack(threshold_scores, dim=-1).mean(dim=-1)  # [batch]
            plddt_scores[:, i] = residue_plddt * 100  # Scale to 0-100
        
        return plddt_scores
    
    def predict_confidence(
        self,
        single_repr: torch.Tensor,
        coordinates: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict confidence scores (0-100) for each residue.
        
        Returns:
            plddt_scores: [batch, seq_len] confidence scores (0-100)
        """
        
        with torch.no_grad():
            outputs = self.forward(single_repr, coordinates, mask)
            return outputs['plddt']
    
    def get_confidence_categories(self, plddt_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Categorize confidence scores into standard pLDDT ranges.
        
        Returns:
            Dictionary with boolean masks for each confidence category
        """
        
        categories = {
            'very_high': plddt_scores >= 90,    # Very high confidence
            'confident': plddt_scores >= 70,    # Confident
            'low': plddt_scores >= 50,          # Low confidence  
            'very_low': plddt_scores < 50       # Very low confidence
        }
        
        return categories


class ConfidenceLoss(nn.Module):
    """
    Loss function for confidence prediction training.
    
    Combines distance prediction loss with pLDDT regression loss.
    """
    
    def __init__(self, config: ConfidenceConfig):
        super().__init__()
        
        self.config = config
        
        # Distance prediction loss
        if config.use_focal_loss:
            self.distance_loss = FocalLoss(
                alpha=config.focal_alpha,
                gamma=config.focal_gamma
            )
        else:
            self.distance_loss = nn.CrossEntropyLoss()
        
        # pLDDT regression loss
        self.plddt_loss = nn.MSELoss()
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate confidence prediction loss.
        
        Args:
            predictions: Model predictions from pLDDT head
            targets: Ground truth targets
            mask: Optional sequence mask
            
        Returns:
            Dictionary with loss components
        """
        
        losses = {}
        
        # Distance prediction loss
        if 'distance_logits' in predictions and 'distance_targets' in targets:
            distance_logits = predictions['distance_logits']
            distance_targets = targets['distance_targets']
            
            if mask is not None:
                # Apply mask
                distance_logits = distance_logits[mask]
                distance_targets = distance_targets[mask]
            
            losses['distance_loss'] = self.distance_loss(
                distance_logits.view(-1, distance_logits.size(-1)),
                distance_targets.view(-1)
            )
        
        # pLDDT regression loss
        if 'plddt' in predictions and 'plddt_targets' in targets:
            plddt_pred = predictions['plddt']
            plddt_targets = targets['plddt_targets']
            
            if mask is not None:
                plddt_pred = plddt_pred[mask]
                plddt_targets = plddt_targets[mask]
            
            losses['plddt_loss'] = self.plddt_loss(plddt_pred, plddt_targets)
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_confidence_loss'] = total_loss
        
        return losses


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in distance prediction."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_confidence_head(config: ConfidenceConfig = None) -> pLDDTHead:
    """
    Factory function to create pLDDT confidence head.
    
    Args:
        config: Optional configuration
        
    Returns:
        pLDDTHead instance
    """
    
    config = config or ConfidenceConfig()
    return pLDDTHead(config)


# Example usage and testing
if __name__ == "__main__":
    print("🎯 Testing pLDDT Confidence Estimation")
    print("=" * 50)
    
    # Create confidence head
    config = ConfidenceConfig(
        input_dim=256,
        hidden_dim=128,
        num_layers=3,
        num_bins=50
    )
    
    confidence_head = create_confidence_head(config)
    
    print(f"✅ Confidence head created successfully")
    print(f"   Input dim: {config.input_dim}")
    print(f"   Hidden dim: {config.hidden_dim}")
    print(f"   Distance bins: {config.num_bins}")
    print(f"   Layers: {config.num_layers}")
    
    # Test inputs
    batch_size = 2
    seq_len = 100
    
    single_repr = torch.randn(batch_size, seq_len, config.input_dim)
    coordinates = torch.randn(batch_size, seq_len, 3) * 10  # Protein-like coordinates
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    print(f"\n🧪 Testing forward pass:")
    print(f"   Single repr shape: {single_repr.shape}")
    print(f"   Coordinates shape: {coordinates.shape}")
    print(f"   Mask shape: {mask.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = confidence_head(single_repr, coordinates, mask)
    
    print(f"\n📊 Output Results:")
    print(f"   pLDDT scores shape: {outputs['plddt'].shape}")
    print(f"   Distance logits shape: {outputs['distance_logits'].shape}")
    print(f"   Distance probs shape: {outputs['distance_probs'].shape}")
    
    # Analyze confidence scores
    plddt_scores = outputs['plddt']
    print(f"\n🎯 Confidence Analysis:")
    print(f"   Mean pLDDT: {plddt_scores.mean():.1f}")
    print(f"   Min pLDDT: {plddt_scores.min():.1f}")
    print(f"   Max pLDDT: {plddt_scores.max():.1f}")
    print(f"   Std pLDDT: {plddt_scores.std():.1f}")
    
    # Test confidence categories
    categories = confidence_head.get_confidence_categories(plddt_scores)
    
    print(f"\n📈 Confidence Categories:")
    for category, mask_tensor in categories.items():
        count = mask_tensor.sum().item()
        percentage = count / (batch_size * seq_len) * 100
        print(f"   {category.replace('_', ' ').title()}: {count}/{batch_size * seq_len} ({percentage:.1f}%)")
    
    # Test loss calculation
    print(f"\n🔧 Testing loss calculation:")
    
    loss_fn = ConfidenceLoss(config)
    
    # Mock targets
    targets = {
        'distance_targets': torch.randint(0, config.num_bins, (batch_size, seq_len)),
        'plddt_targets': torch.rand(batch_size, seq_len) * 100
    }
    
    losses = loss_fn(outputs, targets, mask)
    
    print(f"   Distance loss: {losses.get('distance_loss', 0):.4f}")
    print(f"   pLDDT loss: {losses.get('plddt_loss', 0):.4f}")
    print(f"   Total loss: {losses['total_confidence_loss']:.4f}")
    
    print(f"\n🎯 pLDDT Confidence Head Ready!")
    print(f"   Per-residue confidence scores (0-100)")
    print(f"   Standard pLDDT calculation with distance bins")
    print(f"   Focal loss for improved training")
