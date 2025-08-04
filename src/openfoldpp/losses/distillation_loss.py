#!/usr/bin/env python3
"""
Distillation Loss Module for OpenFold++

This module implements various distillation losses for teacher-student training:
- Coordinate loss (MSE with rigid-body alignment)
- pLDDT loss (KL divergence)
- Pair representation loss (MSE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass


@dataclass
class DistillationConfig:
    """Configuration for distillation losses."""
    # Loss weights
    coord_weight: float = 1.0
    plddt_weight: float = 0.5
    pair_weight: float = 0.1
    
    # Coordinate loss settings
    align_structures: bool = True
    coord_loss_type: str = "mse"  # "mse", "huber", "smooth_l1"
    coord_clamp_max: float = 10.0  # Clamp large coordinate differences
    
    # pLDDT loss settings
    plddt_temperature: float = 1.0
    plddt_loss_type: str = "kl"  # "kl", "mse", "cross_entropy"
    
    # Pair loss settings
    pair_loss_type: str = "mse"  # "mse", "cosine"
    pair_mask_diagonal: bool = True
    
    # Scheduling
    use_curriculum: bool = True
    curriculum_steps: int = 10000
    warmup_steps: int = 1000


class RigidBodyAlignment:
    """
    Rigid body alignment for coordinate loss computation.
    
    Aligns predicted coordinates to teacher coordinates using Kabsch algorithm.
    """
    
    @staticmethod
    def kabsch_alignment(
        pred_coords: torch.Tensor,
        target_coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align predicted coordinates to target using Kabsch algorithm.
        
        Args:
            pred_coords: Predicted coordinates [batch, seq_len, 3]
            target_coords: Target coordinates [batch, seq_len, 3]
            mask: Optional mask [batch, seq_len]
            
        Returns:
            Aligned predicted coordinates and transformation matrix
        """
        batch_size, seq_len, _ = pred_coords.shape
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=pred_coords.device)
        
        aligned_coords = []
        transforms = []
        
        for b in range(batch_size):
            # Get valid coordinates
            valid_mask = mask[b].bool()
            pred_b = pred_coords[b][valid_mask]  # [valid_len, 3]
            target_b = target_coords[b][valid_mask]  # [valid_len, 3]
            
            if len(pred_b) < 3:
                # Not enough points for alignment
                aligned_coords.append(pred_coords[b])
                transforms.append(torch.eye(4, device=pred_coords.device))
                continue
            
            # Center coordinates
            pred_center = pred_b.mean(dim=0)
            target_center = target_b.mean(dim=0)
            
            pred_centered = pred_b - pred_center
            target_centered = target_b - target_center
            
            # Compute covariance matrix
            H = pred_centered.T @ target_centered
            
            # SVD
            U, S, Vt = torch.linalg.svd(H)
            
            # Compute rotation matrix
            R = Vt.T @ U.T
            
            # Ensure proper rotation (det(R) = 1)
            if torch.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Compute translation
            t = target_center - R @ pred_center
            
            # Apply transformation to all coordinates
            aligned_b = (R @ pred_coords[b].T).T + t
            aligned_coords.append(aligned_b)
            
            # Store transformation matrix
            transform = torch.eye(4, device=pred_coords.device)
            transform[:3, :3] = R
            transform[:3, 3] = t
            transforms.append(transform)
        
        aligned_coords = torch.stack(aligned_coords)
        transforms = torch.stack(transforms)
        
        return aligned_coords, transforms
    
    @staticmethod
    def simple_alignment(
        pred_coords: torch.Tensor,
        target_coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Simple center-of-mass alignment (faster alternative).
        
        Args:
            pred_coords: Predicted coordinates [batch, seq_len, 3]
            target_coords: Target coordinates [batch, seq_len, 3]
            mask: Optional mask [batch, seq_len]
            
        Returns:
            Aligned predicted coordinates
        """
        if mask is None:
            mask = torch.ones(pred_coords.shape[:2], device=pred_coords.device)
        
        # Expand mask for broadcasting
        mask_expanded = mask.unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Compute centers of mass
        pred_center = (pred_coords * mask_expanded).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).unsqueeze(-1)
        target_center = (target_coords * mask_expanded).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).unsqueeze(-1)
        
        # Align centers
        aligned_coords = pred_coords - pred_center + target_center
        
        return aligned_coords


class CoordinateLoss(nn.Module):
    """Coordinate-based distillation loss with rigid body alignment."""
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        
        if config.coord_loss_type == "huber":
            self.loss_fn = nn.HuberLoss(reduction='none')
        elif config.coord_loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss(reduction='none')
        else:  # mse
            self.loss_fn = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        pred_coords: torch.Tensor,
        target_coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute coordinate loss.
        
        Args:
            pred_coords: Predicted coordinates [batch, seq_len, 3]
            target_coords: Target coordinates [batch, seq_len, 3]
            mask: Optional mask [batch, seq_len]
            
        Returns:
            Coordinate loss scalar
        """
        
        # Align structures if requested
        if self.config.align_structures:
            if self.config.coord_loss_type == "mse":
                # Use simple alignment for speed
                aligned_coords = RigidBodyAlignment.simple_alignment(
                    pred_coords, target_coords, mask
                )
            else:
                # Use full Kabsch alignment
                aligned_coords, _ = RigidBodyAlignment.kabsch_alignment(
                    pred_coords, target_coords, mask
                )
        else:
            aligned_coords = pred_coords
        
        # Compute coordinate differences
        coord_diff = aligned_coords - target_coords
        
        # Clamp large differences to avoid instability
        if self.config.coord_clamp_max > 0:
            coord_diff = torch.clamp(coord_diff, -self.config.coord_clamp_max, self.config.coord_clamp_max)
        
        # Compute loss
        loss = self.loss_fn(aligned_coords, target_coords)  # [batch, seq_len, 3]
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # [batch, seq_len, 1]
            loss = loss * mask_expanded
            loss = loss.sum() / mask_expanded.sum()
        else:
            loss = loss.mean()
        
        return loss


class PLDDTLoss(nn.Module):
    """pLDDT-based distillation loss."""
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        self.temperature = config.plddt_temperature
    
    def forward(
        self,
        pred_plddt_logits: torch.Tensor,
        target_plddt: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute pLDDT loss.
        
        Args:
            pred_plddt_logits: Predicted pLDDT logits [batch, seq_len, num_bins]
            target_plddt: Target pLDDT scores [batch, seq_len] (0-100)
            mask: Optional mask [batch, seq_len]
            
        Returns:
            pLDDT loss scalar
        """
        
        if self.config.plddt_loss_type == "kl":
            # Convert target pLDDT to soft distribution
            target_dist = self._plddt_to_distribution(target_plddt, pred_plddt_logits.shape[-1])
            
            # Apply temperature scaling
            pred_log_probs = F.log_softmax(pred_plddt_logits / self.temperature, dim=-1)
            
            # KL divergence
            loss = F.kl_div(pred_log_probs, target_dist, reduction='none').sum(dim=-1)
            
        elif self.config.plddt_loss_type == "mse":
            # Convert logits to pLDDT scores and use MSE
            pred_plddt = self._logits_to_plddt(pred_plddt_logits)
            loss = F.mse_loss(pred_plddt, target_plddt, reduction='none')
            
        else:  # cross_entropy
            # Convert pLDDT to bin indices
            target_bins = self._plddt_to_bins(target_plddt, pred_plddt_logits.shape[-1])
            loss = F.cross_entropy(pred_plddt_logits.view(-1, pred_plddt_logits.shape[-1]), 
                                 target_bins.view(-1), reduction='none')
            loss = loss.view(target_plddt.shape)
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss
    
    def _plddt_to_distribution(self, plddt: torch.Tensor, num_bins: int) -> torch.Tensor:
        """Convert pLDDT scores to soft probability distribution."""
        
        # Create bin centers (0-100 range)
        bin_centers = torch.linspace(0, 100, num_bins, device=plddt.device)
        
        # Compute distances to bin centers
        distances = torch.abs(plddt.unsqueeze(-1) - bin_centers.unsqueeze(0).unsqueeze(0))
        
        # Convert to probabilities (closer bins get higher probability)
        sigma = 100.0 / num_bins  # Smoothing parameter
        probs = torch.exp(-distances**2 / (2 * sigma**2))
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        return probs
    
    def _logits_to_plddt(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert pLDDT logits to pLDDT scores."""
        
        num_bins = logits.shape[-1]
        bin_centers = torch.linspace(0, 100, num_bins, device=logits.device)
        
        probs = F.softmax(logits, dim=-1)
        plddt = (probs * bin_centers).sum(dim=-1)
        
        return plddt
    
    def _plddt_to_bins(self, plddt: torch.Tensor, num_bins: int) -> torch.Tensor:
        """Convert pLDDT scores to bin indices."""
        
        # Clamp to valid range
        plddt_clamped = torch.clamp(plddt, 0, 100)
        
        # Convert to bin indices
        bin_indices = (plddt_clamped / 100.0 * (num_bins - 1)).round().long()
        bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
        
        return bin_indices


class PairRepresentationLoss(nn.Module):
    """Pair representation distillation loss."""
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
    
    def forward(
        self,
        pred_pair: torch.Tensor,
        target_pair: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute pair representation loss.
        
        Args:
            pred_pair: Predicted pair representations [batch, seq_len, seq_len, hidden_dim]
            target_pair: Target pair representations [batch, seq_len, seq_len, hidden_dim]
            mask: Optional mask [batch, seq_len, seq_len]
            
        Returns:
            Pair representation loss scalar
        """
        
        if self.config.pair_loss_type == "cosine":
            # Cosine similarity loss
            pred_flat = pred_pair.view(-1, pred_pair.shape[-1])
            target_flat = target_pair.view(-1, target_pair.shape[-1])
            
            cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1)
            loss = 1 - cosine_sim  # Convert similarity to loss
            loss = loss.view(pred_pair.shape[:-1])  # [batch, seq_len, seq_len]
            
        else:  # mse
            loss = F.mse_loss(pred_pair, target_pair, reduction='none')
            loss = loss.mean(dim=-1)  # Average over hidden dimension
        
        # Mask diagonal if requested
        if self.config.pair_mask_diagonal:
            batch_size, seq_len = loss.shape[:2]
            diagonal_mask = ~torch.eye(seq_len, dtype=torch.bool, device=loss.device)
            loss = loss * diagonal_mask.unsqueeze(0)
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss


class DistillationLoss(nn.Module):
    """
    Combined distillation loss module.
    
    Combines coordinate, pLDDT, and pair representation losses with configurable weights.
    """
    
    def __init__(self, config: DistillationConfig = None):
        super().__init__()
        
        self.config = config or DistillationConfig()
        
        # Initialize individual loss modules
        self.coord_loss = CoordinateLoss(self.config)
        self.plddt_loss = PLDDTLoss(self.config)
        self.pair_loss = PairRepresentationLoss(self.config)
        
        # Training step counter for curriculum learning
        self.register_buffer('training_step', torch.tensor(0))
        
        logging.info(f"DistillationLoss initialized with weights: "
                    f"coord={self.config.coord_weight}, "
                    f"plddt={self.config.plddt_weight}, "
                    f"pair={self.config.pair_weight}")
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined distillation loss.
        
        Args:
            predictions: Dictionary of model predictions
            targets: Dictionary of teacher targets
            masks: Optional dictionary of masks
            
        Returns:
            Dictionary of losses
        """
        
        losses = {}
        total_loss = 0.0
        
        # Get current loss weights (with curriculum if enabled)
        weights = self._get_current_weights()
        
        # Coordinate loss
        if 'coordinates' in predictions and 'coordinates' in targets:
            coord_mask = masks.get('coordinates') if masks else None
            coord_loss = self.coord_loss(
                predictions['coordinates'],
                targets['coordinates'],
                coord_mask
            )
            losses['coord_loss'] = coord_loss
            total_loss += weights['coord_weight'] * coord_loss
        
        # pLDDT loss
        if 'plddt_logits' in predictions and 'plddt' in targets:
            plddt_mask = masks.get('plddt') if masks else None
            plddt_loss = self.plddt_loss(
                predictions['plddt_logits'],
                targets['plddt'],
                plddt_mask
            )
            losses['plddt_loss'] = plddt_loss
            total_loss += weights['plddt_weight'] * plddt_loss
        
        # Pair representation loss
        if 'pair_repr' in predictions and 'pair_repr' in targets:
            pair_mask = masks.get('pair_repr') if masks else None
            pair_loss = self.pair_loss(
                predictions['pair_repr'],
                targets['pair_repr'],
                pair_mask
            )
            losses['pair_loss'] = pair_loss
            total_loss += weights['pair_weight'] * pair_loss
        
        losses['total_loss'] = total_loss
        losses['weights'] = weights
        
        # Update training step
        self.training_step += 1
        
        return losses
    
    def _get_current_weights(self) -> Dict[str, float]:
        """Get current loss weights with curriculum learning."""
        
        if not self.config.use_curriculum:
            return {
                'coord_weight': self.config.coord_weight,
                'plddt_weight': self.config.plddt_weight,
                'pair_weight': self.config.pair_weight
            }
        
        # Curriculum learning: gradually increase complex loss weights
        step = self.training_step.item()
        
        if step < self.config.warmup_steps:
            # Warmup: only coordinate loss
            alpha = step / self.config.warmup_steps
            return {
                'coord_weight': self.config.coord_weight,
                'plddt_weight': alpha * self.config.plddt_weight,
                'pair_weight': 0.0
            }
        
        elif step < self.config.curriculum_steps:
            # Curriculum: gradually add pair loss
            alpha = (step - self.config.warmup_steps) / (self.config.curriculum_steps - self.config.warmup_steps)
            return {
                'coord_weight': self.config.coord_weight,
                'plddt_weight': self.config.plddt_weight,
                'pair_weight': alpha * self.config.pair_weight
            }
        
        else:
            # Full curriculum: all losses
            return {
                'coord_weight': self.config.coord_weight,
                'plddt_weight': self.config.plddt_weight,
                'pair_weight': self.config.pair_weight
            }


# Factory function
def create_distillation_loss(config: DistillationConfig = None) -> DistillationLoss:
    """
    Factory function to create distillation loss.
    
    Args:
        config: Optional distillation configuration
        
    Returns:
        DistillationLoss instance
    """
    return DistillationLoss(config)


# Example usage and testing
if __name__ == "__main__":
    # Test distillation loss
    config = DistillationConfig()
    loss_fn = create_distillation_loss(config)
    
    # Test data
    batch_size, seq_len, hidden_dim = 2, 64, 128
    
    predictions = {
        'coordinates': torch.randn(batch_size, seq_len, 3),
        'plddt_logits': torch.randn(batch_size, seq_len, 50),  # 50 bins
        'pair_repr': torch.randn(batch_size, seq_len, seq_len, hidden_dim)
    }
    
    targets = {
        'coordinates': torch.randn(batch_size, seq_len, 3),
        'plddt': torch.rand(batch_size, seq_len) * 100,  # 0-100 range
        'pair_repr': torch.randn(batch_size, seq_len, seq_len, hidden_dim)
    }
    
    # Compute losses
    losses = loss_fn(predictions, targets)
    
    print("✅ Distillation loss test successful!")
    print(f"   Total loss: {losses['total_loss']:.4f}")
    print(f"   Coord loss: {losses['coord_loss']:.4f}")
    print(f"   pLDDT loss: {losses['plddt_loss']:.4f}")
    print(f"   Pair loss: {losses['pair_loss']:.4f}")
    print(f"   Weights: {losses['weights']}")
