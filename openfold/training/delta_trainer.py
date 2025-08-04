"""
Training script for the delta prediction model.

This module provides training utilities for the GNN-based mutation effect predictor,
including data loading, loss functions, and training loops.
"""

import os
import logging
import json
import pickle
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from dataclasses import dataclass
import random

from openfold.np import protein, residue_constants
from openfold.model.delta_predictor import (
    DeltaPredictor, 
    MutationInput, 
    DeltaPrediction,
    create_delta_predictor
)


@dataclass
class MutationDataPoint:
    """Single training data point for mutation prediction."""
    original_structure: protein.Protein
    mutated_structure: protein.Protein
    mutation_position: int
    original_aa: str
    target_aa: str
    ddg: Optional[float] = None  # Free energy change (kcal/mol)
    source: str = "unknown"  # Data source (e.g., "foldx", "experimental")


class MutationDataset(Dataset):
    """Dataset for mutation effect training data."""
    
    def __init__(self, 
                 data_points: List[MutationDataPoint],
                 local_radius: float = 10.0,
                 augment_data: bool = True):
        """
        Args:
            data_points: List of mutation data points
            local_radius: Radius for local environment extraction
            augment_data: Whether to apply data augmentation
        """
        self.data_points = data_points
        self.local_radius = local_radius
        self.augment_data = augment_data
    
    def __len__(self) -> int:
        return len(self.data_points)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        data_point = self.data_points[idx]
        
        # Create mutation input
        mutation_input = MutationInput(
            protein_structure=data_point.original_structure,
            mutation_position=data_point.mutation_position,
            original_aa=data_point.original_aa,
            target_aa=data_point.target_aa,
            local_radius=self.local_radius
        )
        
        # Compute ground truth deltas
        orig_pos = data_point.original_structure.atom_positions
        mut_pos = data_point.mutated_structure.atom_positions
        
        # Ensure same shape (handle insertions/deletions)
        min_len = min(len(orig_pos), len(mut_pos))
        position_deltas = mut_pos[:min_len] - orig_pos[:min_len]
        
        # Flatten to atom-level deltas
        position_deltas = position_deltas.reshape(-1, 3)
        
        # Create confidence mask (atoms that actually moved)
        movement_magnitude = np.linalg.norm(position_deltas, axis=1)
        confidence_targets = (movement_magnitude > 0.1).astype(np.float32)  # 0.1 Å threshold
        
        # Data augmentation
        if self.augment_data and random.random() < 0.5:
            # Add small random noise to original structure
            noise_scale = 0.05  # 0.05 Å noise
            noise = np.random.normal(0, noise_scale, orig_pos.shape)
            mutation_input.protein_structure = protein.Protein(
                atom_positions=orig_pos + noise,
                atom_mask=data_point.original_structure.atom_mask,
                aatype=data_point.original_structure.aatype,
                residue_index=data_point.original_structure.residue_index,
                b_factors=data_point.original_structure.b_factors
            )
        
        return {
            'mutation_input': mutation_input,
            'position_deltas': torch.tensor(position_deltas, dtype=torch.float32),
            'confidence_targets': torch.tensor(confidence_targets, dtype=torch.float32),
            'ddg': torch.tensor(data_point.ddg if data_point.ddg is not None else 0.0, dtype=torch.float32)
        }


class DeltaPredictionLoss(nn.Module):
    """Loss function for delta prediction training."""
    
    def __init__(self,
                 position_weight: float = 1.0,
                 confidence_weight: float = 0.5,
                 ddg_weight: float = 0.1,
                 huber_delta: float = 1.0):
        """
        Args:
            position_weight: Weight for position prediction loss
            confidence_weight: Weight for confidence prediction loss
            ddg_weight: Weight for energy change prediction loss
            huber_delta: Delta parameter for Huber loss
        """
        super().__init__()
        self.position_weight = position_weight
        self.confidence_weight = confidence_weight
        self.ddg_weight = ddg_weight
        self.huber_delta = huber_delta
    
    def forward(self, 
                predictions: DeltaPrediction,
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for delta predictions.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Position prediction loss (Huber loss for robustness)
        position_loss = F.huber_loss(
            predictions.position_deltas,
            targets['position_deltas'],
            delta=self.huber_delta
        )
        losses['position_loss'] = position_loss
        
        # Confidence prediction loss (BCE)
        confidence_loss = F.binary_cross_entropy(
            predictions.confidence_scores,
            targets['confidence_targets']
        )
        losses['confidence_loss'] = confidence_loss
        
        # Energy change loss (if available)
        if predictions.energy_change is not None:
            ddg_loss = F.mse_loss(
                predictions.energy_change,
                targets['ddg']
            )
            losses['ddg_loss'] = ddg_loss
        else:
            losses['ddg_loss'] = torch.tensor(0.0)
        
        # Total loss
        total_loss = (
            self.position_weight * position_loss +
            self.confidence_weight * confidence_loss +
            self.ddg_weight * losses['ddg_loss']
        )
        losses['total_loss'] = total_loss
        
        return losses


class DeltaTrainer:
    """Trainer for delta prediction model."""
    
    def __init__(self,
                 model: DeltaPredictor,
                 train_dataset: MutationDataset,
                 val_dataset: Optional[MutationDataset] = None,
                 batch_size: int = 32,
                 learning_rate: float = 1e-4,
                 device: str = "cuda"):
        """
        Args:
            model: Delta prediction model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for training
            learning_rate: Learning rate
            device: Device for training
        """
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self._collate_fn
            )
        
        # Optimizer and loss
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.loss_fn = DeltaPredictionLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, List]:
        """Custom collate function for mutation data."""
        # Since each sample has different graph structure,
        # we return lists instead of batched tensors
        collated = {
            'mutation_inputs': [item['mutation_input'] for item in batch],
            'position_deltas': [item['position_deltas'] for item in batch],
            'confidence_targets': [item['confidence_targets'] for item in batch],
            'ddg': torch.stack([item['ddg'] for item in batch])
        }
        return collated
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {'total_loss': 0.0, 'position_loss': 0.0, 'confidence_loss': 0.0, 'ddg_loss': 0.0}
        num_batches = 0
        
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            batch_losses = {'total_loss': 0.0, 'position_loss': 0.0, 'confidence_loss': 0.0, 'ddg_loss': 0.0}
            
            # Process each sample in batch
            for i, mutation_input in enumerate(batch['mutation_inputs']):
                try:
                    # Forward pass
                    prediction = self.model(mutation_input)
                    
                    # Prepare targets
                    targets = {
                        'position_deltas': batch['position_deltas'][i].to(self.device),
                        'confidence_targets': batch['confidence_targets'][i].to(self.device),
                        'ddg': batch['ddg'][i].to(self.device)
                    }
                    
                    # Compute loss
                    losses = self.loss_fn(prediction, targets)
                    
                    # Accumulate losses
                    for key, value in losses.items():
                        batch_losses[key] += value
                
                except Exception as e:
                    logging.warning(f"Error processing sample {i}: {e}")
                    continue
            
            # Average losses over batch
            for key in batch_losses:
                batch_losses[key] /= len(batch['mutation_inputs'])
            
            # Backward pass
            batch_losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate epoch losses
            for key, value in batch_losses.items():
                epoch_losses[key] += value.item()
            num_batches += 1
        
        # Average over batches
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if not self.val_dataset:
            return {}
        
        self.model.eval()
        val_losses = {'total_loss': 0.0, 'position_loss': 0.0, 'confidence_loss': 0.0, 'ddg_loss': 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch_losses = {'total_loss': 0.0, 'position_loss': 0.0, 'confidence_loss': 0.0, 'ddg_loss': 0.0}
                
                for i, mutation_input in enumerate(batch['mutation_inputs']):
                    try:
                        prediction = self.model(mutation_input)
                        
                        targets = {
                            'position_deltas': batch['position_deltas'][i].to(self.device),
                            'confidence_targets': batch['confidence_targets'][i].to(self.device),
                            'ddg': batch['ddg'][i].to(self.device)
                        }
                        
                        losses = self.loss_fn(prediction, targets)
                        
                        for key, value in losses.items():
                            batch_losses[key] += value
                    
                    except Exception as e:
                        logging.warning(f"Error validating sample {i}: {e}")
                        continue
                
                # Average over batch
                for key in batch_losses:
                    batch_losses[key] /= len(batch['mutation_inputs'])
                    val_losses[key] += batch_losses[key].item()
                
                num_batches += 1
        
        # Average over batches
        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)
        
        return val_losses
    
    def train(self, num_epochs: int, save_dir: str = "./checkpoints") -> None:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Update scheduler
            if val_losses:
                self.scheduler.step(val_losses['total_loss'])
            
            # Log progress
            log_msg = f"Epoch {epoch+1}/{num_epochs}"
            log_msg += f" | Train Loss: {train_losses['total_loss']:.4f}"
            if val_losses:
                log_msg += f" | Val Loss: {val_losses['total_loss']:.4f}"
            
            logging.info(log_msg)
            
            # Save checkpoint
            if val_losses and val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.save_checkpoint(os.path.join(save_dir, "best_model.pt"))
            
            # Save training history
            history_entry = {
                'epoch': epoch + 1,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(history_entry)
            
            # Save history
            with open(os.path.join(save_dir, "training_history.json"), 'w') as f:
                json.dump(self.training_history, f, indent=2)
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }
        torch.save(checkpoint, path)
        logging.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        logging.info(f"Checkpoint loaded from {path}")


def create_synthetic_training_data(num_samples: int = 1000) -> List[MutationDataPoint]:
    """
    Create synthetic training data for testing.
    
    Args:
        num_samples: Number of synthetic samples to create
        
    Returns:
        List of synthetic mutation data points
    """
    data_points = []
    
    for i in range(num_samples):
        # Create a small synthetic protein (10 residues)
        n_res = 10
        
        # Random but reasonable coordinates
        np.random.seed(i)
        positions = np.random.randn(n_res, 37, 3) * 2.0
        
        # Create backbone
        for j in range(n_res):
            positions[j, 0] = [j * 3.8, 0, 0]      # N
            positions[j, 1] = [j * 3.8 + 1.5, 0, 0]  # CA
            positions[j, 2] = [j * 3.8 + 3.0, 0, 0]  # C
        
        # Atom mask (backbone only)
        atom_mask = np.zeros((n_res, 37))
        atom_mask[:, :3] = 1.0
        
        # Random amino acid types
        aatype = np.random.randint(0, 20, n_res)
        residue_index = np.arange(n_res)
        b_factors = np.ones((n_res, 37)) * 50.0
        
        # Original structure
        orig_prot = protein.Protein(
            atom_positions=positions,
            atom_mask=atom_mask,
            aatype=aatype,
            residue_index=residue_index,
            b_factors=b_factors
        )
        
        # Mutated structure (add small random displacement)
        mut_positions = positions.copy()
        mutation_pos = np.random.randint(0, n_res)
        
        # Add displacement around mutation site
        for j in range(max(0, mutation_pos-2), min(n_res, mutation_pos+3)):
            displacement = np.random.normal(0, 0.5, (37, 3))  # 0.5 Å displacement
            mut_positions[j] += displacement
        
        mut_aatype = aatype.copy()
        original_aa = residue_constants.restypes[aatype[mutation_pos]]
        new_aa_idx = np.random.randint(0, 20)
        mut_aatype[mutation_pos] = new_aa_idx
        target_aa = residue_constants.restypes[new_aa_idx]
        
        mut_prot = protein.Protein(
            atom_positions=mut_positions,
            atom_mask=atom_mask,
            aatype=mut_aatype,
            residue_index=residue_index,
            b_factors=b_factors
        )
        
        # Synthetic ddG (random but correlated with displacement magnitude)
        displacement_mag = np.linalg.norm(mut_positions - positions)
        ddg = np.random.normal(displacement_mag * 0.1, 0.5)  # Rough correlation
        
        data_point = MutationDataPoint(
            original_structure=orig_prot,
            mutated_structure=mut_prot,
            mutation_position=mutation_pos,
            original_aa=original_aa,
            target_aa=target_aa,
            ddg=ddg,
            source="synthetic"
        )
        
        data_points.append(data_point)
    
    return data_points
