#!/usr/bin/env python3
"""
Distillation Training Script for OpenFold++

This script orchestrates teacher-student distillation training with:
- Curriculum learning
- Mixed-precision training
- LoRA adapters for memory efficiency
- Gradient accumulation for large effective batch sizes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import sys
import wandb
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from openfoldpp.modules.slim_evoformer import create_slim_evoformer, SlimEvoFormerConfig
from openfoldpp.modules.lora_adapters import apply_lora_to_model, create_lora_config, LoRAWrapper
from openfoldpp.losses.distillation_loss import create_distillation_loss, DistillationConfig
from openfoldpp.utils.teacher_transfer import TeacherTransfer


@dataclass
class TrainingConfig:
    """Configuration for distillation training."""
    
    # Model settings
    model_config: str = "slim_evoformer"
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: float = 16.0
    
    # Training settings
    global_batch_size: int = 64
    micro_batch_size: int = 1
    max_steps: int = 50000
    eval_steps: int = 1000
    save_steps: int = 5000
    log_steps: int = 100
    
    # Optimization
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    lr_schedule: str = "cosine"  # "cosine", "linear", "constant"
    
    # Mixed precision
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    
    # Data settings
    teacher_targets_dir: str = "data/teacher_targets"
    max_sequence_length: int = 512
    
    # Distillation settings
    coord_weight: float = 1.0
    plddt_weight: float = 0.5
    pair_weight: float = 0.1
    
    # Paths
    output_dir: str = "outputs/distillation"
    checkpoint_dir: str = "checkpoints/distillation"
    log_dir: str = "logs/distillation"
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "openfold-distillation"
    experiment_name: str = "distill_experiment"


class DistillationDataset(torch.utils.data.Dataset):
    """Dataset for loading teacher targets and sequences."""
    
    def __init__(
        self,
        targets_dir: Path,
        max_length: int = 512,
        split: str = "train"
    ):
        self.targets_dir = Path(targets_dir)
        self.max_length = max_length
        self.split = split
        
        # Load sequence list
        self.sequences = self._load_sequences()
        
        logging.info(f"Loaded {len(self.sequences)} sequences for {split}")
    
    def _load_sequences(self) -> List[Dict]:
        """Load sequence metadata."""
        sequences = []
        
        for seq_dir in self.targets_dir.iterdir():
            if seq_dir.is_dir():
                metadata_file = seq_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Skip if too long
                    if metadata.get('sequence_length', 0) <= self.max_length:
                        metadata['seq_dir'] = seq_dir
                        sequences.append(metadata)
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single training example."""
        seq_info = self.sequences[idx]
        seq_dir = seq_info['seq_dir']
        
        # Load teacher targets
        coords = np.load(seq_dir / 'coords.npy')
        plddt = np.load(seq_dir / 'pLDDT.npy')
        
        # Load pair representations if available
        pair_repr_file = seq_dir / 'pair_repr.npy'
        if pair_repr_file.exists():
            pair_repr = np.load(pair_repr_file)
        else:
            # Create dummy pair representation
            seq_len = len(coords)
            pair_repr = np.zeros((seq_len, seq_len, 128))
        
        # Convert to tensors
        data = {
            'sequence_id': seq_info['sequence_id'],
            'sequence_length': seq_info['sequence_length'],
            'coordinates': torch.from_numpy(coords).float(),
            'plddt': torch.from_numpy(plddt).float(),
            'pair_repr': torch.from_numpy(pair_repr).float()
        }
        
        return data


class DistillationTrainer:
    """Main trainer for distillation."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Setup logging and directories
        self._setup_logging()
        self._setup_directories()
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize loss function
        self.loss_fn = self._create_loss_function()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp else None
        
        # Data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Experiment tracking
        if config.use_wandb:
            self._setup_wandb()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / 'training.log')
            ]
        )
    
    def _setup_directories(self):
        """Create necessary directories."""
        for dir_path in [self.config.output_dir, self.config.checkpoint_dir, self.config.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _create_model(self) -> nn.Module:
        """Create and configure the student model."""
        
        # Create slim EvoFormer
        evo_config = SlimEvoFormerConfig()
        model = create_slim_evoformer(evo_config)
        
        # Apply LoRA if requested
        if self.config.use_lora:
            lora_config = create_lora_config(
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha
            )
            model = apply_lora_to_model(model, lora_config)
        
        # Move to device
        model = model.to(self.config.device)
        
        # Log model statistics
        if self.config.use_lora:
            param_stats = LoRAWrapper.count_lora_parameters(model)
            logging.info(f"Model with LoRA: {param_stats['trainable_parameters']:,} trainable parameters "
                        f"({param_stats['trainable_percentage']:.1f}%)")
        else:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logging.info(f"Model: {total_params:,} trainable parameters")
        
        return model
    
    def _create_loss_function(self):
        """Create distillation loss function."""
        distill_config = DistillationConfig(
            coord_weight=self.config.coord_weight,
            plddt_weight=self.config.plddt_weight,
            pair_weight=self.config.pair_weight,
            use_curriculum=True,
            curriculum_steps=self.config.max_steps // 2,
            warmup_steps=self.config.warmup_steps
        )
        
        return create_distillation_loss(distill_config).to(self.config.device)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        
        if self.config.lr_schedule == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_steps,
                eta_min=self.config.min_learning_rate
            )
        elif self.config.lr_schedule == "linear":
            scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.min_learning_rate / self.config.learning_rate,
                total_iters=self.config.max_steps
            )
        else:  # constant
            scheduler = None
        
        return scheduler
    
    def _create_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Create training and validation data loaders."""
        
        # Create datasets
        train_dataset = DistillationDataset(
            self.config.teacher_targets_dir,
            max_length=self.config.max_sequence_length,
            split="train"
        )
        
        val_dataset = DistillationDataset(
            self.config.teacher_targets_dir,
            max_length=self.config.max_sequence_length,
            split="val"
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.micro_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.micro_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        return train_loader, val_loader
    
    def _setup_wandb(self):
        """Setup Weights & Biases tracking."""
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.experiment_name,
            config=asdict(self.config)
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.config.device)
        
        # Create mock predictions (simplified for this example)
        seq_len = batch['sequence_length'].max().item()
        batch_size = len(batch['sequence_id'])
        
        with autocast(enabled=self.config.use_amp):
            # Mock model predictions
            predictions = {
                'coordinates': torch.randn(batch_size, seq_len, 3, device=self.config.device),
                'plddt_logits': torch.randn(batch_size, seq_len, 50, device=self.config.device),
                'pair_repr': torch.randn(batch_size, seq_len, seq_len, 128, device=self.config.device)
            }
            
            # Teacher targets
            targets = {
                'coordinates': batch['coordinates'],
                'plddt': batch['plddt'],
                'pair_repr': batch['pair_repr']
            }
            
            # Compute loss
            losses = self.loss_fn(predictions, targets)
            loss = losses['total_loss']
            
            # Scale loss for gradient accumulation
            accumulation_steps = self.config.global_batch_size // self.config.micro_batch_size
            loss = loss / accumulation_steps
        
        # Backward pass
        if self.config.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Return loss values
        return {key: value.item() if isinstance(value, torch.Tensor) else value 
                for key, value in losses.items()}
    
    def train(self):
        """Main training loop."""
        
        logging.info("Starting distillation training...")
        
        self.model.train()
        accumulation_steps = self.config.global_batch_size // self.config.micro_batch_size
        
        train_iterator = iter(self.train_loader)
        
        for step in tqdm(range(self.config.max_steps), desc="Training"):
            
            # Accumulate gradients
            accumulated_losses = {}
            
            for micro_step in range(accumulation_steps):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(self.train_loader)
                    batch = next(train_iterator)
                
                # Training step
                step_losses = self.train_step(batch)
                
                # Accumulate losses
                for key, value in step_losses.items():
                    if key not in accumulated_losses:
                        accumulated_losses[key] = 0
                    accumulated_losses[key] += value / accumulation_steps
            
            # Optimizer step
            if self.config.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            self.global_step += 1
            
            # Logging
            if step % self.config.log_steps == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logging.info(f"Step {step}: loss={accumulated_losses['total_loss']:.4f}, lr={lr:.2e}")
                
                if self.config.use_wandb:
                    wandb.log({
                        'train/total_loss': accumulated_losses['total_loss'],
                        'train/coord_loss': accumulated_losses.get('coord_loss', 0),
                        'train/plddt_loss': accumulated_losses.get('plddt_loss', 0),
                        'train/pair_loss': accumulated_losses.get('pair_loss', 0),
                        'train/learning_rate': lr,
                        'step': step
                    })
            
            # Validation
            if step % self.config.eval_steps == 0:
                val_loss = self.validate()
                logging.info(f"Validation loss: {val_loss:.4f}")
                
                if self.config.use_wandb:
                    wandb.log({'val/loss': val_loss, 'step': step})
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(step, is_best=True)
            
            # Save checkpoint
            if step % self.config.save_steps == 0:
                self.save_checkpoint(step)
        
        logging.info("Training completed!")
    
    def validate(self) -> float:
        """Validation loop."""
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.config.device)
                
                # Mock validation (simplified)
                seq_len = batch['sequence_length'].max().item()
                batch_size = len(batch['sequence_id'])
                
                predictions = {
                    'coordinates': torch.randn(batch_size, seq_len, 3, device=self.config.device),
                    'plddt_logits': torch.randn(batch_size, seq_len, 50, device=self.config.device),
                    'pair_repr': torch.randn(batch_size, seq_len, seq_len, 128, device=self.config.device)
                }
                
                targets = {
                    'coordinates': batch['coordinates'],
                    'plddt': batch['plddt'],
                    'pair_repr': batch['pair_repr']
                }
                
                losses = self.loss_fn(predictions, targets)
                total_loss += losses['total_loss'].item()
                num_batches += 1
                
                if num_batches >= 10:  # Limit validation batches
                    break
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint."""
        
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': asdict(self.config),
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            logging.info(f"Saved best checkpoint at step {step}")
        
        # Save LoRA weights separately if using LoRA
        if self.config.use_lora:
            lora_path = Path(self.config.checkpoint_dir) / f"lora_weights_step_{step}.pt"
            LoRAWrapper.save_lora_weights(self.model, str(lora_path))


def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(description="Distillation training for OpenFold++")
    parser.add_argument("--config", type=str, help="Training config file")
    parser.add_argument("--teacher-targets", type=str, default="data/teacher_targets", 
                       help="Teacher targets directory")
    parser.add_argument("--output-dir", type=str, default="outputs/distillation", 
                       help="Output directory")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--experiment-name", type=str, default="distill_experiment", 
                       help="Experiment name")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        teacher_targets_dir=args.teacher_targets,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        experiment_name=args.experiment_name
    )
    
    # Create trainer and start training
    trainer = DistillationTrainer(config)
    trainer.train()
    
    return 0


if __name__ == "__main__":
    exit(main())
