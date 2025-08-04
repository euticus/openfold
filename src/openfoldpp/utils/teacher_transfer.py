#!/usr/bin/env python3
"""
Teacher Transfer Utility for OpenFold++

This module handles transferring weights from a 48-layer teacher model
to a 24-layer student model for Phase B optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import OrderedDict
import copy


class TeacherTransfer:
    """
    Handles weight transfer from teacher (48-layer) to student (24-layer) models.
    
    Strategies:
    - even_layers: Copy even-numbered layers (0, 2, 4, ..., 46)
    - uniform_sample: Uniformly sample 24 layers from 48
    - first_last: Copy first 12 and last 12 layers
    - interpolate: Interpolate between adjacent layers
    """
    
    def __init__(
        self,
        strategy: str = "even_layers",
        interpolate_missing: bool = False,
        freeze_transferred: bool = False
    ):
        self.strategy = strategy
        self.interpolate_missing = interpolate_missing
        self.freeze_transferred = freeze_transferred
        
        # Create layer mapping based on strategy
        self.layer_mapping = self._create_layer_mapping()
        
        logging.info(f"TeacherTransfer initialized with strategy: {strategy}")
    
    def _create_layer_mapping(self) -> Dict[int, int]:
        """
        Create mapping from student layer indices to teacher layer indices.
        
        Returns:
            Dictionary mapping student_idx -> teacher_idx
        """
        mapping = {}
        
        if self.strategy == "even_layers":
            # Map to even-numbered layers: 0->0, 1->2, 2->4, ..., 23->46
            for student_idx in range(24):
                teacher_idx = student_idx * 2
                mapping[student_idx] = teacher_idx
                
        elif self.strategy == "uniform_sample":
            # Uniformly sample 24 layers from 48
            teacher_indices = np.linspace(0, 47, 24, dtype=int)
            for student_idx, teacher_idx in enumerate(teacher_indices):
                mapping[student_idx] = teacher_idx
                
        elif self.strategy == "first_last":
            # First 12 layers + last 12 layers
            for student_idx in range(12):
                mapping[student_idx] = student_idx  # First 12
            for student_idx in range(12, 24):
                mapping[student_idx] = 36 + (student_idx - 12)  # Last 12 (36-47)
                
        elif self.strategy == "interpolate":
            # Use even layers as base, will interpolate in transfer
            for student_idx in range(24):
                teacher_idx = student_idx * 2
                mapping[student_idx] = teacher_idx
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return mapping
    
    def transfer_weights(
        self,
        teacher_state_dict: Dict[str, torch.Tensor],
        student_model: nn.Module
    ) -> nn.Module:
        """
        Transfer weights from teacher to student model.
        
        Args:
            teacher_state_dict: State dict from 48-layer teacher model
            student_model: 24-layer student model to update
            
        Returns:
            Updated student model
        """
        logging.info("Starting teacher-to-student weight transfer...")
        
        # Get student state dict
        student_state_dict = student_model.state_dict()
        
        # Transfer EvoFormer block weights
        transferred_count = 0
        
        for student_idx, teacher_idx in self.layer_mapping.items():
            transferred_count += self._transfer_block_weights(
                teacher_state_dict,
                student_state_dict,
                teacher_idx,
                student_idx
            )
        
        # Transfer non-block weights (embeddings, final layers, etc.)
        self._transfer_non_block_weights(teacher_state_dict, student_state_dict)
        
        # Load updated weights into student model
        student_model.load_state_dict(student_state_dict)
        
        # Freeze transferred weights if requested
        if self.freeze_transferred:
            self._freeze_transferred_weights(student_model)
        
        logging.info(f"Weight transfer complete: {transferred_count} blocks transferred")
        return student_model
    
    def _transfer_block_weights(
        self,
        teacher_state_dict: Dict[str, torch.Tensor],
        student_state_dict: Dict[str, torch.Tensor],
        teacher_idx: int,
        student_idx: int
    ) -> int:
        """
        Transfer weights for a single EvoFormer block.
        
        Args:
            teacher_state_dict: Teacher model state dict
            student_state_dict: Student model state dict to update
            teacher_idx: Teacher block index
            student_idx: Student block index
            
        Returns:
            Number of parameters transferred
        """
        transferred = 0
        
        # Find all parameters for this block
        teacher_prefix = f"blocks.{teacher_idx}."
        student_prefix = f"blocks.{student_idx}."
        
        # Handle shared blocks if using weight sharing
        if "shared_blocks" in str(student_state_dict.keys()):
            shared_idx = student_idx // 4  # Assuming weight_sharing_interval = 4
            student_prefix = f"shared_blocks.{shared_idx}."
        
        for key, tensor in teacher_state_dict.items():
            if key.startswith(teacher_prefix):
                # Map to student key
                student_key = key.replace(teacher_prefix, student_prefix)
                
                if student_key in student_state_dict:
                    if self.strategy == "interpolate" and student_idx < 23:
                        # Interpolate with next layer
                        next_teacher_idx = (student_idx + 1) * 2
                        next_teacher_key = key.replace(f"blocks.{teacher_idx}.", f"blocks.{next_teacher_idx}.")
                        
                        if next_teacher_key in teacher_state_dict:
                            # Average the two layers
                            interpolated = (tensor + teacher_state_dict[next_teacher_key]) / 2
                            student_state_dict[student_key] = interpolated.clone()
                        else:
                            student_state_dict[student_key] = tensor.clone()
                    else:
                        # Direct copy
                        student_state_dict[student_key] = tensor.clone()
                    
                    transferred += tensor.numel()
        
        return transferred
    
    def _transfer_non_block_weights(
        self,
        teacher_state_dict: Dict[str, torch.Tensor],
        student_state_dict: Dict[str, torch.Tensor]
    ):
        """
        Transfer non-block weights (embeddings, final layers, etc.).
        
        Args:
            teacher_state_dict: Teacher model state dict
            student_state_dict: Student model state dict to update
        """
        # List of non-block prefixes to transfer
        non_block_prefixes = [
            "linear.",  # Final linear layer
            "embed.",   # Embeddings
            "norm.",    # Layer norms
            "pos_enc.", # Positional encoding
        ]
        
        for key, tensor in teacher_state_dict.items():
            # Check if this is a non-block parameter
            is_non_block = any(key.startswith(prefix) for prefix in non_block_prefixes)
            is_block = key.startswith("blocks.") or key.startswith("shared_blocks.")
            
            if is_non_block or not is_block:
                if key in student_state_dict:
                    # Check if shapes match
                    if tensor.shape == student_state_dict[key].shape:
                        student_state_dict[key] = tensor.clone()
                    else:
                        logging.warning(f"Shape mismatch for {key}: "
                                      f"teacher {tensor.shape} vs student {student_state_dict[key].shape}")
    
    def _freeze_transferred_weights(self, model: nn.Module):
        """
        Freeze transferred weights to prevent updates during initial training.
        
        Args:
            model: Student model with transferred weights
        """
        frozen_count = 0
        
        for name, param in model.named_parameters():
            # Freeze block parameters that were transferred
            if any(f"blocks.{i}." in name or f"shared_blocks.{i//4}." in name 
                   for i in self.layer_mapping.keys()):
                param.requires_grad = False
                frozen_count += 1
        
        logging.info(f"Frozen {frozen_count} transferred parameters")
    
    def unfreeze_all_weights(self, model: nn.Module):
        """
        Unfreeze all weights for full fine-tuning.
        
        Args:
            model: Model to unfreeze
        """
        for param in model.parameters():
            param.requires_grad = True
        
        logging.info("All weights unfrozen for fine-tuning")
    
    def validate_transfer(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        test_input: Tuple[torch.Tensor, ...]
    ) -> Dict[str, float]:
        """
        Validate that weight transfer preserves model behavior.
        
        Args:
            teacher_model: Original 48-layer model
            student_model: 24-layer model with transferred weights
            test_input: Test input tensors
            
        Returns:
            Dictionary of validation metrics
        """
        teacher_model.eval()
        student_model.eval()
        
        with torch.no_grad():
            # Get teacher outputs for mapped layers
            teacher_outputs = []
            student_outputs = []
            
            # Hook to capture intermediate outputs
            def capture_output(outputs_list):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        outputs_list.append(output[0].clone())  # MSA output
                    else:
                        outputs_list.append(output.clone())
                return hook
            
            # Register hooks for teacher
            teacher_hooks = []
            for student_idx, teacher_idx in self.layer_mapping.items():
                if hasattr(teacher_model, 'blocks'):
                    hook = teacher_model.blocks[teacher_idx].register_forward_hook(
                        capture_output(teacher_outputs)
                    )
                    teacher_hooks.append(hook)
            
            # Register hooks for student
            student_hooks = []
            if hasattr(student_model, 'blocks'):
                for i in range(len(student_model.blocks)):
                    hook = student_model.blocks[i].register_forward_hook(
                        capture_output(student_outputs)
                    )
                    student_hooks.append(hook)
            elif hasattr(student_model, 'shared_blocks'):
                for i in range(len(student_model.shared_blocks)):
                    hook = student_model.shared_blocks[i].register_forward_hook(
                        capture_output(student_outputs)
                    )
                    student_hooks.append(hook)
            
            # Run forward passes
            teacher_final = teacher_model(*test_input)
            student_final = student_model(*test_input)
            
            # Remove hooks
            for hook in teacher_hooks + student_hooks:
                hook.remove()
        
        # Calculate validation metrics
        metrics = {}
        
        # Compare final outputs
        if isinstance(teacher_final, tuple) and isinstance(student_final, tuple):
            for i, (t_out, s_out) in enumerate(zip(teacher_final, student_final)):
                mse = torch.mean((t_out - s_out) ** 2).item()
                cosine_sim = torch.cosine_similarity(
                    t_out.flatten(), s_out.flatten(), dim=0
                ).item()
                
                metrics[f"final_output_{i}_mse"] = mse
                metrics[f"final_output_{i}_cosine_sim"] = cosine_sim
        
        # Compare intermediate outputs
        if teacher_outputs and student_outputs:
            min_len = min(len(teacher_outputs), len(student_outputs))
            for i in range(min_len):
                mse = torch.mean((teacher_outputs[i] - student_outputs[i]) ** 2).item()
                metrics[f"layer_{i}_mse"] = mse
        
        return metrics


def load_teacher_model(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    """
    Load teacher model state dict from checkpoint.
    
    Args:
        checkpoint_path: Path to teacher model checkpoint
        
    Returns:
        Teacher model state dict
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    logging.info(f"Loaded teacher model from {checkpoint_path}")
    return state_dict


def save_student_model(
    student_model: nn.Module,
    save_path: Path,
    transfer_info: Dict = None
):
    """
    Save student model with transfer information.
    
    Args:
        student_model: Student model to save
        save_path: Path to save checkpoint
        transfer_info: Information about the transfer process
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': student_model.state_dict(),
        'transfer_info': transfer_info or {},
        'model_config': getattr(student_model, 'config', None)
    }
    
    torch.save(checkpoint, save_path)
    logging.info(f"Student model saved to {save_path}")


# Example usage
if __name__ == "__main__":
    # Test teacher transfer
    from openfoldpp.modules.slim_evoformer import create_slim_evoformer, SlimEvoFormerConfig
    
    # Create student model
    config = SlimEvoFormerConfig()
    student_model = create_slim_evoformer(config)
    
    # Create mock teacher state dict
    teacher_state_dict = {}
    for i in range(48):  # 48 teacher blocks
        teacher_state_dict[f"blocks.{i}.weight"] = torch.randn(256, 256)
        teacher_state_dict[f"blocks.{i}.bias"] = torch.randn(256)
    
    # Initialize teacher transfer
    transfer = TeacherTransfer(strategy="even_layers")
    
    # Transfer weights
    student_model = transfer.transfer_weights(teacher_state_dict, student_model)
    
    print("✅ Teacher transfer test complete!")
    print(f"   Layer mapping: {len(transfer.layer_mapping)} mappings")
    print(f"   Student parameters: {student_model.count_parameters():,}")
