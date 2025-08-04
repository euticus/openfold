#!/usr/bin/env python3
"""
LoRA (Low-Rank Adaptation) for OpenFold++

This module implements LoRA adapters for efficient fine-tuning of EvoFormer layers.
Only trains low-rank adapters (rank=8) to save VRAM during distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    rank: int = 8  # Low-rank dimension
    alpha: float = 16.0  # Scaling factor
    dropout: float = 0.1
    target_modules: List[str] = None  # Modules to apply LoRA to
    bias: str = "none"  # "none", "all", "lora_only"
    task_type: str = "FEATURE_EXTRACTION"
    

class LoRALayer(nn.Module):
    """
    LoRA layer implementation.
    
    Adds low-rank adaptation to existing linear layers:
    h = W_0 * x + (B * A) * x * (alpha / rank)
    
    Where:
    - W_0: Original frozen weights
    - A, B: Low-rank adaptation matrices
    - alpha: Scaling factor
    - rank: Low-rank dimension
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA parameters."""
        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layer.
        
        Args:
            x: Input tensor
            
        Returns:
            LoRA adaptation output
        """
        # LoRA computation: B @ A @ x
        result = F.linear(x, self.lora_A.T)  # x @ A.T
        result = self.dropout(result)
        result = F.linear(result, self.lora_B.T)  # result @ B.T
        
        return result * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    Combines original linear layer (frozen) with LoRA adaptation (trainable).
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        freeze_original: bool = True
    ):
        super().__init__()
        
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # Original layer (typically frozen)
        self.original_layer = original_layer
        if freeze_original:
            for param in self.original_layer.parameters():
                param.requires_grad = False
        
        # LoRA adaptation
        self.lora = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Track if LoRA is enabled
        self.lora_enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining original layer and LoRA.
        
        Args:
            x: Input tensor
            
        Returns:
            Combined output
        """
        # Original layer output
        original_output = self.original_layer(x)
        
        # Add LoRA adaptation if enabled
        if self.lora_enabled:
            lora_output = self.lora(x)
            return original_output + lora_output
        else:
            return original_output
    
    def enable_lora(self):
        """Enable LoRA adaptation."""
        self.lora_enabled = True
    
    def disable_lora(self):
        """Disable LoRA adaptation."""
        self.lora_enabled = False
    
    def merge_lora(self):
        """
        Merge LoRA weights into original layer.
        
        This permanently applies the LoRA adaptation to the original weights.
        """
        if not self.lora_enabled:
            return
        
        # Compute LoRA weight matrix: B @ A * scaling
        lora_weight = self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
        
        # Add to original weights
        with torch.no_grad():
            self.original_layer.weight.data += lora_weight
        
        # Reset LoRA parameters
        self.lora.reset_parameters()
        
        logging.info(f"Merged LoRA weights into linear layer ({self.in_features} -> {self.out_features})")


class LoRAMultiheadAttention(nn.Module):
    """
    MultiheadAttention with LoRA adapters on Q, K, V projections.
    """
    
    def __init__(
        self,
        original_attention: nn.MultiheadAttention,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        target_modules: List[str] = None
    ):
        super().__init__()
        
        self.original_attention = original_attention
        
        # Freeze original attention
        for param in self.original_attention.parameters():
            param.requires_grad = False
        
        # Default target modules
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        
        # Add LoRA to specified projections
        self.lora_adapters = nn.ModuleDict()
        
        if "q_proj" in target_modules and hasattr(original_attention, 'q_proj_weight'):
            self.lora_adapters["q_proj"] = LoRALayer(
                original_attention.embed_dim, original_attention.embed_dim, rank, alpha, dropout
            )
        
        if "k_proj" in target_modules and hasattr(original_attention, 'k_proj_weight'):
            self.lora_adapters["k_proj"] = LoRALayer(
                original_attention.kdim or original_attention.embed_dim, 
                original_attention.embed_dim, rank, alpha, dropout
            )
        
        if "v_proj" in target_modules and hasattr(original_attention, 'v_proj_weight'):
            self.lora_adapters["v_proj"] = LoRALayer(
                original_attention.vdim or original_attention.embed_dim,
                original_attention.embed_dim, rank, alpha, dropout
            )
        
        if "out_proj" in target_modules:
            self.lora_adapters["out_proj"] = LoRALayer(
                original_attention.embed_dim, original_attention.embed_dim, rank, alpha, dropout
            )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with LoRA-adapted attention.
        
        Args:
            query: Query tensor
            key: Key tensor  
            value: Value tensor
            key_padding_mask: Optional padding mask
            need_weights: Whether to return attention weights
            attn_mask: Optional attention mask
            average_attn_weights: Whether to average attention weights
            
        Returns:
            Attention output and optional weights
        """
        
        # Apply LoRA adaptations to inputs if available
        if "q_proj" in self.lora_adapters:
            query = query + self.lora_adapters["q_proj"](query)
        
        if "k_proj" in self.lora_adapters:
            key = key + self.lora_adapters["k_proj"](key)
        
        if "v_proj" in self.lora_adapters:
            value = value + self.lora_adapters["v_proj"](value)
        
        # Original attention computation
        attn_output, attn_weights = self.original_attention(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights
        )
        
        # Apply LoRA to output projection if available
        if "out_proj" in self.lora_adapters:
            attn_output = attn_output + self.lora_adapters["out_proj"](attn_output)
        
        return attn_output, attn_weights


class LoRAWrapper:
    """
    Utility class to wrap existing models with LoRA adapters.
    """
    
    @staticmethod
    def wrap_linear_layers(
        model: nn.Module,
        config: LoRAConfig,
        target_modules: Optional[List[str]] = None
    ) -> nn.Module:
        """
        Wrap linear layers in a model with LoRA adapters.
        
        Args:
            model: Model to wrap
            config: LoRA configuration
            target_modules: Optional list of module names to target
            
        Returns:
            Model with LoRA adapters
        """
        
        if target_modules is None:
            target_modules = config.target_modules or ["linear", "Linear"]
        
        def replace_linear_with_lora(module, name=""):
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                if isinstance(child, nn.Linear):
                    # Check if this module should be wrapped
                    should_wrap = any(target in full_name.lower() for target in target_modules)
                    
                    if should_wrap:
                        # Replace with LoRA linear
                        lora_linear = LoRALinear(
                            original_layer=child,
                            rank=config.rank,
                            alpha=config.alpha,
                            dropout=config.dropout,
                            freeze_original=True
                        )
                        setattr(module, child_name, lora_linear)
                        logging.info(f"Wrapped {full_name} with LoRA (rank={config.rank})")
                
                else:
                    # Recursively process child modules
                    replace_linear_with_lora(child, full_name)
        
        replace_linear_with_lora(model)
        return model
    
    @staticmethod
    def wrap_attention_layers(
        model: nn.Module,
        config: LoRAConfig
    ) -> nn.Module:
        """
        Wrap attention layers in a model with LoRA adapters.
        
        Args:
            model: Model to wrap
            config: LoRA configuration
            
        Returns:
            Model with LoRA attention adapters
        """
        
        def replace_attention_with_lora(module, name=""):
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                if isinstance(child, nn.MultiheadAttention):
                    # Replace with LoRA attention
                    lora_attention = LoRAMultiheadAttention(
                        original_attention=child,
                        rank=config.rank,
                        alpha=config.alpha,
                        dropout=config.dropout
                    )
                    setattr(module, child_name, lora_attention)
                    logging.info(f"Wrapped {full_name} with LoRA attention (rank={config.rank})")
                
                else:
                    # Recursively process child modules
                    replace_attention_with_lora(child, full_name)
        
        replace_attention_with_lora(model)
        return model
    
    @staticmethod
    def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
        """
        Count LoRA parameters in a model.
        
        Args:
            model: Model with LoRA adapters
            
        Returns:
            Dictionary with parameter counts
        """
        
        total_params = 0
        trainable_params = 0
        lora_params = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            
            if param.requires_grad:
                trainable_params += param.numel()
                
                if "lora" in name.lower():
                    lora_params += param.numel()
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'lora_parameters': lora_params,
            'trainable_percentage': trainable_params / total_params * 100,
            'lora_percentage': lora_params / total_params * 100
        }
    
    @staticmethod
    def save_lora_weights(model: nn.Module, save_path: str):
        """
        Save only LoRA weights from a model.
        
        Args:
            model: Model with LoRA adapters
            save_path: Path to save LoRA weights
        """
        
        lora_state_dict = {}
        
        for name, param in model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                lora_state_dict[name] = param.data.clone()
        
        torch.save(lora_state_dict, save_path)
        logging.info(f"Saved {len(lora_state_dict)} LoRA parameters to {save_path}")
    
    @staticmethod
    def load_lora_weights(model: nn.Module, load_path: str):
        """
        Load LoRA weights into a model.
        
        Args:
            model: Model with LoRA adapters
            load_path: Path to LoRA weights
        """
        
        lora_state_dict = torch.load(load_path, map_location='cpu')
        
        # Load only LoRA parameters
        model_state_dict = model.state_dict()
        for name, param in lora_state_dict.items():
            if name in model_state_dict:
                model_state_dict[name].copy_(param)
        
        logging.info(f"Loaded {len(lora_state_dict)} LoRA parameters from {load_path}")


# Factory functions
def create_lora_config(
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: List[str] = None
) -> LoRAConfig:
    """
    Create LoRA configuration.
    
    Args:
        rank: Low-rank dimension
        alpha: Scaling factor
        target_modules: Target module names
        
    Returns:
        LoRA configuration
    """
    return LoRAConfig(
        rank=rank,
        alpha=alpha,
        target_modules=target_modules or ["linear", "attention"]
    )


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig = None
) -> nn.Module:
    """
    Apply LoRA adapters to a model.
    
    Args:
        model: Model to adapt
        config: LoRA configuration
        
    Returns:
        Model with LoRA adapters
    """
    
    if config is None:
        config = create_lora_config()
    
    # Wrap linear layers
    model = LoRAWrapper.wrap_linear_layers(model, config)
    
    # Wrap attention layers
    model = LoRAWrapper.wrap_attention_layers(model, config)
    
    # Log parameter statistics
    param_stats = LoRAWrapper.count_lora_parameters(model)
    logging.info(f"LoRA adaptation complete:")
    logging.info(f"  Total parameters: {param_stats['total_parameters']:,}")
    logging.info(f"  Trainable parameters: {param_stats['trainable_parameters']:,} "
                f"({param_stats['trainable_percentage']:.1f}%)")
    logging.info(f"  LoRA parameters: {param_stats['lora_parameters']:,} "
                f"({param_stats['lora_percentage']:.1f}%)")
    
    return model


# Example usage and testing
if __name__ == "__main__":
    # Test LoRA implementation
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(256, 512)
            self.linear2 = nn.Linear(512, 256)
            self.attention = nn.MultiheadAttention(256, 8)
        
        def forward(self, x):
            x = self.linear1(x)
            x = F.relu(x)
            x = self.linear2(x)
            attn_out, _ = self.attention(x, x, x)
            return attn_out
    
    # Create model and apply LoRA
    model = SimpleModel()
    
    # Count original parameters
    original_params = sum(p.numel() for p in model.parameters())
    
    # Apply LoRA
    config = create_lora_config(rank=8, alpha=16.0)
    lora_model = apply_lora_to_model(model, config)
    
    # Test forward pass
    test_input = torch.randn(10, 32, 256)
    output = lora_model(test_input)
    
    print("✅ LoRA test successful!")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Original parameters: {original_params:,}")
    
    # Check parameter statistics
    param_stats = LoRAWrapper.count_lora_parameters(lora_model)
    print(f"   LoRA parameters: {param_stats['lora_parameters']:,}")
    print(f"   Trainable percentage: {param_stats['trainable_percentage']:.1f}%")
