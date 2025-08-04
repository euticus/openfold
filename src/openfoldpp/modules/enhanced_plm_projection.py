#!/usr/bin/env python3
"""
Enhanced PLM Projection for ESM-2-3B

This module provides enhanced projection layers to map ESM-2-3B embeddings
(2560-dim) to OpenFold++ MSA space with better information preservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import math


@dataclass
class EnhancedProjectionConfig:
    """Configuration for enhanced PLM projection."""
    input_dim: int = 2560  # ESM-2-3B embedding dimension
    output_dim: int = 256  # Target MSA dimension
    projection_type: str = "multi_layer"  # "linear", "multi_layer", "attention", "bottleneck"
    hidden_dim: int = 512  # Hidden dimension for multi-layer projection
    num_layers: int = 2  # Number of projection layers
    dropout: float = 0.1
    use_layer_norm: bool = True
    use_residual: bool = True
    activation: str = "gelu"  # "relu", "gelu", "swish"


class LinearProjection(nn.Module):
    """Simple linear projection."""
    
    def __init__(self, config: EnhancedProjectionConfig):
        super().__init__()
        
        self.projection = nn.Linear(config.input_dim, config.output_dim)
        self.layer_norm = nn.LayerNorm(config.output_dim) if config.use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class MultiLayerProjection(nn.Module):
    """Multi-layer projection with residual connections."""
    
    def __init__(self, config: EnhancedProjectionConfig):
        super().__init__()
        
        self.config = config
        
        # Activation function
        if config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "gelu":
            self.activation = nn.GELU()
        elif config.activation == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()
        
        # Build layers
        layers = []
        
        # Input projection
        layers.append(nn.Linear(config.input_dim, config.hidden_dim))
        layers.append(nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity())
        layers.append(self.activation)
        layers.append(nn.Dropout(config.dropout))
        
        # Hidden layers
        for _ in range(config.num_layers - 1):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity())
            layers.append(self.activation)
            layers.append(nn.Dropout(config.dropout))
        
        # Output projection
        layers.append(nn.Linear(config.hidden_dim, config.output_dim))
        layers.append(nn.LayerNorm(config.output_dim) if config.use_layer_norm else nn.Identity())
        
        self.layers = nn.Sequential(*layers)
        
        # Residual connection (if dimensions match)
        if config.use_residual and config.input_dim == config.output_dim:
            self.residual = True
        else:
            self.residual = False
            if config.use_residual:
                # Learnable residual projection
                self.residual_proj = nn.Linear(config.input_dim, config.output_dim)
                self.residual = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.layers(x)
        
        if self.residual:
            if hasattr(self, 'residual_proj'):
                residual = self.residual_proj(x)
            else:
                residual = x
            output = output + residual
        
        return output


class AttentionProjection(nn.Module):
    """Attention-based projection for better information selection."""
    
    def __init__(self, config: EnhancedProjectionConfig):
        super().__init__()
        
        self.config = config
        
        # Multi-head attention for feature selection
        self.attention = nn.MultiheadAttention(
            embed_dim=config.input_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.LayerNorm(config.output_dim) if config.use_layer_norm else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention for feature refinement
        attended, _ = self.attention(x, x, x)
        
        # Residual connection
        x = x + attended
        
        # Project to output dimension
        output = self.projection(x)
        
        return output


class BottleneckProjection(nn.Module):
    """Bottleneck projection for efficient dimension reduction."""
    
    def __init__(self, config: EnhancedProjectionConfig):
        super().__init__()
        
        bottleneck_dim = config.hidden_dim // 2
        
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.LayerNorm(config.output_dim) if config.use_layer_norm else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode to bottleneck
        encoded = self.encoder(x)
        
        # Decode to output
        output = self.decoder(encoded)
        
        return output


class EnhancedPLMProjector(nn.Module):
    """
    Enhanced PLM projector for ESM-2-3B embeddings.
    
    Maps 2560-dim ESM-2-3B embeddings to OpenFold++ MSA space
    with better information preservation than simple linear projection.
    """
    
    def __init__(self, config: EnhancedProjectionConfig = None):
        super().__init__()
        
        self.config = config or EnhancedProjectionConfig()
        
        # Create projection layer based on type
        if self.config.projection_type == "linear":
            self.projector = LinearProjection(self.config)
        elif self.config.projection_type == "multi_layer":
            self.projector = MultiLayerProjection(self.config)
        elif self.config.projection_type == "attention":
            self.projector = AttentionProjection(self.config)
        elif self.config.projection_type == "bottleneck":
            self.projector = BottleneckProjection(self.config)
        else:
            raise ValueError(f"Unknown projection type: {self.config.projection_type}")
        
        # Initialize weights
        self._init_weights()
        
        logging.info(f"Enhanced PLM projector created: {self.config.projection_type}")
        logging.info(f"  Input dim: {self.config.input_dim}")
        logging.info(f"  Output dim: {self.config.output_dim}")
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        
        def init_recursive(module):
            if isinstance(module, nn.Linear):
                # Xavier initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        self.apply(init_recursive)
    
    def forward(self, plm_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project PLM embeddings to MSA space.
        
        Args:
            plm_embeddings: [batch_size, seq_len, input_dim] PLM embeddings
            
        Returns:
            [batch_size, seq_len, output_dim] projected embeddings
        """
        
        # Validate input dimensions
        if plm_embeddings.size(-1) != self.config.input_dim:
            raise ValueError(f"Expected input dim {self.config.input_dim}, got {plm_embeddings.size(-1)}")
        
        # Project embeddings
        projected = self.projector(plm_embeddings)
        
        return projected
    
    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_memory_footprint(self) -> Dict[str, float]:
        """Get memory footprint in MB."""
        
        param_count = self.get_parameter_count()
        param_size_mb = param_count * 4 / (1024**2)  # 4 bytes per float32
        
        return {
            'parameter_count': param_count,
            'parameter_size_mb': param_size_mb,
            'projection_type': self.config.projection_type
        }


def create_enhanced_plm_projector(
    projection_type: str = "multi_layer",
    input_dim: int = 2560,
    output_dim: int = 256,
    **kwargs
) -> EnhancedPLMProjector:
    """
    Factory function to create enhanced PLM projector.
    
    Args:
        projection_type: Type of projection ("linear", "multi_layer", "attention", "bottleneck")
        input_dim: Input embedding dimension
        output_dim: Output MSA dimension
        **kwargs: Additional configuration parameters
        
    Returns:
        EnhancedPLMProjector instance
    """
    
    config = EnhancedProjectionConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        projection_type=projection_type,
        **kwargs
    )
    
    return EnhancedPLMProjector(config)


# Benchmark different projection types
def benchmark_projection_types(
    input_dim: int = 2560,
    output_dim: int = 256,
    batch_size: int = 2,
    seq_len: int = 100
) -> Dict[str, Dict[str, float]]:
    """Benchmark different projection types."""
    
    projection_types = ["linear", "multi_layer", "attention", "bottleneck"]
    results = {}
    
    # Create test input
    test_input = torch.randn(batch_size, seq_len, input_dim)
    
    for proj_type in projection_types:
        try:
            # Create projector
            projector = create_enhanced_plm_projector(
                projection_type=proj_type,
                input_dim=input_dim,
                output_dim=output_dim
            )
            projector.eval()
            
            # Benchmark inference
            import time
            
            # Warmup
            with torch.no_grad():
                _ = projector(test_input)
            
            # Time inference
            start_time = time.time()
            with torch.no_grad():
                output = projector(test_input)
            inference_time = time.time() - start_time
            
            # Get memory footprint
            memory_info = projector.get_memory_footprint()
            
            results[proj_type] = {
                'inference_time_ms': inference_time * 1000,
                'parameter_count': memory_info['parameter_count'],
                'parameter_size_mb': memory_info['parameter_size_mb'],
                'output_shape': list(output.shape)
            }
            
        except Exception as e:
            results[proj_type] = {'error': str(e)}
    
    return results


# Example usage and testing
if __name__ == "__main__":
    print("🔧 Testing Enhanced PLM Projection")
    print("=" * 50)
    
    # Test different projection types
    results = benchmark_projection_types()
    
    print("📊 Projection Type Comparison:")
    for proj_type, metrics in results.items():
        if 'error' in metrics:
            print(f"   {proj_type}: ❌ {metrics['error']}")
        else:
            print(f"   {proj_type}:")
            print(f"     Time: {metrics['inference_time_ms']:.2f}ms")
            print(f"     Params: {metrics['parameter_count']:,}")
            print(f"     Size: {metrics['parameter_size_mb']:.2f}MB")
    
    # Test specific projector
    print(f"\n🧪 Testing Multi-Layer Projector:")
    
    projector = create_enhanced_plm_projector(
        projection_type="multi_layer",
        input_dim=2560,
        output_dim=256,
        hidden_dim=512,
        num_layers=2
    )
    
    # Test forward pass
    test_input = torch.randn(2, 100, 2560)  # [batch, seq_len, esm_3b_dim]
    
    with torch.no_grad():
        output = projector(test_input)
    
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {projector.get_parameter_count():,}")
    
    print(f"\n✅ Enhanced PLM Projection Ready!")
    print(f"   Supports ESM-2-3B (2560-dim) → MSA (256-dim)")
    print(f"   Multiple projection strategies available")
