"""
GPU memory layout optimization for OpenFold++.

This module provides utilities for optimizing memory access patterns,
tensor layouts, and memory bandwidth utilization on GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
import math
from dataclasses import dataclass


@dataclass
class MemoryLayoutConfig:
    """Configuration for memory layout optimization."""
    enable_memory_coalescing: bool = True
    prefer_channels_last: bool = True
    use_memory_efficient_attention: bool = True
    enable_tensor_fusion: bool = True
    optimize_for_mixed_precision: bool = True
    target_memory_bandwidth_gb: float = 900.0  # Target bandwidth (e.g., A100)
    cache_line_size_bytes: int = 128
    warp_size: int = 32


class MemoryLayoutOptimizer:
    """Optimizes tensor memory layouts for GPU efficiency."""
    
    def __init__(self, config: MemoryLayoutConfig = None):
        """
        Args:
            config: Memory layout optimization configuration
        """
        self.config = config or MemoryLayoutConfig()
        self.device_properties = self._get_device_properties()
        
    def _get_device_properties(self) -> Dict:
        """Get GPU device properties for optimization."""
        if not torch.cuda.is_available():
            return {}
        
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        return {
            'name': props.name,
            'major': props.major,
            'minor': props.minor,
            'total_memory': props.total_memory,
            'multiprocessor_count': props.multiprocessor_count,
            'max_threads_per_multiprocessor': props.max_threads_per_multiprocessor,
            'max_threads_per_block': props.max_threads_per_block,
            'warp_size': props.warp_size,
            'memory_clock_rate': getattr(props, 'memory_clock_rate', 0),
            'memory_bus_width': getattr(props, 'memory_bus_width', 0)
        }
    
    def optimize_tensor_layout(self, tensor: torch.Tensor, 
                              operation_type: str = "attention") -> torch.Tensor:
        """
        Optimize tensor memory layout for specific operations.
        
        Args:
            tensor: Input tensor to optimize
            operation_type: Type of operation ("attention", "linear", "conv", etc.)
            
        Returns:
            Optimized tensor with better memory layout
        """
        if not tensor.is_cuda:
            return tensor
        
        original_shape = tensor.shape
        original_stride = tensor.stride()
        
        # Choose optimal layout based on operation type
        if operation_type == "attention" and len(tensor.shape) >= 4:
            # For attention: optimize for [batch, heads, seq_len, head_dim]
            optimized = self._optimize_attention_layout(tensor)
        elif operation_type == "linear":
            # For linear layers: optimize for matrix multiplication
            optimized = self._optimize_linear_layout(tensor)
        elif operation_type == "conv" and len(tensor.shape) == 4:
            # For convolutions: use channels_last format
            optimized = self._optimize_conv_layout(tensor)
        else:
            # Default optimization
            optimized = self._optimize_default_layout(tensor)
        
        # Verify optimization didn't change semantics
        assert optimized.shape == original_shape, "Layout optimization changed tensor shape"
        
        return optimized
    
    def _optimize_attention_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize memory layout for attention operations."""
        if len(tensor.shape) != 4:
            return tensor
        
        # For attention tensors [batch, seq_i, seq_j, channels]
        # Optimize for coalesced access in attention computation
        
        # Check if current layout is already optimal
        if self._is_memory_coalesced(tensor):
            return tensor
        
        # Transpose to optimize memory access pattern
        # Move channels to second dimension for better coalescing
        if tensor.shape[-1] % self.config.warp_size == 0:
            # Channels divisible by warp size - good for coalescing
            return tensor.contiguous()
        else:
            # Pad channels to next multiple of warp size
            channels = tensor.shape[-1]
            padded_channels = math.ceil(channels / self.config.warp_size) * self.config.warp_size
            
            if padded_channels != channels:
                padding = padded_channels - channels
                tensor = F.pad(tensor, (0, padding))
        
        return tensor.contiguous()
    
    def _optimize_linear_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize memory layout for linear operations."""
        # For linear operations, ensure contiguous memory for GEMM
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Optimize for mixed precision if enabled
        if self.config.optimize_for_mixed_precision and tensor.dtype == torch.float32:
            # Consider converting to half precision for bandwidth optimization
            if self._should_use_half_precision(tensor):
                logging.info("Converting tensor to half precision for memory optimization")
                tensor = tensor.half()
        
        return tensor
    
    def _optimize_conv_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize memory layout for convolution operations."""
        if self.config.prefer_channels_last and len(tensor.shape) == 4:
            # Convert to channels_last format for better memory access
            try:
                if hasattr(tensor, 'memory_format') and tensor.memory_format != torch.channels_last:
                    tensor = tensor.to(memory_format=torch.channels_last)
                elif not hasattr(tensor, 'memory_format'):
                    # Fallback for older PyTorch versions
                    tensor = tensor.contiguous()
            except:
                tensor = tensor.contiguous()
        
        return tensor
    
    def _optimize_default_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        """Default memory layout optimization."""
        # Ensure tensor is contiguous
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        return tensor
    
    def _is_memory_coalesced(self, tensor: torch.Tensor) -> bool:
        """Check if tensor memory access is coalesced."""
        if not tensor.is_cuda:
            return True
        
        # Check stride pattern for coalesced access
        strides = tensor.stride()
        
        # For coalesced access, innermost dimension should have stride 1
        if strides[-1] != 1:
            return False
        
        # Check if strides follow expected pattern
        for i in range(len(strides) - 2, -1, -1):
            expected_stride = strides[i + 1] * tensor.shape[i + 1]
            if strides[i] != expected_stride:
                return False
        
        return True
    
    def _should_use_half_precision(self, tensor: torch.Tensor) -> bool:
        """Determine if tensor should use half precision."""
        # Use half precision for large tensors to save bandwidth
        tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
        
        # Use half precision for tensors larger than 100MB
        return tensor_size_mb > 100.0
    
    def optimize_attention_memory_pattern(self, 
                                        query: torch.Tensor,
                                        key: torch.Tensor, 
                                        value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optimize memory access patterns for attention computation.
        
        Args:
            query: Query tensor [batch, heads, seq_len, head_dim]
            key: Key tensor [batch, heads, seq_len, head_dim]
            value: Value tensor [batch, heads, seq_len, head_dim]
            
        Returns:
            Tuple of optimized (query, key, value) tensors
        """
        # Ensure all tensors have optimal layout for attention
        opt_query = self.optimize_tensor_layout(query, "attention")
        opt_key = self.optimize_tensor_layout(key, "attention")
        opt_value = self.optimize_tensor_layout(value, "attention")
        
        # Optimize for memory bandwidth
        if self.config.enable_tensor_fusion:
            # Pack QKV into single tensor for better memory locality
            qkv_packed = torch.stack([opt_query, opt_key, opt_value], dim=-2)
            qkv_packed = qkv_packed.contiguous()
            
            # Unpack with optimized layout
            opt_query, opt_key, opt_value = qkv_packed.unbind(dim=-2)
        
        return opt_query, opt_key, opt_value
    
    def create_memory_efficient_mask(self, 
                                   mask: torch.Tensor,
                                   target_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Create memory-efficient mask with optimal layout.
        
        Args:
            mask: Input mask tensor
            target_shape: Target shape for broadcasting
            
        Returns:
            Memory-optimized mask tensor
        """
        # Optimize mask storage and access patterns
        if mask.dtype != torch.bool:
            # Use boolean masks for memory efficiency
            mask = mask.bool()
        
        # Optimize mask shape for broadcasting
        optimized_mask = mask
        
        # Ensure mask can be efficiently broadcast
        if len(mask.shape) < len(target_shape):
            # Add dimensions for efficient broadcasting
            for _ in range(len(target_shape) - len(mask.shape)):
                optimized_mask = optimized_mask.unsqueeze(-1)
        
        # Make contiguous for efficient access
        optimized_mask = optimized_mask.contiguous()
        
        return optimized_mask
    
    def estimate_memory_bandwidth_utilization(self, 
                                            tensor: torch.Tensor,
                                            operation: str = "read") -> float:
        """
        Estimate memory bandwidth utilization for tensor operations.
        
        Args:
            tensor: Tensor to analyze
            operation: Type of operation ("read", "write", "readwrite")
            
        Returns:
            Estimated bandwidth utilization as fraction of peak
        """
        if not tensor.is_cuda or not self.device_properties:
            return 0.0
        
        # Calculate tensor size in bytes
        tensor_size_bytes = tensor.numel() * tensor.element_size()
        
        # Estimate memory transactions based on access pattern
        if self._is_memory_coalesced(tensor):
            # Coalesced access - optimal bandwidth utilization
            transactions = math.ceil(tensor_size_bytes / self.config.cache_line_size_bytes)
            effective_bytes = transactions * self.config.cache_line_size_bytes
        else:
            # Non-coalesced access - reduced efficiency
            # Assume worst case: one transaction per element
            transactions = tensor.numel()
            effective_bytes = transactions * self.config.cache_line_size_bytes
        
        # Calculate bandwidth requirement
        if operation == "readwrite":
            effective_bytes *= 2  # Both read and write
        
        # Estimate utilization (simplified model)
        theoretical_peak_gb = self.config.target_memory_bandwidth_gb
        required_bandwidth_gb = effective_bytes / (1024**3)
        
        utilization = min(required_bandwidth_gb / theoretical_peak_gb, 1.0)
        
        return utilization
    
    def get_optimization_report(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """
        Generate memory optimization report for a set of tensors.
        
        Args:
            tensors: Dictionary of tensor name -> tensor
            
        Returns:
            Optimization report dictionary
        """
        report = {
            'device_info': self.device_properties,
            'config': self.config,
            'tensor_analysis': {},
            'recommendations': []
        }
        
        total_memory_mb = 0
        coalesced_count = 0
        
        for name, tensor in tensors.items():
            if not tensor.is_cuda:
                continue
            
            size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
            is_coalesced = self._is_memory_coalesced(tensor)
            bandwidth_util = self.estimate_memory_bandwidth_utilization(tensor)
            
            report['tensor_analysis'][name] = {
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'size_mb': size_mb,
                'is_contiguous': tensor.is_contiguous(),
                'is_coalesced': is_coalesced,
                'memory_format': str(getattr(tensor, 'memory_format', 'contiguous_format')),
                'bandwidth_utilization': bandwidth_util
            }
            
            total_memory_mb += size_mb
            if is_coalesced:
                coalesced_count += 1
        
        # Generate recommendations
        coalesced_ratio = coalesced_count / len(tensors) if tensors else 0
        
        if coalesced_ratio < 0.8:
            report['recommendations'].append(
                "Consider optimizing tensor layouts for better memory coalescing"
            )
        
        if total_memory_mb > 1000:  # > 1GB
            report['recommendations'].append(
                "Consider using mixed precision to reduce memory bandwidth requirements"
            )
        
        report['summary'] = {
            'total_memory_mb': total_memory_mb,
            'coalesced_ratio': coalesced_ratio,
            'num_tensors': len(tensors)
        }
        
        return report


class MemoryEfficientAttention(nn.Module):
    """Memory-efficient attention implementation with optimized layouts."""
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 memory_optimizer: MemoryLayoutOptimizer = None):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            memory_optimizer: Memory layout optimizer
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.memory_optimizer = memory_optimizer or MemoryLayoutOptimizer()
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Memory-optimized attention forward pass.
        
        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor [batch, seq_len, embed_dim]
            value: Value tensor [batch, seq_len, embed_dim]
            attn_mask: Optional attention mask
            
        Returns:
            Attention output [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = query.shape
        
        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Optimize memory layout
        q, k, v = self.memory_optimizer.optimize_attention_memory_pattern(q, k, v)
        
        # Optimize mask if provided
        if attn_mask is not None:
            attn_mask = self.memory_optimizer.create_memory_efficient_mask(
                attn_mask, (batch_size, self.num_heads, seq_len, seq_len)
            )
        
        # Compute attention with memory-efficient implementation
        attn_output = self._memory_efficient_attention(q, k, v, attn_mask)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        output = self.out_proj(attn_output)
        
        return output
    
    def _memory_efficient_attention(self, 
                                  q: torch.Tensor,
                                  k: torch.Tensor, 
                                  v: torch.Tensor,
                                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Memory-efficient attention computation."""
        # Use scaled dot-product attention with memory optimizations
        scale = 1.0 / math.sqrt(self.head_dim)
        
        # Compute attention scores with optimal memory access
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match attention scores shape [batch, heads, seq_len, seq_len]
            if mask.dim() == 2:  # [batch, seq_len]
                # Create attention mask: [batch, seq_len] -> [batch, 1, seq_len, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
                mask = mask.expand(-1, -1, attn_scores.shape[-2], -1)  # [batch, 1, seq_len, seq_len]
                mask = mask.expand(-1, attn_scores.shape[1], -1, -1)  # [batch, heads, seq_len, seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output


def optimize_model_memory_layout(model: nn.Module, 
                                config: MemoryLayoutConfig = None) -> nn.Module:
    """
    Optimize memory layout for an entire model.
    
    Args:
        model: PyTorch model to optimize
        config: Memory optimization configuration
        
    Returns:
        Model with optimized memory layouts
    """
    optimizer = MemoryLayoutOptimizer(config)
    
    # Convert model to use memory-efficient operations
    def replace_attention_modules(module):
        for name, child in module.named_children():
            if isinstance(child, nn.MultiheadAttention):
                # Replace with memory-efficient version
                efficient_attn = MemoryEfficientAttention(
                    embed_dim=child.embed_dim,
                    num_heads=child.num_heads,
                    memory_optimizer=optimizer
                )
                setattr(module, name, efficient_attn)
                logging.info(f"Replaced {name} with memory-efficient attention")
            else:
                replace_attention_modules(child)
    
    replace_attention_modules(model)
    
    # Optimize model for mixed precision if enabled
    if config and config.optimize_for_mixed_precision:
        model = model.half()
        logging.info("Converted model to half precision")
    
    return model
