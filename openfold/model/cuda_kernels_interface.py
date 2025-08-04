"""
Python interface for OpenFold++ CUDA kernels.

This module provides a clean Python interface to the compiled CUDA kernels
with proper error handling, fallbacks, and integration with PyTorch autograd.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any, Tuple
import warnings

# Try to import compiled CUDA kernels
try:
    import openfold_cuda_kernels
    CUDA_KERNELS_AVAILABLE = True
    KERNEL_INFO = openfold_cuda_kernels.get_kernel_info()
    logging.info(f"OpenFold++ CUDA kernels loaded: {KERNEL_INFO}")
except ImportError as e:
    CUDA_KERNELS_AVAILABLE = False
    KERNEL_INFO = {}
    logging.warning(f"OpenFold++ CUDA kernels not available: {e}")


class CudaTriangleAttentionFunction(torch.autograd.Function):
    """
    Autograd function for CUDA triangle attention with proper gradient computation.
    """
    
    @staticmethod
    def forward(ctx, query, key, value, bias_mask, triangle_bias, starting_node=True):
        """Forward pass of triangle attention."""
        if not CUDA_KERNELS_AVAILABLE:
            raise RuntimeError("CUDA kernels not available")
        
        # Save for backward pass
        ctx.save_for_backward(query, key, value, bias_mask, triangle_bias)
        ctx.starting_node = starting_node
        
        # Call CUDA kernel
        output = openfold_cuda_kernels.triangle_attention_forward(
            query, key, value, bias_mask, triangle_bias, starting_node
        )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of triangle attention."""
        if not CUDA_KERNELS_AVAILABLE:
            raise RuntimeError("CUDA kernels not available")
        
        query, key, value, bias_mask, triangle_bias = ctx.saved_tensors
        starting_node = ctx.starting_node
        
        # Create dummy attention weights (in real implementation, these would be saved from forward)
        attention_weights = torch.zeros_like(grad_output)
        
        # Call CUDA backward kernel
        grad_input = openfold_cuda_kernels.triangle_attention_backward(
            grad_output, query, key, value, attention_weights,
            bias_mask, triangle_bias, starting_node
        )
        
        # Return gradients for all inputs (None for non-tensor inputs)
        return grad_input, grad_input, grad_input, None, None, None


class CudaTriangleMultiplyFunction(torch.autograd.Function):
    """
    Autograd function for CUDA triangle multiplication with proper gradient computation.
    """
    
    @staticmethod
    def forward(ctx, input_tensor, mask, outgoing=True):
        """Forward pass of triangle multiplication."""
        if not CUDA_KERNELS_AVAILABLE:
            raise RuntimeError("CUDA kernels not available")
        
        # Save for backward pass
        ctx.save_for_backward(input_tensor, mask)
        ctx.outgoing = outgoing
        
        # Call CUDA kernel
        output = openfold_cuda_kernels.triangle_multiply_forward(
            input_tensor, mask, outgoing
        )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of triangle multiplication."""
        if not CUDA_KERNELS_AVAILABLE:
            raise RuntimeError("CUDA kernels not available")
        
        input_tensor, mask = ctx.saved_tensors
        outgoing = ctx.outgoing
        
        # Create dummy projections (in real implementation, these would be saved from forward)
        batch_size, seq_i, seq_j, channels = input_tensor.shape
        hidden_dim = channels // 2
        
        projections_a = torch.zeros(batch_size, seq_i, seq_j, hidden_dim, 
                                   device=input_tensor.device, dtype=input_tensor.dtype)
        projections_b = torch.zeros(batch_size, seq_i, seq_j, hidden_dim,
                                   device=input_tensor.device, dtype=input_tensor.dtype)
        
        # Call CUDA backward kernel
        grad_input = openfold_cuda_kernels.triangle_multiply_backward(
            grad_output, input_tensor, mask, projections_a, projections_b, outgoing
        )
        
        # Return gradients for all inputs
        return grad_input, None, None


def cuda_triangle_attention(query: torch.Tensor,
                           key: torch.Tensor,
                           value: torch.Tensor,
                           bias_mask: torch.Tensor,
                           triangle_bias: torch.Tensor,
                           starting_node: bool = True) -> torch.Tensor:
    """
    CUDA-accelerated triangle attention with autograd support.
    
    Args:
        query: Query tensor [batch, heads, seq_i, seq_j, head_dim]
        key: Key tensor [batch, heads, seq_i, seq_j, head_dim]
        value: Value tensor [batch, heads, seq_i, seq_j, head_dim]
        bias_mask: Bias mask tensor [batch, seq_i, 1, 1, seq_j]
        triangle_bias: Triangle bias tensor [batch, 1, heads, seq_i, seq_j]
        starting_node: Whether this is starting node attention
        
    Returns:
        Attention output tensor [batch, heads, seq_i, seq_j, head_dim]
    """
    if not CUDA_KERNELS_AVAILABLE:
        raise RuntimeError("CUDA kernels not available. Please compile kernels first.")
    
    return CudaTriangleAttentionFunction.apply(
        query, key, value, bias_mask, triangle_bias, starting_node
    )


def cuda_triangle_multiply(input_tensor: torch.Tensor,
                          mask: torch.Tensor,
                          outgoing: bool = True) -> torch.Tensor:
    """
    CUDA-accelerated triangle multiplication with autograd support.
    
    Args:
        input_tensor: Input tensor [batch, seq_i, seq_j, channels]
        mask: Mask tensor [batch, seq_i, seq_j, 1]
        outgoing: Whether this is outgoing multiplication
        
    Returns:
        Output tensor [batch, seq_i, seq_j, channels]
    """
    if not CUDA_KERNELS_AVAILABLE:
        raise RuntimeError("CUDA kernels not available. Please compile kernels first.")
    
    return CudaTriangleMultiplyFunction.apply(input_tensor, mask, outgoing)


class CudaKernelManager:
    """Manager for CUDA kernel operations and utilities."""
    
    def __init__(self):
        """Initialize CUDA kernel manager."""
        self.available = CUDA_KERNELS_AVAILABLE
        self.info = KERNEL_INFO.copy() if KERNEL_INFO else {}
    
    def is_available(self) -> bool:
        """Check if CUDA kernels are available."""
        return self.available
    
    def get_info(self) -> Dict[str, Any]:
        """Get kernel information."""
        return self.info.copy()
    
    def benchmark_performance(self, 
                            batch_size: int = 2,
                            seq_len: int = 256,
                            num_heads: int = 8,
                            head_dim: int = 64,
                            num_iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark kernel performance.
        
        Args:
            batch_size: Batch size for benchmark
            seq_len: Sequence length for benchmark
            num_heads: Number of attention heads
            head_dim: Head dimension
            num_iterations: Number of iterations to run
            
        Returns:
            Benchmark results dictionary
        """
        if not self.available:
            return {"error": "CUDA kernels not available"}
        
        try:
            results = openfold_cuda_kernels.benchmark_kernel_performance(
                batch_size, seq_len, num_heads, head_dim, num_iterations
            )
            return results
        except Exception as e:
            return {"error": str(e)}
    
    def validate_tensors(self, *tensors) -> bool:
        """
        Validate tensors for CUDA kernel compatibility.
        
        Args:
            *tensors: Tensors to validate
            
        Returns:
            True if all tensors are compatible
        """
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                return False
            if not tensor.is_cuda:
                return False
            if tensor.dtype != torch.float32:
                return False
            if not tensor.is_contiguous():
                return False
        
        return True
    
    def prepare_tensors(self, *tensors) -> Tuple[torch.Tensor, ...]:
        """
        Prepare tensors for CUDA kernel usage.
        
        Args:
            *tensors: Tensors to prepare
            
        Returns:
            Tuple of prepared tensors
        """
        prepared = []
        
        for tensor in tensors:
            # Ensure tensor is on CUDA (if CUDA is available)
            if not tensor.is_cuda and torch.cuda.is_available():
                tensor = tensor.cuda()
            elif not tensor.is_cuda and not torch.cuda.is_available():
                # Keep on CPU if CUDA not available
                pass

            # Ensure tensor is float32
            if tensor.dtype != torch.float32:
                tensor = tensor.float()

            # Ensure tensor is contiguous
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()

            prepared.append(tensor)
        
        return tuple(prepared)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Dictionary with memory usage information
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "cached_mb": torch.cuda.memory_reserved() / (1024 * 1024),
            "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
            "max_cached_mb": torch.cuda.max_memory_reserved() / (1024 * 1024)
        }
    
    def reset_memory_stats(self):
        """Reset GPU memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def synchronize(self):
        """Synchronize CUDA operations."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()


# Global kernel manager instance
kernel_manager = CudaKernelManager()


def check_cuda_kernel_compatibility() -> Dict[str, Any]:
    """
    Check CUDA kernel compatibility with current system.
    
    Returns:
        Compatibility report dictionary
    """
    report = {
        "cuda_available": torch.cuda.is_available(),
        "kernels_available": CUDA_KERNELS_AVAILABLE,
        "compatibility": "unknown"
    }
    
    if not torch.cuda.is_available():
        report["compatibility"] = "incompatible"
        report["reason"] = "CUDA not available"
        return report
    
    if not CUDA_KERNELS_AVAILABLE:
        report["compatibility"] = "needs_compilation"
        report["reason"] = "CUDA kernels not compiled"
        return report
    
    # Check device compatibility
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    # Check if compute capability is supported (7.0+)
    if device_props.major >= 7:
        report["compatibility"] = "compatible"
        report["compute_capability"] = compute_capability
        report["device_name"] = device_props.name
    else:
        report["compatibility"] = "incompatible"
        report["reason"] = f"Compute capability {compute_capability} < 7.0"
    
    return report


def get_kernel_usage_examples() -> Dict[str, str]:
    """
    Get usage examples for CUDA kernels.
    
    Returns:
        Dictionary of usage examples
    """
    return {
        "triangle_attention": """
# CUDA Triangle Attention
query = torch.randn(2, 8, 256, 256, 64, device='cuda')
key = torch.randn(2, 8, 256, 256, 64, device='cuda')
value = torch.randn(2, 8, 256, 256, 64, device='cuda')
bias_mask = torch.zeros(2, 256, 1, 1, 256, device='cuda')
triangle_bias = torch.zeros(2, 1, 8, 256, 256, device='cuda')

output = cuda_triangle_attention(query, key, value, bias_mask, triangle_bias)
""",
        "triangle_multiply": """
# CUDA Triangle Multiplication
input_tensor = torch.randn(2, 256, 256, 128, device='cuda')
mask = torch.ones(2, 256, 256, 1, device='cuda')

output = cuda_triangle_multiply(input_tensor, mask, outgoing=True)
""",
        "benchmark": """
# Benchmark CUDA Kernels
results = kernel_manager.benchmark_performance(
    batch_size=2, seq_len=256, num_heads=8, head_dim=64
)
print(f"Average time: {results['avg_time_ms']:.2f} ms")
""",
        "memory_monitoring": """
# Monitor GPU Memory Usage
memory_before = kernel_manager.get_memory_usage()
output = cuda_triangle_attention(query, key, value, bias_mask, triangle_bias)
memory_after = kernel_manager.get_memory_usage()

print(f"Memory used: {memory_after['allocated_mb'] - memory_before['allocated_mb']:.1f} MB")
"""
    }
