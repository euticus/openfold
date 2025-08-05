"""
Fused Triangle Multiplication for OdinFold

Optimized triangle multiplication with fused kernels, linear approximations,
and memory-efficient implementations for improved performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import logging

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None
    logging.warning("Triton not available. Using PyTorch implementation.")

logger = logging.getLogger(__name__)


if TRITON_AVAILABLE:
    @triton.jit
    def triangle_multiply_kernel(
        # Input pointers
        input_ptr, mask_ptr, output_ptr,
        # Projection pointers
        a_proj_ptr, b_proj_ptr,
        # Tensor dimensions
        batch_size, seq_len_i, seq_len_j, channels, hidden_dim,
        # Strides
        input_batch_stride, input_i_stride, input_j_stride, input_c_stride,
        mask_batch_stride, mask_i_stride, mask_j_stride,
        proj_batch_stride, proj_i_stride, proj_j_stride, proj_c_stride,
        # Configuration
        outgoing: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for triangle multiplication."""

        # Get program IDs
        batch_id = tl.program_id(0)
        i_id = tl.program_id(1)
        j_id = tl.program_id(2)

        # Bounds check
        if batch_id >= batch_size or i_id >= seq_len_i or j_id >= seq_len_j:
            return

        # Calculate base offsets
        input_base = batch_id * input_batch_stride + i_id * input_i_stride + j_id * input_j_stride
        mask_base = batch_id * mask_batch_stride + i_id * mask_i_stride + j_id * mask_j_stride
        proj_base = batch_id * proj_batch_stride

        # Load mask value
        mask_val = tl.load(mask_ptr + mask_base)

        # Process channels in blocks
        for c_start in range(0, channels, BLOCK_SIZE):
            c_end = min(c_start + BLOCK_SIZE, channels)
            c_range = tl.arange(0, BLOCK_SIZE)
            c_mask = c_range < (c_end - c_start)

            # Initialize accumulator
            result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

            # Triangle multiplication loop
            for k in range(seq_len_j):
                if outgoing:
                    # Outgoing: a_ik * b_kj
                    a_offset = proj_base + i_id * proj_i_stride + k * proj_j_stride
                    b_offset = proj_base + k * proj_i_stride + j_id * proj_j_stride
                else:
                    # Incoming: a_ki * b_kj
                    a_offset = proj_base + k * proj_i_stride + i_id * proj_j_stride
                    b_offset = proj_base + k * proj_i_stride + j_id * proj_j_stride

                # Load projections
                a_vals = tl.load(a_proj_ptr + a_offset + (c_start + c_range) % hidden_dim, mask=c_mask)
                b_vals = tl.load(b_proj_ptr + b_offset + (c_start + c_range) % hidden_dim, mask=c_mask)

                # Load mask for k position
                if outgoing:
                    k_mask_offset = batch_id * mask_batch_stride + i_id * mask_i_stride + k * mask_j_stride
                else:
                    k_mask_offset = batch_id * mask_batch_stride + k * mask_i_stride + i_id * mask_j_stride

                k_mask = tl.load(mask_ptr + k_mask_offset)

                # Accumulate
                result += a_vals * b_vals * k_mask

            # Apply final mask and store
            result = result * mask_val
            output_offset = input_base + (c_start + c_range) * input_c_stride
            tl.store(output_ptr + output_offset, result, mask=c_mask)
else:
    # Dummy function when Triton not available
    def triangle_multiply_kernel(*args, **kwargs):
        raise RuntimeError("Triton not available")


class FusedTriangleMultiplication(nn.Module):
    """
    Fused triangle multiplication with multiple optimization strategies.
    
    Combines Triton kernels, linear approximations, and memory-efficient
    implementations for optimal performance across different scenarios.
    """
    
    def __init__(self, 
                 c_in: int,
                 c_hidden: int,
                 outgoing: bool = True,
                 use_triton: bool = True,
                 use_linear_approx: bool = False,
                 chunk_size: Optional[int] = None):
        """
        Args:
            c_in: Input channel dimension
            c_hidden: Hidden channel dimension  
            outgoing: Whether this is outgoing multiplication
            use_triton: Whether to use Triton kernels
            use_linear_approx: Whether to use linear approximation
            chunk_size: Optional chunk size for memory efficiency
        """
        super().__init__()
        
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.outgoing = outgoing
        self.use_triton = use_triton and TRITON_AVAILABLE
        self.use_linear_approx = use_linear_approx
        self.chunk_size = chunk_size
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(c_in)
        
        if use_linear_approx:
            # Linear approximation: single linear layer with masking
            self.linear_approx = nn.Linear(c_in, c_in)
            self.mask_proj = nn.Linear(c_in, c_in)
        else:
            # Standard triangle multiplication projections
            self.linear_a_p = nn.Linear(c_in, c_hidden)
            self.linear_a_g = nn.Linear(c_in, c_hidden, bias=False)
            self.linear_b_p = nn.Linear(c_in, c_hidden)
            self.linear_b_g = nn.Linear(c_in, c_hidden, bias=False)
            self.linear_z = nn.Linear(c_hidden, c_in)
        
        # Gating activation
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        
        if self.use_linear_approx:
            # Initialize linear approximation
            nn.init.normal_(self.linear_approx.weight, std=0.02)
            nn.init.zeros_(self.linear_approx.bias)
            nn.init.normal_(self.mask_proj.weight, std=0.02)
            nn.init.zeros_(self.mask_proj.bias)
        else:
            # Initialize standard projections
            for proj in [self.linear_a_p, self.linear_b_p]:
                nn.init.normal_(proj.weight, std=0.02)
                nn.init.zeros_(proj.bias)
            
            # Gating projections (smaller initialization)
            for proj in [self.linear_a_g, self.linear_b_g]:
                nn.init.normal_(proj.weight, std=0.01)
            
            # Output projection
            nn.init.normal_(self.linear_z.weight, std=0.02 / math.sqrt(self.c_hidden))
            nn.init.zeros_(self.linear_z.bias)
    
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of fused triangle multiplication.
        
        Args:
            x: Input tensor [*, I, J, C_in]
            mask: Optional mask tensor [*, I, J]
            
        Returns:
            Output tensor [*, I, J, C_in]
        """
        
        if mask is None:
            mask = torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype)
        
        # Layer normalization
        x_norm = self.layer_norm(x)
        
        if self.use_linear_approx:
            # Linear approximation path
            output = self._linear_approximation(x_norm, mask)
        elif self.use_triton and x.is_cuda:
            # Triton kernel path
            try:
                output = self._triton_triangle_multiply(x_norm, mask)
            except Exception as e:
                logger.warning(f"Triton kernel failed: {e}. Falling back to PyTorch.")
                output = self._pytorch_triangle_multiply(x_norm, mask)
        else:
            # PyTorch implementation
            output = self._pytorch_triangle_multiply(x_norm, mask)
        
        # Residual connection
        return x + output
    
    def _linear_approximation(self,
                            x: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        """
        Linear approximation of triangle multiplication.
        
        Uses a single linear layer with position-dependent masking
        to approximate the triangle multiplication operation.
        """
        
        # Apply linear transformation
        output = self.linear_approx(x)
        
        # Position-dependent masking
        mask_weights = self.mask_proj(x)
        mask_expanded = mask.unsqueeze(-1)
        
        # Apply masking with learned weights
        output = output * mask_expanded * torch.sigmoid(mask_weights)
        
        return output
    
    def _triton_triangle_multiply(self,
                                x: torch.Tensor,
                                mask: torch.Tensor) -> torch.Tensor:
        """
        Triton-accelerated triangle multiplication.
        
        Args:
            x: Input tensor [*, I, J, C_in]
            mask: Mask tensor [*, I, J]
            
        Returns:
            Output tensor [*, I, J, C_in]
        """
        
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton not available")
        
        batch_dims = x.shape[:-3]
        seq_len_i, seq_len_j, c_in = x.shape[-3:]
        
        # Flatten batch dimensions
        x_flat = x.view(-1, seq_len_i, seq_len_j, c_in)
        mask_flat = mask.view(-1, seq_len_i, seq_len_j)
        batch_size = x_flat.shape[0]
        
        # Compute projections
        a_p = self.linear_a_p(x_flat)
        a_g = self.sigmoid(self.linear_a_g(x_flat))
        b_p = self.linear_b_p(x_flat)
        b_g = self.sigmoid(self.linear_b_g(x_flat))
        
        # Apply gating
        a = a_p * a_g
        b = b_p * b_g
        
        # Prepare output tensor
        output = torch.zeros_like(x_flat)
        
        # Launch Triton kernel
        grid = (batch_size, seq_len_i, seq_len_j)
        triangle_multiply_kernel[grid](
            x_flat, mask_flat, output,
            a, b,
            batch_size, seq_len_i, seq_len_j, c_in, self.c_hidden,
            # Input strides
            seq_len_i * seq_len_j * c_in, seq_len_j * c_in, c_in, 1,
            # Mask strides  
            seq_len_i * seq_len_j, seq_len_j, 1,
            # Projection strides
            seq_len_i * seq_len_j * self.c_hidden, seq_len_j * self.c_hidden, self.c_hidden, 1,
            # Configuration
            outgoing=self.outgoing,
            BLOCK_SIZE=32,
        )
        
        # Apply output projection
        output = self.linear_z(output)
        
        # Restore original shape
        output = output.view(*batch_dims, seq_len_i, seq_len_j, c_in)
        
        return output
    
    def _pytorch_triangle_multiply(self,
                                 x: torch.Tensor,
                                 mask: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of triangle multiplication.
        
        Args:
            x: Input tensor [*, I, J, C_in]
            mask: Mask tensor [*, I, J]
            
        Returns:
            Output tensor [*, I, J, C_in]
        """
        
        if self.chunk_size is not None:
            return self._chunked_triangle_multiply(x, mask)
        
        # Compute projections
        a_p = self.linear_a_p(x)
        a_g = self.sigmoid(self.linear_a_g(x))
        b_p = self.linear_b_p(x)
        b_g = self.sigmoid(self.linear_b_g(x))
        
        # Apply gating
        a = a_p * a_g
        b = b_p * b_g
        
        # Apply mask
        mask_expanded = mask.unsqueeze(-1)
        a = a * mask_expanded
        b = b * mask_expanded
        
        # Triangle multiplication
        if self.outgoing:
            # Outgoing: z_ij = sum_k(a_ik * b_kj)
            output = torch.einsum('...ikc,...kjc->...ijc', a, b)
        else:
            # Incoming: z_ij = sum_k(a_ki * b_kj)
            output = torch.einsum('...kic,...kjc->...ijc', a, b)
        
        # Output projection
        output = self.linear_z(output)
        
        return output
    
    def _chunked_triangle_multiply(self,
                                 x: torch.Tensor,
                                 mask: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient chunked triangle multiplication.
        
        Args:
            x: Input tensor [*, I, J, C_in]
            mask: Mask tensor [*, I, J]
            
        Returns:
            Output tensor [*, I, J, C_in]
        """
        
        batch_dims = x.shape[:-3]
        seq_len_i, seq_len_j, c_in = x.shape[-3:]
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process in chunks along I dimension
        for i_start in range(0, seq_len_i, self.chunk_size):
            i_end = min(i_start + self.chunk_size, seq_len_i)
            
            # Extract chunk
            x_chunk = x[..., i_start:i_end, :, :]
            mask_chunk = mask[..., i_start:i_end, :]
            
            # Compute projections for chunk
            a_p = self.linear_a_p(x_chunk)
            a_g = self.sigmoid(self.linear_a_g(x_chunk))
            b_p = self.linear_b_p(x)  # Full tensor for b
            b_g = self.sigmoid(self.linear_b_g(x))
            
            # Apply gating
            a = a_p * a_g
            b = b_p * b_g
            
            # Apply mask
            mask_chunk_expanded = mask_chunk.unsqueeze(-1)
            mask_expanded = mask.unsqueeze(-1)
            a = a * mask_chunk_expanded
            b = b * mask_expanded
            
            # Triangle multiplication for chunk
            if self.outgoing:
                chunk_output = torch.einsum('...ikc,...kjc->...ijc', a, b)
            else:
                chunk_output = torch.einsum('...kic,...kjc->...ijc', a, b)
            
            # Output projection
            chunk_output = self.linear_z(chunk_output)
            
            # Store result
            output[..., i_start:i_end, :, :] = chunk_output
        
        return output


# Convenience classes for outgoing and incoming multiplication
class FusedTriangleMultiplicationOutgoing(FusedTriangleMultiplication):
    """Fused triangle multiplication for outgoing edges."""
    
    def __init__(self, c_in: int, c_hidden: int, **kwargs):
        super().__init__(c_in, c_hidden, outgoing=True, **kwargs)


class FusedTriangleMultiplicationIncoming(FusedTriangleMultiplication):
    """Fused triangle multiplication for incoming edges."""
    
    def __init__(self, c_in: int, c_hidden: int, **kwargs):
        super().__init__(c_in, c_hidden, outgoing=False, **kwargs)


class LinearTriangleApproximation(nn.Module):
    """
    Linear approximation of triangle multiplication for ultra-fast inference.
    
    Replaces the expensive triangle multiplication with a simple linear
    transformation and learned masking patterns.
    """
    
    def __init__(self, c_in: int, outgoing: bool = True):
        super().__init__()
        
        self.c_in = c_in
        self.outgoing = outgoing
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(c_in)
        
        # Linear approximation layers
        self.linear_main = nn.Linear(c_in, c_in)
        self.linear_gate = nn.Linear(c_in, c_in)
        self.position_bias = nn.Parameter(torch.zeros(1, 1, 1, c_in))
        
        # Initialize weights
        nn.init.normal_(self.linear_main.weight, std=0.02)
        nn.init.zeros_(self.linear_main.bias)
        nn.init.normal_(self.linear_gate.weight, std=0.01)
        nn.init.zeros_(self.linear_gate.bias)
    
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of linear triangle approximation.
        
        Args:
            x: Input tensor [*, I, J, C_in]
            mask: Optional mask tensor [*, I, J]
            
        Returns:
            Output tensor [*, I, J, C_in]
        """
        
        if mask is None:
            mask = torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype)
        
        # Layer normalization
        x_norm = self.layer_norm(x)
        
        # Linear transformation
        main_output = self.linear_main(x_norm)
        gate_output = torch.sigmoid(self.linear_gate(x_norm))
        
        # Apply gating and position bias
        output = main_output * gate_output + self.position_bias
        
        # Apply mask
        mask_expanded = mask.unsqueeze(-1)
        output = output * mask_expanded
        
        # Residual connection
        return x + output
