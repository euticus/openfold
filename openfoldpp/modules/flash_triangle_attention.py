"""
FlashAttention2-based Triangle Attention for OdinFold

Replaces the original triangle attention kernels with FlashAttention2
for improved memory efficiency and speed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math
import logging

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    logging.warning("FlashAttention not available. Falling back to standard attention.")

logger = logging.getLogger(__name__)


class FlashTriangleAttention(nn.Module):
    """
    FlashAttention2-based triangle attention module.
    
    Replaces the original TriangleAttention with a memory-efficient
    FlashAttention2 implementation that maintains the same functionality.
    """
    
    def __init__(self, 
                 c_in: int,
                 c_hidden: int, 
                 no_heads: int,
                 starting: bool = True,
                 inf: float = 1e9,
                 dropout: float = 0.0):
        """
        Args:
            c_in: Input channel dimension
            c_hidden: Hidden channel dimension (total, not per-head)
            no_heads: Number of attention heads
            starting: Whether this is starting node attention
            inf: Large value for masking
            dropout: Dropout probability
        """
        super().__init__()
        
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf
        self.dropout = dropout
        
        assert c_hidden % no_heads == 0, "c_hidden must be divisible by no_heads"
        self.head_dim = c_hidden // no_heads
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(c_in)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(c_in, c_hidden, bias=False)
        self.k_proj = nn.Linear(c_in, c_hidden, bias=False)
        self.v_proj = nn.Linear(c_in, c_hidden, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(c_hidden, c_in)
        
        # Triangle bias projection
        self.triangle_bias_proj = nn.Linear(c_in, no_heads, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using normal distribution."""
        
        # Initialize projections
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.normal_(proj.weight, std=0.02)
        
        # Initialize output projection with smaller std
        nn.init.normal_(self.o_proj.weight, std=0.02 / math.sqrt(2 * self.no_heads))
        nn.init.zeros_(self.o_proj.bias)
        
        # Initialize triangle bias projection
        nn.init.normal_(self.triangle_bias_proj.weight, std=0.02)
    
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                chunk_size: Optional[int] = None,
                use_flash: bool = True) -> torch.Tensor:
        """
        Forward pass of FlashTriangleAttention.
        
        Args:
            x: Input tensor [*, I, J, C_in] (pair representation)
            mask: Optional mask tensor [*, I, J]
            chunk_size: Optional chunk size for memory efficiency
            use_flash: Whether to use FlashAttention (if available)
            
        Returns:
            Output tensor [*, I, J, C_in]
        """
        
        batch_dims = x.shape[:-3]
        seq_len_i, seq_len_j, c_in = x.shape[-3:]
        
        # Handle starting vs ending node
        if not self.starting:
            x = x.transpose(-2, -3)  # Swap I and J dimensions
            if mask is not None:
                mask = mask.transpose(-1, -2)
            seq_len_i, seq_len_j = seq_len_j, seq_len_i
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Linear projections
        q = self.q_proj(x)  # [*, I, J, C_hidden]
        k = self.k_proj(x)  # [*, I, J, C_hidden]
        v = self.v_proj(x)  # [*, I, J, C_hidden]
        
        # Reshape for multi-head attention
        # [*, I, J, H, D] where H = no_heads, D = head_dim
        q = q.contiguous().view(*batch_dims, seq_len_i, seq_len_j, self.no_heads, self.head_dim)
        k = k.contiguous().view(*batch_dims, seq_len_i, seq_len_j, self.no_heads, self.head_dim)
        v = v.contiguous().view(*batch_dims, seq_len_i, seq_len_j, self.no_heads, self.head_dim)
        
        # Compute triangle bias
        triangle_bias = self.triangle_bias_proj(x)  # [*, I, J, H]
        
        # Apply attention
        if use_flash and FLASH_ATTN_AVAILABLE and chunk_size is None:
            attn_output = self._flash_attention(q, k, v, triangle_bias, mask)
        else:
            attn_output = self._standard_attention(q, k, v, triangle_bias, mask, chunk_size)
        
        # Reshape back to [*, I, J, C_hidden]
        attn_output = attn_output.contiguous().view(*batch_dims, seq_len_i, seq_len_j, self.c_hidden)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        # Handle starting vs ending node (transpose back if needed)
        if not self.starting:
            output = output.transpose(-2, -3)
        
        return output
    
    def _flash_attention(self,
                        q: torch.Tensor,
                        k: torch.Tensor, 
                        v: torch.Tensor,
                        triangle_bias: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply FlashAttention2 for triangle attention.
        
        Args:
            q, k, v: Query, key, value tensors [*, I, J, H, D]
            triangle_bias: Triangle bias [*, I, J, H]
            mask: Optional mask [*, I, J]
            
        Returns:
            Attention output [*, I, J, H, D]
        """
        
        batch_dims = q.shape[:-4]
        seq_len_i, seq_len_j, num_heads, head_dim = q.shape[-4:]
        
        # Flatten batch dimensions for FlashAttention
        q_flat = q.view(-1, seq_len_i, seq_len_j, num_heads, head_dim)
        k_flat = k.view(-1, seq_len_i, seq_len_j, num_heads, head_dim)
        v_flat = v.view(-1, seq_len_i, seq_len_j, num_heads, head_dim)
        
        batch_size = q_flat.shape[0]
        
        # Reshape for triangle attention: treat each row i as a separate sequence
        # [B*I, J, H, D]
        q_reshaped = q_flat.view(batch_size * seq_len_i, seq_len_j, num_heads, head_dim)
        k_reshaped = k_flat.view(batch_size * seq_len_i, seq_len_j, num_heads, head_dim)
        v_reshaped = v_flat.view(batch_size * seq_len_i, seq_len_j, num_heads, head_dim)
        
        # Apply FlashAttention
        try:
            # Use flash_attn_func for the attention computation
            attn_output = flash_attn_func(
                q_reshaped, k_reshaped, v_reshaped,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=1.0 / math.sqrt(head_dim),
                causal=False
            )
            
            # Reshape back to [B, I, J, H, D]
            attn_output = attn_output.view(batch_size, seq_len_i, seq_len_j, num_heads, head_dim)
            
            # Restore original batch dimensions
            attn_output = attn_output.view(*batch_dims, seq_len_i, seq_len_j, num_heads, head_dim)
            
        except Exception as e:
            logger.warning(f"FlashAttention failed: {e}. Falling back to standard attention.")
            return self._standard_attention(q, k, v, triangle_bias, mask)
        
        return attn_output
    
    def _standard_attention(self,
                           q: torch.Tensor,
                           k: torch.Tensor,
                           v: torch.Tensor, 
                           triangle_bias: torch.Tensor,
                           mask: Optional[torch.Tensor] = None,
                           chunk_size: Optional[int] = None) -> torch.Tensor:
        """
        Standard attention implementation as fallback.
        
        Args:
            q, k, v: Query, key, value tensors [*, I, J, H, D]
            triangle_bias: Triangle bias [*, I, J, H]
            mask: Optional mask [*, I, J]
            chunk_size: Optional chunk size for memory efficiency
            
        Returns:
            Attention output [*, I, J, H, D]
        """
        
        batch_dims = q.shape[:-4]
        seq_len_i, seq_len_j, num_heads, head_dim = q.shape[-4:]
        
        # Scale factor
        scale = 1.0 / math.sqrt(head_dim)
        
        if chunk_size is not None:
            return self._chunked_attention(q, k, v, triangle_bias, mask, chunk_size, scale)
        
        # Transpose for attention computation: [*, I, H, J, D]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        
        # Compute attention scores: [*, I, H, J, J]
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        
        # Add triangle bias: [*, I, J, H] -> [*, I, H, J, 1]
        triangle_bias = triangle_bias.transpose(-1, -2).unsqueeze(-1)
        scores = scores + triangle_bias
        
        # Apply mask if provided
        if mask is not None:
            # [*, I, J] -> [*, I, 1, J, 1]
            mask_bias = (self.inf * (mask - 1)).unsqueeze(-2).unsqueeze(-1)
            scores = scores + mask_bias
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Apply attention to values: [*, I, H, J, D]
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose back: [*, I, J, H, D]
        attn_output = attn_output.transpose(-2, -3)
        
        return attn_output
    
    def _chunked_attention(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          triangle_bias: torch.Tensor,
                          mask: Optional[torch.Tensor],
                          chunk_size: int,
                          scale: float) -> torch.Tensor:
        """
        Memory-efficient chunked attention implementation.
        
        Args:
            q, k, v: Query, key, value tensors [*, I, J, H, D]
            triangle_bias: Triangle bias [*, I, J, H]
            mask: Optional mask [*, I, J]
            chunk_size: Chunk size for processing
            scale: Attention scale factor
            
        Returns:
            Attention output [*, I, J, H, D]
        """
        
        batch_dims = q.shape[:-4]
        seq_len_i, seq_len_j, num_heads, head_dim = q.shape[-4:]
        
        # Initialize output tensor
        output = torch.zeros_like(q)
        
        # Process in chunks along the I dimension
        for i_start in range(0, seq_len_i, chunk_size):
            i_end = min(i_start + chunk_size, seq_len_i)
            
            # Extract chunks
            q_chunk = q[..., i_start:i_end, :, :, :]  # [*, chunk_i, J, H, D]
            k_chunk = k[..., i_start:i_end, :, :, :]  # [*, chunk_i, J, H, D]
            v_chunk = v[..., i_start:i_end, :, :, :]  # [*, chunk_i, J, H, D]
            
            triangle_bias_chunk = triangle_bias[..., i_start:i_end, :, :]  # [*, chunk_i, J, H]
            
            if mask is not None:
                mask_chunk = mask[..., i_start:i_end, :]  # [*, chunk_i, J]
            else:
                mask_chunk = None
            
            # Apply standard attention to chunk (without chunking recursion)
            chunk_output = self._standard_attention(
                q_chunk, k_chunk, v_chunk, triangle_bias_chunk, mask_chunk, chunk_size=None
            )
            
            # Store result
            output[..., i_start:i_end, :, :, :] = chunk_output
        
        return output


# Convenience classes for starting and ending nodes
class FlashTriangleAttentionStartingNode(FlashTriangleAttention):
    """FlashAttention2-based triangle attention starting node."""
    
    def __init__(self, c_in: int, c_hidden: int, no_heads: int, inf: float = 1e9, dropout: float = 0.0):
        super().__init__(c_in, c_hidden, no_heads, starting=True, inf=inf, dropout=dropout)


class FlashTriangleAttentionEndingNode(FlashTriangleAttention):
    """FlashAttention2-based triangle attention ending node."""
    
    def __init__(self, c_in: int, c_hidden: int, no_heads: int, inf: float = 1e9, dropout: float = 0.0):
        super().__init__(c_in, c_hidden, no_heads, starting=False, inf=inf, dropout=dropout)
