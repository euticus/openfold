#!/usr/bin/env python3
"""
Slim EvoFormer Implementation for OpenFold++

This module implements the optimized EvoFormer architecture for Phase B:
- Reduced layer count (48 → 24)
- Grouped-Query Attention (GQA)
- SwiGLU MLP replacement
- Weight sharing across layers
- FlashAttention-2 integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass

try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    logging.warning("FlashAttention not available. Install with: pip install flash-attn")


@dataclass
class SlimEvoFormerConfig:
    """Configuration for Slim EvoFormer."""
    c_m: int = 256  # MSA channel dimension
    c_z: int = 128  # Pair channel dimension
    c_s: int = 384  # Single representation dimension
    no_blocks: int = 24  # Reduced from 48
    no_heads_msa: int = 8
    no_heads_pair: int = 4
    c_hidden_msa_att: int = 32
    c_hidden_pair_att: int = 32
    c_hidden_opm: int = 32
    c_hidden_mul: int = 128
    transition_n: int = 4
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25
    
    # Phase B optimizations
    use_gqa: bool = True
    gqa_groups: int = 4  # k=4 for KV sharing
    use_swiglu: bool = True
    swiglu_hidden_ratio: float = 2.0  # 2x instead of 4x
    use_weight_sharing: bool = True
    weight_sharing_interval: int = 4  # Share every 4 layers
    use_flash_attention: bool = True
    
    # Other settings
    blocks_per_ckpt: int = 4
    clear_cache_between_blocks: bool = True
    inf: float = 1e9
    eps: float = 1e-10


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) for memory efficiency.
    
    Reduces memory by sharing key-value pairs across query groups.
    """
    
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gqa_groups: int = 4,
        gating: bool = True,
        use_flash: bool = True
    ):
        super().__init__()
        
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gqa_groups = gqa_groups
        self.gating = gating
        self.use_flash = use_flash and FLASH_ATTENTION_AVAILABLE
        
        # Ensure heads are divisible by groups
        assert no_heads % gqa_groups == 0, f"no_heads ({no_heads}) must be divisible by gqa_groups ({gqa_groups})"
        
        self.heads_per_group = no_heads // gqa_groups
        
        # Linear projections
        self.linear_q = nn.Linear(c_q, no_heads * c_hidden, bias=False)
        self.linear_k = nn.Linear(c_k, gqa_groups * c_hidden, bias=False)  # Reduced KV size
        self.linear_v = nn.Linear(c_v, gqa_groups * c_hidden, bias=False)  # Reduced KV size
        self.linear_o = nn.Linear(no_heads * c_hidden, c_q)
        
        if gating:
            self.linear_g = nn.Linear(c_q, no_heads * c_hidden)
        
        self.rescale_factor = 1.0 / math.sqrt(c_hidden)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with grouped-query attention.
        
        Args:
            q: Query tensor [batch, seq_len_q, c_q]
            k: Key tensor [batch, seq_len_k, c_k]
            v: Value tensor [batch, seq_len_v, c_v]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len_q, c_q]
        """
        batch_size, seq_len_q, _ = q.shape
        seq_len_k = k.shape[1]

        # Store original q for gating
        q_orig = q

        # Project to query, key, value
        q = self.linear_q(q)  # [batch, seq_len_q, no_heads * c_hidden]
        k = self.linear_k(k)  # [batch, seq_len_k, gqa_groups * c_hidden]
        v = self.linear_v(v)  # [batch, seq_len_v, gqa_groups * c_hidden]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len_q, self.no_heads, self.c_hidden)
        k = k.view(batch_size, seq_len_k, self.gqa_groups, self.c_hidden)
        v = v.view(batch_size, seq_len_k, self.gqa_groups, self.c_hidden)
        
        # Expand k, v to match query heads
        k = k.repeat_interleave(self.heads_per_group, dim=2)  # [batch, seq_len_k, no_heads, c_hidden]
        v = v.repeat_interleave(self.heads_per_group, dim=2)  # [batch, seq_len_v, no_heads, c_hidden]
        
        if self.use_flash and mask is None:
            # Use FlashAttention if available and no custom mask
            q = q.transpose(1, 2)  # [batch, no_heads, seq_len_q, c_hidden]
            k = k.transpose(1, 2)  # [batch, no_heads, seq_len_k, c_hidden]
            v = v.transpose(1, 2)  # [batch, no_heads, seq_len_v, c_hidden]
            
            attn_output = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
            attn_output = attn_output.transpose(1, 2)  # [batch, seq_len_q, no_heads, c_hidden]
        else:
            # Standard attention computation
            q = q.transpose(1, 2)  # [batch, no_heads, seq_len_q, c_hidden]
            k = k.transpose(1, 2)  # [batch, no_heads, seq_len_k, c_hidden]
            v = v.transpose(1, 2)  # [batch, no_heads, seq_len_v, c_hidden]
            
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.rescale_factor
            
            # Apply mask if provided
            if mask is not None:
                scores = scores + mask
            
            # Softmax
            attn_weights = F.softmax(scores, dim=-1)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2)  # [batch, seq_len_q, no_heads, c_hidden]
        
        # Reshape and project output
        attn_output = attn_output.contiguous().view(batch_size, seq_len_q, -1)
        
        # Apply gating if enabled
        if self.gating:
            # Use original q input for gating
            gate = torch.sigmoid(self.linear_g(q_orig))
            attn_output = attn_output * gate
        
        # Final output projection
        output = self.linear_o(attn_output)
        
        return output


class SwiGLU(nn.Module):
    """
    SwiGLU activation function for efficient MLP.
    
    Replaces standard 4x hidden MLP with gated 2x hidden MLP.
    """
    
    def __init__(self, dim: int, hidden_ratio: float = 2.0):
        super().__init__()
        
        hidden_dim = int(dim * hidden_ratio)
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)  # Value projection
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)  # Output projection
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with same shape as input
        """
        gate = F.silu(self.w1(x))  # Swish activation
        value = self.w2(x)
        return self.w3(gate * value)


class SharedEvoFormerBlock(nn.Module):
    """
    Shared EvoFormer block for weight sharing optimization.
    
    This block can be reused across multiple layer positions.
    """
    
    def __init__(self, config: SlimEvoFormerConfig):
        super().__init__()
        self.config = config
        
        # MSA attention with GQA
        self.msa_att = GroupedQueryAttention(
            c_q=config.c_m,
            c_k=config.c_m,
            c_v=config.c_m,
            c_hidden=config.c_hidden_msa_att,
            no_heads=config.no_heads_msa,
            gqa_groups=config.gqa_groups if config.use_gqa else config.no_heads_msa,
            use_flash=config.use_flash_attention
        )
        
        # Pair attention with GQA
        self.pair_att = GroupedQueryAttention(
            c_q=config.c_z,
            c_k=config.c_z,
            c_v=config.c_z,
            c_hidden=config.c_hidden_pair_att,
            no_heads=config.no_heads_pair,
            gqa_groups=config.gqa_groups if config.use_gqa else config.no_heads_pair,
            use_flash=config.use_flash_attention
        )
        
        # MLP layers
        if config.use_swiglu:
            self.msa_transition = SwiGLU(config.c_m, config.swiglu_hidden_ratio)
            self.pair_transition = SwiGLU(config.c_z, config.swiglu_hidden_ratio)
        else:
            # Standard MLP
            hidden_dim_msa = config.c_m * config.transition_n
            hidden_dim_pair = config.c_z * config.transition_n
            
            self.msa_transition = nn.Sequential(
                nn.Linear(config.c_m, hidden_dim_msa),
                nn.ReLU(),
                nn.Linear(hidden_dim_msa, config.c_m)
            )
            
            self.pair_transition = nn.Sequential(
                nn.Linear(config.c_z, hidden_dim_pair),
                nn.ReLU(),
                nn.Linear(hidden_dim_pair, config.c_z)
            )
        
        # Layer normalization
        self.msa_ln = nn.LayerNorm(config.c_m)
        self.pair_ln = nn.LayerNorm(config.c_z)
        
        # Dropout
        self.msa_dropout = nn.Dropout(config.msa_dropout)
        self.pair_dropout = nn.Dropout(config.pair_dropout)
    
    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        msa_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through shared EvoFormer block.
        
        Args:
            msa: MSA representation [batch, n_seq, n_res, c_m]
            pair: Pair representation [batch, n_res, n_res, c_z]
            msa_mask: Optional MSA mask
            pair_mask: Optional pair mask
            
        Returns:
            Updated (msa, pair) representations
        """
        # MSA self-attention
        batch_size, n_seq, n_res, c_m = msa.shape
        msa_flat = msa.view(batch_size * n_seq, n_res, c_m)
        
        msa_att_out = self.msa_att(msa_flat, msa_flat, msa_flat, msa_mask)
        msa_att_out = msa_att_out.view(batch_size, n_seq, n_res, c_m)
        
        # MSA residual and layer norm
        msa = self.msa_ln(msa + self.msa_dropout(msa_att_out))
        
        # MSA transition
        msa_trans_out = self.msa_transition(msa)
        msa = msa + self.msa_dropout(msa_trans_out)
        
        # Pair self-attention
        batch_size, n_res, _, c_z = pair.shape
        pair_flat = pair.view(batch_size * n_res, n_res, c_z)
        
        pair_att_out = self.pair_att(pair_flat, pair_flat, pair_flat, pair_mask)
        pair_att_out = pair_att_out.view(batch_size, n_res, n_res, c_z)
        
        # Pair residual and layer norm
        pair = self.pair_ln(pair + self.pair_dropout(pair_att_out))
        
        # Pair transition
        pair_trans_out = self.pair_transition(pair)
        pair = pair + self.pair_dropout(pair_trans_out)
        
        return msa, pair


class SlimEvoFormerStack(nn.Module):
    """
    Slim EvoFormer stack with all Phase B optimizations.
    
    Features:
    - Reduced layer count (24 blocks)
    - Grouped-Query Attention
    - SwiGLU MLP
    - Weight sharing
    - FlashAttention-2
    """
    
    def __init__(self, config: SlimEvoFormerConfig = None):
        super().__init__()
        
        self.config = config or SlimEvoFormerConfig()
        
        # Create shared blocks for weight sharing
        if self.config.use_weight_sharing:
            num_shared_blocks = (self.config.no_blocks + self.config.weight_sharing_interval - 1) // self.config.weight_sharing_interval
            self.shared_blocks = nn.ModuleList([
                SharedEvoFormerBlock(self.config) for _ in range(num_shared_blocks)
            ])
        else:
            # Create individual blocks
            self.blocks = nn.ModuleList([
                SharedEvoFormerBlock(self.config) for _ in range(self.config.no_blocks)
            ])
        
        # Final linear layer
        self.linear = nn.Linear(self.config.c_m, self.config.c_s)
        
        logging.info(f"SlimEvoFormer initialized: {self.config.no_blocks} blocks, "
                    f"GQA: {self.config.use_gqa}, SwiGLU: {self.config.use_swiglu}, "
                    f"Weight sharing: {self.config.use_weight_sharing}")
    
    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        msa_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through slim EvoFormer stack.
        
        Args:
            msa: MSA representation [batch, n_seq, n_res, c_m]
            pair: Pair representation [batch, n_res, n_res, c_z]
            msa_mask: Optional MSA mask
            pair_mask: Optional pair mask
            
        Returns:
            (msa_output, pair_output, single_output)
        """
        
        for i in range(self.config.no_blocks):
            if self.config.use_weight_sharing:
                # Use shared block
                block_idx = i // self.config.weight_sharing_interval
                block = self.shared_blocks[block_idx]
            else:
                # Use individual block
                block = self.blocks[i]
            
            msa, pair = block(msa, pair, msa_mask, pair_mask)
            
            # Clear cache between blocks if enabled
            if self.config.clear_cache_between_blocks and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Extract single representation (first sequence of MSA)
        single = msa[:, 0, :, :]  # [batch, n_res, c_m]
        single = self.linear(single)  # [batch, n_res, c_s]
        
        return msa, pair, single
    
    def count_parameters(self) -> int:
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Factory function
def create_slim_evoformer(config: SlimEvoFormerConfig = None) -> SlimEvoFormerStack:
    """
    Factory function to create slim EvoFormer stack.
    
    Args:
        config: Optional configuration
        
    Returns:
        SlimEvoFormerStack instance
    """
    return SlimEvoFormerStack(config)


# Example usage and testing
if __name__ == "__main__":
    # Test slim EvoFormer
    config = SlimEvoFormerConfig()
    model = create_slim_evoformer(config)
    
    # Test input
    batch_size = 2
    n_seq = 64
    n_res = 128
    
    msa = torch.randn(batch_size, n_seq, n_res, config.c_m)
    pair = torch.randn(batch_size, n_res, n_res, config.c_z)
    
    # Forward pass
    msa_out, pair_out, single_out = model(msa, pair)
    
    print(f"✅ Slim EvoFormer test successful!")
    print(f"   Input MSA: {msa.shape}")
    print(f"   Input Pair: {pair.shape}")
    print(f"   Output MSA: {msa_out.shape}")
    print(f"   Output Pair: {pair_out.shape}")
    print(f"   Output Single: {single_out.shape}")
    print(f"   Parameters: {model.count_parameters():,}")
    
    # Test memory efficiency
    if torch.cuda.is_available():
        model = model.cuda()
        msa = msa.cuda()
        pair = pair.cuda()
        
        torch.cuda.reset_peak_memory_stats()
        msa_out, pair_out, single_out = model(msa, pair)
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        print(f"   Peak GPU memory: {peak_memory:.1f} MB")
