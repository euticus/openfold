"""
OdinFold Multimer Module

Multi-chain protein complex folding capabilities for OdinFold.
Enables folding of protein complexes like AlphaFold-Multimer.
"""

from .attention_utils import MultimerAttentionMask, InterChainAttention
from .chain_processing import ChainBreakProcessor, MultimerPositionalEncoding
from .multimer_runner import MultimerFoldRunner

__all__ = [
    'MultimerAttentionMask',
    'InterChainAttention', 
    'ChainBreakProcessor',
    'MultimerPositionalEncoding',
    'MultimerFoldRunner'
]
