"""
OdinFold Ligand Module

Ligand-aware attention and binding pocket prediction for OdinFold.
Enables conditioning protein folding on ligand presence and geometry.
"""

from .ligand_encoder import LigandEncoder, LigandAtomEmbedding
from .ligand_cross_attention import LigandCrossAttention, ProteinLigandAttention
from .ligand_utils import parse_ligand_input, calculate_binding_pocket

__all__ = [
    'LigandEncoder',
    'LigandAtomEmbedding',
    'LigandCrossAttention',
    'ProteinLigandAttention',
    'parse_ligand_input',
    'calculate_binding_pocket'
]
