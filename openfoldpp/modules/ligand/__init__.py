"""
Ligand-Aware Folding Module for OdinFold

Enables protein structure prediction with ligand awareness through:
- Molecular graph encoding of ligands
- Cross-attention between protein and ligand features
- Binding pocket-aware structure prediction
- Ligand-conditioned folding heads
"""

from .ligand_encoder import (
    LigandEncoder,
    MolecularGraphEncoder,
    LigandFeatureExtractor,
    AtomTypeEmbedding
)
from .ligand_attention import (
    LigandProteinCrossAttention,
    LigandAwareFoldingHead,
    BindingPocketAttention,
    LigandConditionedStructureModule
)
from .ligand_utils import (
    smiles_to_graph,
    mol_to_graph,
    compute_ligand_protein_distances,
    get_binding_pocket_mask
)

__all__ = [
    'LigandEncoder',
    'MolecularGraphEncoder', 
    'LigandFeatureExtractor',
    'AtomTypeEmbedding',
    'LigandProteinCrossAttention',
    'LigandAwareFoldingHead',
    'BindingPocketAttention',
    'LigandConditionedStructureModule',
    'smiles_to_graph',
    'mol_to_graph',
    'compute_ligand_protein_distances',
    'get_binding_pocket_mask'
]
