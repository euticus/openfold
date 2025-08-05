#!/usr/bin/env python3
"""
Test Suite for Ligand-Aware Folding

Tests ligand encoding, cross-attention, and ligand-aware structure prediction.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openfoldpp.modules.ligand import (
    LigandEncoder,
    MolecularGraphEncoder,
    LigandFeatureExtractor,
    AtomTypeEmbedding,
    LigandProteinCrossAttention,
    LigandAwareFoldingHead,
    BindingPocketAttention,
    LigandConditionedStructureModule
)
from openfoldpp.modules.ligand.ligand_utils import (
    smiles_to_graph,
    mol_to_graph,
    compute_ligand_protein_distances,
    get_binding_pocket_mask,
    batch_process_ligands,
    validate_ligand_data
)


class TestAtomTypeEmbedding:
    """Test atom type embedding."""
    
    def test_atom_embedding_init(self):
        """Test initialization of AtomTypeEmbedding."""
        
        d_model = 128
        embedding = AtomTypeEmbedding(d_model)
        
        assert embedding.d_model == d_model
        assert hasattr(embedding, 'atom_embedding')
        assert hasattr(embedding, 'property_proj')
        assert hasattr(embedding, 'hybridization_embedding')
        assert hasattr(embedding, 'charge_embedding')
    
    def test_atom_embedding_forward(self):
        """Test forward pass of AtomTypeEmbedding."""
        
        d_model = 64
        num_atoms = 10
        
        embedding = AtomTypeEmbedding(d_model)
        
        atom_types = torch.randint(0, 20, (num_atoms,))
        hybridization = torch.randint(0, 5, (num_atoms,))
        formal_charges = torch.randint(-2, 3, (num_atoms,))
        
        output = embedding(atom_types, hybridization, formal_charges)
        
        assert output.shape == (num_atoms, d_model)
        assert torch.isfinite(output).all()
    
    def test_get_atom_index(self):
        """Test atom index lookup."""

        embedding = AtomTypeEmbedding()

        assert embedding.get_atom_index('C') == 2
        assert embedding.get_atom_index('N') == 3
        assert embedding.get_atom_index('O') == 4
        assert embedding.get_atom_index('X') == 0  # Unknown


class TestLigandFeatureExtractor:
    """Test ligand feature extractor."""
    
    def test_feature_extractor_init(self):
        """Test initialization of LigandFeatureExtractor."""
        
        d_model = 128
        extractor = LigandFeatureExtractor(d_model)
        
        assert extractor.d_model == d_model
        assert hasattr(extractor, 'bond_embedding')
        assert hasattr(extractor, 'ring_embedding')
        assert hasattr(extractor, 'pharmacophore_proj')
        assert hasattr(extractor, 'descriptor_proj')
    
    def test_feature_extractor_forward(self):
        """Test forward pass of LigandFeatureExtractor."""
        
        d_model = 64
        num_atoms = 8
        num_bonds = 12
        
        extractor = LigandFeatureExtractor(d_model)
        
        bond_types = torch.randint(0, 5, (num_bonds,))
        ring_info = torch.randint(0, 3, (num_atoms,))
        pharmacophore_features = torch.randn(num_atoms, 8)
        molecular_descriptors = torch.randn(num_atoms, 6)
        
        output = extractor(bond_types, ring_info, pharmacophore_features, molecular_descriptors)
        
        assert output.shape == (num_atoms, d_model)
        assert torch.isfinite(output).all()


class TestMolecularGraphEncoder:
    """Test molecular graph encoder."""
    
    def test_graph_encoder_init(self):
        """Test initialization of MolecularGraphEncoder."""
        
        d_model, num_layers = 64, 3
        encoder = MolecularGraphEncoder(d_model, num_layers)
        
        assert encoder.d_model == d_model
        assert encoder.num_layers == num_layers
        assert len(encoder.gat_layers) == num_layers
        assert len(encoder.layer_norms) == num_layers
    
    def test_graph_encoder_forward(self):
        """Test forward pass of MolecularGraphEncoder."""
        
        d_model = 64
        num_atoms = 10
        
        encoder = MolecularGraphEncoder(d_model, num_layers=2)
        
        node_features = torch.randn(num_atoms, d_model)
        edge_index = torch.randint(0, num_atoms, (2, num_atoms * 2))
        
        output = encoder(node_features, edge_index)
        
        assert 'node_embeddings' in output
        assert 'graph_embedding' in output
        assert 'num_atoms' in output
        
        assert output['node_embeddings'].shape == (num_atoms, d_model)
        assert output['graph_embedding'].shape == (d_model,)
        assert output['num_atoms'] == num_atoms


class TestLigandEncoder:
    """Test complete ligand encoder."""
    
    def test_ligand_encoder_init(self):
        """Test initialization of LigandEncoder."""
        
        d_model = 128
        encoder = LigandEncoder(d_model)
        
        assert encoder.d_model == d_model
        assert hasattr(encoder, 'atom_embedding')
        assert hasattr(encoder, 'feature_extractor')
        assert hasattr(encoder, 'graph_encoder')
        assert hasattr(encoder, 'fusion_layer')
    
    def test_ligand_encoder_forward(self):
        """Test forward pass of LigandEncoder."""
        
        d_model = 64
        num_atoms = 8
        num_bonds = 12
        
        encoder = LigandEncoder(d_model, num_gnn_layers=2)
        
        ligand_data = {
            'atom_types': torch.randint(0, 20, (num_atoms,)),
            'edge_index': torch.randint(0, num_atoms, (2, num_bonds)),
            'bond_types': torch.randint(0, 5, (num_bonds,)),
            'ring_info': torch.randint(0, 3, (num_atoms,)),
            'pharmacophore_features': torch.randn(num_atoms, 8),
            'molecular_descriptors': torch.randn(num_atoms, 6),
            'hybridization': torch.randint(0, 5, (num_atoms,)),
            'formal_charges': torch.randint(-2, 3, (num_atoms,))
        }
        
        output = encoder(ligand_data)
        
        assert 'atom_embeddings' in output
        assert 'ligand_embedding' in output
        assert 'num_atoms' in output
        
        assert output['atom_embeddings'].shape == (num_atoms, d_model)
        assert output['ligand_embedding'].shape == (d_model,)
        assert output['num_atoms'] == num_atoms


class TestLigandProteinCrossAttention:
    """Test ligand-protein cross-attention."""
    
    def test_cross_attention_init(self):
        """Test initialization of LigandProteinCrossAttention."""
        
        protein_dim, ligand_dim, d_model = 384, 128, 256
        cross_attn = LigandProteinCrossAttention(protein_dim, ligand_dim, d_model)
        
        assert cross_attn.protein_dim == protein_dim
        assert cross_attn.ligand_dim == ligand_dim
        assert cross_attn.d_model == d_model
        assert hasattr(cross_attn, 'protein_proj')
        assert hasattr(cross_attn, 'ligand_proj')
    
    def test_cross_attention_forward(self):
        """Test forward pass of LigandProteinCrossAttention."""
        
        batch_size, seq_len, num_atoms = 2, 50, 15
        protein_dim, ligand_dim, d_model = 192, 64, 128
        
        cross_attn = LigandProteinCrossAttention(protein_dim, ligand_dim, d_model)
        
        protein_features = torch.randn(batch_size, seq_len, protein_dim)
        ligand_features = torch.randn(batch_size, num_atoms, ligand_dim)
        protein_coords = torch.randn(batch_size, seq_len, 3)
        ligand_coords = torch.randn(batch_size, num_atoms, 3)
        
        output = cross_attn(protein_features, ligand_features, protein_coords, ligand_coords)
        
        assert 'protein_features' in output
        assert 'ligand_features' in output
        assert 'cross_attention_weights' in output
        assert 'interaction_mask' in output
        
        assert output['protein_features'].shape == (batch_size, seq_len, protein_dim)
        assert output['ligand_features'].shape == (batch_size, num_atoms, ligand_dim)


class TestBindingPocketAttention:
    """Test binding pocket attention."""
    
    def test_pocket_attention_init(self):
        """Test initialization of BindingPocketAttention."""
        
        d_model = 256
        pocket_attn = BindingPocketAttention(d_model)
        
        assert pocket_attn.d_model == d_model
        assert hasattr(pocket_attn, 'pocket_predictor')
        assert hasattr(pocket_attn, 'pocket_attention')
    
    def test_pocket_attention_forward(self):
        """Test forward pass of BindingPocketAttention."""
        
        batch_size, seq_len, num_atoms = 2, 30, 10
        d_model = 128
        
        pocket_attn = BindingPocketAttention(d_model)
        
        protein_features = torch.randn(batch_size, seq_len, d_model)
        ligand_coords = torch.randn(batch_size, num_atoms, 3)
        protein_coords = torch.randn(batch_size, seq_len, 3)
        
        output = pocket_attn(protein_features, ligand_coords, protein_coords)
        
        assert 'protein_features' in output
        assert 'pocket_scores' in output
        assert 'pocket_mask' in output
        assert 'attention_weights' in output
        
        assert output['protein_features'].shape == (batch_size, seq_len, d_model)
        assert output['pocket_scores'].shape == (batch_size, seq_len)
        assert output['pocket_mask'].shape == (batch_size, seq_len)


class TestLigandAwareFoldingHead:
    """Test ligand-aware folding head."""
    
    def test_folding_head_init(self):
        """Test initialization of LigandAwareFoldingHead."""
        
        protein_dim, ligand_dim, d_model = 384, 128, 256
        folding_head = LigandAwareFoldingHead(protein_dim, ligand_dim, d_model)
        
        assert folding_head.protein_dim == protein_dim
        assert folding_head.ligand_dim == ligand_dim
        assert folding_head.d_model == d_model
        assert hasattr(folding_head, 'cross_attention')
        assert hasattr(folding_head, 'pocket_attention')
        assert hasattr(folding_head, 'structure_head')
        assert hasattr(folding_head, 'confidence_head')
    
    def test_folding_head_forward(self):
        """Test forward pass of LigandAwareFoldingHead."""
        
        batch_size, seq_len, num_atoms = 2, 25, 12
        protein_dim, ligand_dim, d_model = 192, 64, 128
        
        folding_head = LigandAwareFoldingHead(protein_dim, ligand_dim, d_model)
        
        protein_features = torch.randn(batch_size, seq_len, protein_dim)
        ligand_features = torch.randn(batch_size, num_atoms, ligand_dim)
        protein_coords = torch.randn(batch_size, seq_len, 3)
        ligand_coords = torch.randn(batch_size, num_atoms, 3)
        
        output = folding_head(protein_features, ligand_features, protein_coords, ligand_coords)
        
        assert 'coordinates' in output
        assert 'confidence' in output
        assert 'pocket_scores' in output
        assert 'ligand_context' in output
        
        assert output['coordinates'].shape == (batch_size, seq_len, 3)
        assert output['confidence'].shape == (batch_size, seq_len)
        assert output['pocket_scores'].shape == (batch_size, seq_len)
        assert output['ligand_context'].shape == (batch_size, ligand_dim)


class TestLigandUtils:
    """Test ligand utility functions."""
    
    def test_smiles_to_graph(self):
        """Test SMILES to graph conversion."""
        
        smiles = "CCO"  # Ethanol
        graph = smiles_to_graph(smiles)
        
        assert 'atom_types' in graph
        assert 'edge_index' in graph
        assert 'bond_types' in graph
        assert 'coordinates' in graph
        assert 'num_atoms' in graph
        
        assert graph['num_atoms'] > 0
        assert graph['atom_types'].shape[0] == graph['num_atoms']
    
    def test_compute_distances(self):
        """Test distance computation."""
        
        ligand_coords = torch.randn(10, 3)
        protein_coords = torch.randn(20, 3)
        
        distances = compute_ligand_protein_distances(ligand_coords, protein_coords)
        
        assert distances.shape == (20, 10)
        assert (distances >= 0).all()
    
    def test_binding_pocket_mask(self):
        """Test binding pocket mask generation."""
        
        ligand_coords = torch.zeros(5, 3)  # Ligand at origin
        protein_coords = torch.randn(15, 3) * 10  # Spread out protein
        
        pocket_mask = get_binding_pocket_mask(ligand_coords, protein_coords, cutoff=5.0)
        
        assert pocket_mask.shape == (15,)
        assert pocket_mask.dtype == torch.bool
    
    def test_batch_process_ligands(self):
        """Test batch processing of ligands."""
        
        smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]  # Ethanol, acetic acid, benzene
        
        batched_data = batch_process_ligands(smiles_list)
        
        assert 'atom_types' in batched_data
        assert 'ligand_mask' in batched_data
        assert 'num_atoms' in batched_data
        
        batch_size = len(smiles_list)
        assert batched_data['atom_types'].shape[0] == batch_size
        assert batched_data['ligand_mask'].shape[0] == batch_size
        assert batched_data['num_atoms'].shape[0] == batch_size
    
    def test_validate_ligand_data(self):
        """Test ligand data validation."""
        
        # Valid data
        valid_data = {
            'atom_types': torch.randint(0, 20, (10,)),
            'edge_index': torch.randint(0, 10, (2, 15)),
            'bond_types': torch.randint(0, 5, (15,)),
            'ring_info': torch.randint(0, 3, (10,)),
            'pharmacophore_features': torch.randn(10, 8),
            'molecular_descriptors': torch.randn(10, 6),
            'coordinates': torch.randn(10, 3),
            'num_atoms': 10
        }
        
        assert validate_ligand_data(valid_data)
        
        # Invalid data (missing key)
        invalid_data = valid_data.copy()
        del invalid_data['atom_types']
        
        assert not validate_ligand_data(invalid_data)


def test_ligand_aware_folding_integration():
    """Integration test for complete ligand-aware folding pipeline."""
    
    print("🧪 Testing ligand-aware folding integration...")
    
    # Parameters
    batch_size, seq_len, num_atoms = 2, 30, 15
    protein_dim, ligand_dim, d_model = 192, 64, 128
    
    # Create components
    ligand_encoder = LigandEncoder(ligand_dim, num_gnn_layers=2)
    folding_head = LigandAwareFoldingHead(protein_dim, ligand_dim, d_model)
    structure_module = LigandConditionedStructureModule(
        protein_dim, ligand_dim, d_model, num_layers=2
    )
    
    print("✅ Components created successfully")
    
    # Mock ligand data
    ligand_data = {
        'atom_types': torch.randint(0, 20, (num_atoms,)),
        'edge_index': torch.randint(0, num_atoms, (2, num_atoms * 2)),
        'bond_types': torch.randint(0, 5, (num_atoms * 2,)),
        'ring_info': torch.randint(0, 3, (num_atoms,)),
        'pharmacophore_features': torch.randn(num_atoms, 8),
        'molecular_descriptors': torch.randn(num_atoms, 6),
        'hybridization': torch.randint(0, 5, (num_atoms,)),
        'formal_charges': torch.randint(-2, 3, (num_atoms,))
    }
    
    # Encode ligand
    ligand_output = ligand_encoder(ligand_data)
    print(f"✅ Ligand encoded: {ligand_output['num_atoms']} atoms")
    
    # Mock protein data
    protein_features = torch.randn(batch_size, seq_len, protein_dim)
    protein_coords = torch.randn(batch_size, seq_len, 3)
    ligand_coords = torch.randn(batch_size, num_atoms, 3)
    
    # Expand ligand features for batch
    ligand_features = ligand_output['atom_embeddings'].unsqueeze(0).expand(batch_size, -1, -1)
    
    # Test folding head
    folding_output = folding_head(
        protein_features, ligand_features, protein_coords, ligand_coords
    )
    print(f"✅ Folding head output: {folding_output['coordinates'].shape}")
    
    # Test structure module
    structure_output = structure_module(
        protein_features, ligand_features, protein_coords, ligand_coords
    )
    print(f"✅ Structure module output: {structure_output['final_coordinates'].shape}")
    print(f"✅ Refinement steps: {structure_output['num_refinement_steps']}")
    
    # Validate outputs
    assert folding_output['coordinates'].shape == (batch_size, seq_len, 3)
    assert folding_output['confidence'].shape == (batch_size, seq_len)
    assert structure_output['final_coordinates'].shape == (batch_size, seq_len, 3)
    assert structure_output['final_confidence'].shape == (batch_size, seq_len)
    
    print("🎉 Ligand-aware folding integration test passed!")


if __name__ == "__main__":
    # Run integration test
    test_ligand_aware_folding_integration()
    
    # Run all tests
    pytest.main([__file__, "-v"])
