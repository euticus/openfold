#!/usr/bin/env python3
"""
Test Suite for OdinFold Ligand Functionality

Tests ligand-aware attention and binding pocket prediction.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openfoldpp.ligand import (
    LigandEncoder,
    LigandAtomEmbedding,
    LigandCrossAttention,
    ProteinLigandAttention,
    parse_ligand_input,
    calculate_binding_pocket
)
from openfoldpp.ligand.ligand_encoder import create_mock_ligand
from openfoldpp.ligand.ligand_cross_attention import BindingPocketPredictor


class TestLigandAtomEmbedding:
    """Test ligand atom embedding functionality."""
    
    def test_atom_embedding_forward(self):
        """Test forward pass of atom embedding."""

        d_model = 128
        embedding = LigandAtomEmbedding(d_model)

        num_atoms = 15
        # Use mock ligand data with proper ranges
        ligand_data = create_mock_ligand(num_atoms, d_model)
        atom_features = ligand_data['atom_features']
        positions = ligand_data['positions']

        embeddings = embedding(atom_features, positions)
        
        assert embeddings.shape == (num_atoms, d_model)
        assert torch.isfinite(embeddings).all()
    
    def test_different_atom_counts(self):
        """Test embedding with different numbers of atoms."""

        d_model = 64
        embedding = LigandAtomEmbedding(d_model)

        for num_atoms in [5, 20, 50]:
            # Use mock ligand data with proper ranges
            ligand_data = create_mock_ligand(num_atoms, d_model)
            atom_features = ligand_data['atom_features']
            positions = ligand_data['positions']

            embeddings = embedding(atom_features, positions)
            assert embeddings.shape == (num_atoms, d_model)


class TestLigandEncoder:
    """Test complete ligand encoder."""
    
    def test_ligand_encoder_forward(self):
        """Test forward pass of ligand encoder."""
        
        d_model = 128
        encoder = LigandEncoder(d_model, max_atoms=50)
        
        # Create mock ligand data
        ligand_data = create_mock_ligand(num_atoms=20, d_model=d_model)
        
        results = encoder(ligand_data)
        
        # Check output structure
        assert 'atom_embeddings' in results
        assert 'ligand_embedding' in results
        assert 'predicted_properties' in results
        
        # Check shapes
        assert results['atom_embeddings'].shape == (50, d_model)  # Padded to max_atoms
        assert results['ligand_embedding'].shape == (d_model,)
        
        # Check properties
        properties = results['predicted_properties']
        assert 'molecular_weight' in properties
        assert 'logp' in properties
        assert 'tpsa' in properties
        assert 'num_rotatable_bonds' in properties
    
    def test_ligand_padding_truncation(self):
        """Test padding and truncation of ligands."""
        
        d_model = 64
        max_atoms = 30
        encoder = LigandEncoder(d_model, max_atoms=max_atoms)
        
        # Test small ligand (should be padded)
        small_ligand = create_mock_ligand(num_atoms=10, d_model=d_model)
        results_small = encoder(small_ligand)
        assert results_small['atom_embeddings'].shape[0] == max_atoms
        
        # Test large ligand (should be truncated)
        large_ligand = create_mock_ligand(num_atoms=50, d_model=d_model)
        results_large = encoder(large_ligand)
        assert results_large['atom_embeddings'].shape[0] == max_atoms
    
    def test_smiles_encoding(self):
        """Test SMILES string encoding."""
        
        d_model = 64
        encoder = LigandEncoder(d_model)
        
        # Test with simple SMILES
        smiles = "CCO"  # Ethanol
        ligand_data = encoder.encode_smiles(smiles)
        
        # Should return mock data when RDKit not available
        assert ligand_data is not None
        assert 'atom_features' in ligand_data
        assert 'positions' in ligand_data
        assert ligand_data['smiles'] == smiles


class TestLigandCrossAttention:
    """Test ligand cross-attention mechanism."""
    
    def test_cross_attention_forward(self):
        """Test forward pass of cross-attention."""
        
        d_model = 128
        attention = LigandCrossAttention(d_model, num_heads=4)
        
        batch_size, seq_len, num_atoms = 2, 50, 20
        
        protein_repr = torch.randn(batch_size, seq_len, d_model)
        ligand_repr = torch.randn(batch_size, num_atoms, d_model)
        
        output = attention(protein_repr, ligand_repr)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert torch.isfinite(output).all()
    
    def test_cross_attention_with_coordinates(self):
        """Test cross-attention with coordinate information."""
        
        d_model = 64
        attention = LigandCrossAttention(d_model, num_heads=2)
        
        batch_size, seq_len, num_atoms = 1, 30, 15
        
        protein_repr = torch.randn(batch_size, seq_len, d_model)
        ligand_repr = torch.randn(batch_size, num_atoms, d_model)
        protein_coords = torch.randn(batch_size, seq_len, 3) * 10
        ligand_coords = torch.randn(batch_size, num_atoms, 3) * 5
        
        output = attention(
            protein_repr, ligand_repr,
            protein_coords=protein_coords,
            ligand_coords=ligand_coords
        )
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_cross_attention_with_mask(self):
        """Test cross-attention with ligand mask."""
        
        d_model = 32
        attention = LigandCrossAttention(d_model, num_heads=2)
        
        batch_size, seq_len, num_atoms = 1, 20, 10
        
        protein_repr = torch.randn(batch_size, seq_len, d_model)
        ligand_repr = torch.randn(batch_size, num_atoms, d_model)
        
        # Create mask (first 7 atoms are valid)
        ligand_mask = torch.cat([
            torch.ones(batch_size, 7),
            torch.zeros(batch_size, 3)
        ], dim=1)
        
        output = attention(protein_repr, ligand_repr, ligand_mask=ligand_mask)
        
        assert output.shape == (batch_size, seq_len, d_model)


class TestProteinLigandAttention:
    """Test bidirectional protein-ligand attention."""
    
    def test_bidirectional_attention(self):
        """Test bidirectional attention mechanism."""
        
        d_model = 64
        attention = ProteinLigandAttention(d_model, num_heads=4)
        
        batch_size, seq_len, num_atoms = 1, 25, 12
        
        protein_repr = torch.randn(batch_size, seq_len, d_model)
        ligand_repr = torch.randn(batch_size, num_atoms, d_model)
        
        protein_out, ligand_out = attention(protein_repr, ligand_repr)
        
        assert protein_out.shape == (batch_size, seq_len, d_model)
        assert ligand_out.shape == (batch_size, num_atoms, d_model)
        
        # Should be different from input (attention applied)
        assert not torch.equal(protein_out, protein_repr)
        assert not torch.equal(ligand_out, ligand_repr)


class TestBindingPocketPredictor:
    """Test binding pocket prediction."""
    
    def test_pocket_prediction(self):
        """Test binding pocket prediction."""
        
        d_model = 64
        predictor = BindingPocketPredictor(d_model)
        
        batch_size, seq_len, num_atoms, num_heads = 1, 30, 15, 4
        
        protein_repr = torch.randn(batch_size, seq_len, d_model)
        attention_weights = torch.rand(batch_size, num_heads, seq_len, num_atoms)
        protein_coords = torch.randn(batch_size, seq_len, 3) * 20
        ligand_coords = torch.randn(batch_size, num_atoms, 3) * 5
        
        results = predictor(
            protein_repr, attention_weights,
            protein_coords, ligand_coords
        )
        
        # Check output structure
        assert 'pocket_probabilities' in results
        assert 'pocket_residues' in results
        assert 'min_distances' in results
        assert 'attention_scores' in results
        
        # Check shapes
        assert results['pocket_probabilities'].shape == (batch_size, seq_len)
        assert results['pocket_residues'].shape == (batch_size, seq_len)
        assert results['min_distances'].shape == (batch_size, seq_len)
        assert results['attention_scores'].shape == (batch_size, seq_len)
        
        # Check value ranges
        assert (results['pocket_probabilities'] >= 0).all()
        assert (results['pocket_probabilities'] <= 1).all()
        assert (results['min_distances'] >= 0).all()


class TestLigandUtils:
    """Test ligand utility functions."""
    
    def test_parse_ligand_input_smiles(self):
        """Test parsing SMILES input."""
        
        smiles = "CCO"
        ligand_data = parse_ligand_input(smiles)
        
        assert ligand_data is not None
        assert 'atom_features' in ligand_data
        assert 'positions' in ligand_data
        assert ligand_data['smiles'] == smiles
    
    def test_parse_ligand_input_dict(self):
        """Test parsing dictionary input."""
        
        input_dict = create_mock_ligand(15)
        ligand_data = parse_ligand_input(input_dict)
        
        assert ligand_data == input_dict
    
    def test_calculate_binding_pocket(self):
        """Test binding pocket calculation."""
        
        seq_len, num_atoms = 50, 20
        
        # Create protein coordinates
        protein_coords = torch.randn(seq_len, 3) * 15
        
        # Create ligand coordinates (close to some protein residues)
        ligand_coords = torch.randn(num_atoms, 3) * 3
        
        # Move ligand close to first few residues
        ligand_coords += protein_coords[:5].mean(dim=0)
        
        results = calculate_binding_pocket(
            protein_coords, ligand_coords, cutoff=8.0
        )
        
        # Check output structure
        assert 'pocket_residues' in results
        assert 'pocket_indices' in results
        assert 'min_distances' in results
        assert 'pocket_center' in results
        assert 'pocket_size' in results
        
        # Check shapes
        assert results['pocket_residues'].shape == (seq_len,)
        assert results['min_distances'].shape == (seq_len,)
        assert results['pocket_center'].shape == (3,)
        
        # Should find some pocket residues
        assert results['pocket_size'] > 0


def test_ligand_integration():
    """Integration test for complete ligand pipeline."""
    
    print("🧪 Testing ligand integration pipeline...")
    
    d_model = 128
    batch_size, seq_len, num_atoms = 1, 40, 18
    
    # Create ligand encoder
    ligand_encoder = LigandEncoder(d_model, max_atoms=25)
    
    # Create mock ligand
    ligand_data = create_mock_ligand(num_atoms, d_model)
    
    # Encode ligand
    ligand_results = ligand_encoder(ligand_data)
    
    # Create protein representation
    protein_repr = torch.randn(batch_size, seq_len, d_model)
    
    # Extract ligand embeddings
    ligand_repr = ligand_results['atom_embeddings'].unsqueeze(0)  # Add batch dim
    ligand_mask = ligand_results['atom_mask'].unsqueeze(0) if ligand_results['atom_mask'] is not None else torch.ones(batch_size, max_atoms)
    
    # Create cross-attention
    cross_attention = LigandCrossAttention(d_model, num_heads=8)
    
    # Apply cross-attention
    protein_attended = cross_attention(
        protein_repr, ligand_repr, ligand_mask=ligand_mask
    )
    
    # Create binding pocket predictor
    pocket_predictor = BindingPocketPredictor(d_model)
    
    # Mock attention weights (match ligand encoder max_atoms)
    max_atoms = 25
    attention_weights = torch.rand(batch_size, 8, seq_len, max_atoms)
    protein_coords = torch.randn(batch_size, seq_len, 3) * 15
    ligand_coords = torch.randn(batch_size, max_atoms, 3) * 5  # Use consistent size
    
    # Predict binding pocket
    pocket_results = pocket_predictor(
        protein_attended, attention_weights,
        protein_coords, ligand_coords, ligand_mask
    )
    
    # Validate results
    assert protein_attended.shape == (batch_size, seq_len, d_model)
    assert 'pocket_probabilities' in pocket_results
    assert pocket_results['pocket_probabilities'].shape == (batch_size, seq_len)
    
    print("✅ Ligand integration test passed!")


if __name__ == "__main__":
    # Run integration test
    test_ligand_integration()
    
    # Run all tests
    pytest.main([__file__, "-v"])
