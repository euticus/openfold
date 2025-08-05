#!/usr/bin/env python3
"""
Test Suite for OdinFold Multimer Functionality

Tests multi-chain protein complex folding capabilities.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openfoldpp.multimer import (
    ChainBreakProcessor, 
    MultimerPositionalEncoding,
    MultimerAttentionMask,
    InterChainAttention,
    MultimerFoldRunner
)


class TestChainBreakProcessor:
    """Test chain break processing functionality."""
    
    def test_basic_chain_processing(self):
        """Test basic two-chain processing."""
        
        processor = ChainBreakProcessor()
        sequences = ["MKTVR", "QERLK"]
        
        result = processor.process_chains(sequences)
        
        assert result['num_chains'] == 2
        assert result['concatenated_sequence'] == "MKTVR|QERLK"
        assert result['total_length'] == 11
        assert len(result['chain_infos']) == 2
        
        # Check chain info
        chain_a = result['chain_infos'][0]
        assert chain_a.chain_id == 'A'
        assert chain_a.sequence == 'MKTVR'
        assert chain_a.start_pos == 0
        assert chain_a.end_pos == 4
        
        chain_b = result['chain_infos'][1]
        assert chain_b.chain_id == 'B'
        assert chain_b.sequence == 'QERLK'
        assert chain_b.start_pos == 6  # After chain break token
        assert chain_b.end_pos == 10
    
    def test_chain_id_tensor(self):
        """Test chain ID tensor creation."""
        
        processor = ChainBreakProcessor()
        sequences = ["ABC", "DEF"]
        
        result = processor.process_chains(sequences)
        chain_id_tensor = result['chain_id_tensor']
        
        expected = torch.tensor([0, 0, 0, 0, 1, 1, 1])  # Including chain break
        assert torch.equal(chain_id_tensor, expected)
    
    def test_invalid_sequence(self):
        """Test handling of invalid amino acid sequences."""
        
        processor = ChainBreakProcessor()
        sequences = ["MKTXR", "QERLK"]  # X is invalid
        
        with pytest.raises(ValueError, match="Invalid amino acids"):
            processor.process_chains(sequences)
    
    def test_coordinate_splitting(self):
        """Test splitting coordinates back into chains."""
        
        processor = ChainBreakProcessor()
        sequences = ["ABC", "DEF"]
        
        result = processor.process_chains(sequences)
        
        # Mock coordinates
        coordinates = torch.randn(7, 3)  # 3+1+3 positions
        
        chain_coords = processor.split_coordinates(coordinates, result['chain_infos'])
        
        assert 'A' in chain_coords
        assert 'B' in chain_coords
        assert chain_coords['A'].shape == (3, 3)
        assert chain_coords['B'].shape == (3, 3)


class TestMultimerPositionalEncoding:
    """Test multimer positional encoding."""
    
    def test_basic_encoding(self):
        """Test basic positional encoding with chains."""
        
        d_model = 64
        encoder = MultimerPositionalEncoding(d_model)
        
        seq_len = 10
        x = torch.randn(1, seq_len, d_model)
        chain_id_tensor = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        
        encoded = encoder(x, chain_id_tensor)
        
        assert encoded.shape == (1, seq_len, d_model)
        assert not torch.equal(encoded, x)  # Should be different after encoding
    
    def test_relative_position_encoding(self):
        """Test relative position encoding for attention bias."""
        
        d_model = 32
        encoder = MultimerPositionalEncoding(d_model)
        
        seq_len = 6
        chain_id_tensor = torch.tensor([0, 0, 0, 1, 1, 1])
        
        rel_pos_encoding = encoder.get_relative_position_encoding(seq_len, chain_id_tensor)
        
        assert rel_pos_encoding.shape == (seq_len, seq_len, d_model)


class TestMultimerAttentionMask:
    """Test multimer attention masking."""
    
    def test_basic_mask_creation(self):
        """Test basic attention mask creation."""
        
        mask_creator = MultimerAttentionMask()
        chain_id_tensor = torch.tensor([0, 0, 1, 1])
        
        mask = mask_creator.create_multimer_mask(chain_id_tensor)
        
        assert mask.shape == (4, 4)
        assert mask.dtype == torch.bool
    
    def test_inter_chain_masking(self):
        """Test inter-chain attention masking."""
        
        mask_creator = MultimerAttentionMask(mask_inter_chain=True)
        chain_id_tensor = torch.tensor([0, 0, 1, 1])
        
        mask = mask_creator.create_multimer_mask(chain_id_tensor)
        
        # Check that inter-chain positions are masked
        assert mask[0, 2] == True  # Chain 0 -> Chain 1 should be masked
        assert mask[2, 0] == True  # Chain 1 -> Chain 0 should be masked
        assert mask[0, 1] == False  # Within chain 0 should not be masked
        assert mask[2, 3] == False  # Within chain 1 should not be masked
    
    def test_chain_break_masking(self):
        """Test chain break position masking."""
        
        mask_creator = MultimerAttentionMask(allow_chain_breaks=False)
        chain_id_tensor = torch.tensor([0, 0, 0, 1, 1])
        chain_break_mask = torch.tensor([False, False, True, False, False])  # Position 2 is chain break
        
        mask = mask_creator.create_multimer_mask(chain_id_tensor, chain_break_mask)
        
        # Check that chain break positions are properly masked
        assert mask[0, 2] == True  # Cannot attend to chain break
        assert mask[2, 0] == True  # Chain break cannot attend to others


class TestInterChainAttention:
    """Test inter-chain attention mechanism."""
    
    def test_attention_forward(self):
        """Test forward pass of inter-chain attention."""
        
        d_model = 64
        attention = InterChainAttention(d_model, num_heads=4)
        
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, d_model)
        chain_id_tensor = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        
        output = attention(x, chain_id_tensor)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_attention_with_distances(self):
        """Test attention with distance bias."""
        
        d_model = 32
        attention = InterChainAttention(d_model, num_heads=2)
        
        batch_size, seq_len = 1, 6
        x = torch.randn(batch_size, seq_len, d_model)
        chain_id_tensor = torch.tensor([0, 0, 0, 1, 1, 1])
        distances = torch.rand(seq_len, seq_len) * 10  # Mock distances
        
        output = attention(x, chain_id_tensor, distances=distances)
        
        assert output.shape == (batch_size, seq_len, d_model)


class TestMultimerFoldRunner:
    """Test the complete multimer folding pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        
        # Mock model
        class MockModel:
            def to(self, device):
                return self
        
        self.mock_model = MockModel()
        self.runner = MultimerFoldRunner(self.mock_model, device="cpu")
    
    def test_fold_from_sequences(self):
        """Test folding from sequence list."""
        
        sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "MKTVQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        ]
        
        results = self.runner.fold_multimer_from_sequences(sequences)
        
        # Check basic structure
        assert 'multimer_info' in results
        assert 'coordinates' in results
        assert 'confidence_scores' in results
        assert 'interface_analysis' in results
        
        # Check multimer info
        multimer_info = results['multimer_info']
        assert multimer_info['num_chains'] == 2
        assert multimer_info['chain_ids'] == ['A', 'B']
        
        # Check coordinates
        coords = results['coordinates']
        assert 'full_complex' in coords
        assert 'by_chain' in coords
        assert 'A' in coords['by_chain']
        assert 'B' in coords['by_chain']
        
        # Check interface analysis
        assert isinstance(results['interface_analysis'], dict)
    
    def test_fold_from_fasta(self):
        """Test folding from FASTA file."""
        
        fasta_content = """>Chain_A
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
>Chain_B
MKTVQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            fasta_path = f.name
        
        try:
            results = self.runner.fold_multimer_from_fasta(fasta_path)
            
            assert results['multimer_info']['num_chains'] == 2
            assert results['multimer_info']['chain_ids'] == ['Chain_A', 'Chain_B']
            
        finally:
            Path(fasta_path).unlink()  # Clean up
    
    def test_validation_errors(self):
        """Test input validation errors."""
        
        # Test single chain (should fail)
        with pytest.raises(ValueError, match="Invalid multimer input"):
            self.runner.fold_multimer_from_sequences(["MKTVRQERLK"])
        
        # Test invalid amino acids
        with pytest.raises(ValueError, match="Invalid multimer input"):
            self.runner.fold_multimer_from_sequences(["MKTXR", "QERLK"])
    
    def test_interface_prediction(self):
        """Test interface residue prediction."""
        
        # Create mock coordinates with clear interface
        coordinates = torch.tensor([
            [0.0, 0.0, 0.0],  # Chain A
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],  # Chain B (close to chain A)
            [6.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ])
        
        chain_id_tensor = torch.tensor([0, 0, 0, 1, 1, 1])
        
        interface_residues = self.runner._predict_interface_residues(
            coordinates, chain_id_tensor, cutoff=6.0
        )
        
        # Should find interface between chains
        assert len(interface_residues) > 0


def test_multimer_integration():
    """Integration test for complete multimer pipeline."""
    
    # Test sequences (short for speed)
    sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "QERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGGMKTV"
    ]
    
    # Mock model
    class MockModel:
        def to(self, device):
            return self
    
    # Run complete pipeline
    runner = MultimerFoldRunner(MockModel(), device="cpu")
    results = runner.fold_multimer_from_sequences(sequences)
    
    # Validate results structure
    assert isinstance(results, dict)
    assert 'multimer_info' in results
    assert 'coordinates' in results
    assert 'confidence_scores' in results
    assert 'interface_analysis' in results
    assert 'chain_metrics' in results
    assert 'overall_metrics' in results
    
    # Validate metrics
    overall_metrics = results['overall_metrics']
    assert 'tm_score_estimate' in overall_metrics
    assert 'interface_quality' in overall_metrics
    assert 'complex_compactness' in overall_metrics
    
    print("✅ Multimer integration test passed!")


if __name__ == "__main__":
    # Run integration test
    test_multimer_integration()
    
    # Run all tests
    pytest.main([__file__, "-v"])
