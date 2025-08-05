#!/usr/bin/env python3
"""
Test Suite for Mutation Prediction Modules

Tests ΔΔG prediction, mutation scanning, and stability analysis.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openfoldpp.modules.mutation import (
    DDGPredictor,
    DDGPredictionHead,
    MutationEncoder,
    MutationScanner,
    MutationEffect,
    StabilityPredictor
)
from openfoldpp.modules.mutation.ddg_predictor import (
    amino_acid_to_index,
    index_to_amino_acid,
    create_ddg_loss
)


class TestMutationEncoder:
    """Test mutation encoder."""
    
    def test_mutation_encoder_init(self):
        """Test initialization of MutationEncoder."""
        
        d_model = 256
        encoder = MutationEncoder(d_model)
        
        assert encoder.d_model == d_model
        assert hasattr(encoder, 'aa_embedding')
        assert hasattr(encoder, 'position_embedding')
        assert hasattr(encoder, 'aa_properties')
    
    def test_mutation_encoder_forward(self):
        """Test forward pass of MutationEncoder."""
        
        batch_size = 4
        d_model = 128
        
        encoder = MutationEncoder(d_model)
        
        wt_aa = torch.randint(0, 20, (batch_size,))
        mut_aa = torch.randint(0, 20, (batch_size,))
        position = torch.randint(0, 100, (batch_size,))
        context_features = torch.randn(batch_size, d_model)
        
        output = encoder(wt_aa, mut_aa, position, context_features)
        
        assert output.shape == (batch_size, d_model)
        assert torch.isfinite(output).all()
    
    def test_amino_acid_conversion(self):
        """Test amino acid conversion functions."""
        
        # Test conversion
        assert amino_acid_to_index('A') == 0
        assert amino_acid_to_index('V') == 19
        assert amino_acid_to_index('X') == 20  # Unknown
        
        assert index_to_amino_acid(0) == 'A'
        assert index_to_amino_acid(19) == 'V'
        assert index_to_amino_acid(25) == 'X'  # Out of range


class TestDDGPredictionHead:
    """Test ΔΔG prediction head."""
    
    def test_ddg_head_init(self):
        """Test initialization of DDGPredictionHead."""
        
        d_model, hidden_dim = 256, 512
        head = DDGPredictionHead(d_model, hidden_dim)
        
        assert hasattr(head, 'regression_head')
        assert hasattr(head, 'uncertainty_head')
    
    def test_ddg_head_forward(self):
        """Test forward pass of DDGPredictionHead."""
        
        batch_size = 8
        d_model = 128
        
        head = DDGPredictionHead(d_model, hidden_dim=256)
        
        mutation_features = torch.randn(batch_size, d_model)
        predictions = head(mutation_features)
        
        assert 'ddg_pred' in predictions
        assert 'uncertainty' in predictions
        assert 'confidence' in predictions
        
        assert predictions['ddg_pred'].shape == (batch_size,)
        assert predictions['uncertainty'].shape == (batch_size,)
        assert predictions['confidence'].shape == (batch_size,)
        
        # Check value ranges
        assert (predictions['uncertainty'] >= 0).all()
        assert (predictions['confidence'] >= 0).all()
        assert (predictions['confidence'] <= 1).all()


class TestDDGPredictor:
    """Test complete ΔΔG predictor."""
    
    def test_ddg_predictor_init(self):
        """Test initialization of DDGPredictor."""
        
        structure_dim, d_model = 384, 256
        predictor = DDGPredictor(structure_dim, d_model)
        
        assert predictor.structure_dim == structure_dim
        assert predictor.d_model == d_model
        assert hasattr(predictor, 'mutation_encoder')
        assert hasattr(predictor, 'ddg_head')
    
    def test_ddg_predictor_forward(self):
        """Test forward pass of DDGPredictor."""
        
        batch_size, seq_len = 4, 50
        structure_dim, d_model = 192, 128
        
        predictor = DDGPredictor(structure_dim, d_model)
        
        structure_features = torch.randn(batch_size, seq_len, structure_dim)
        wt_aa = torch.randint(0, 20, (batch_size,))
        mut_aa = torch.randint(0, 20, (batch_size,))
        position = torch.randint(0, seq_len, (batch_size,))
        
        predictions = predictor(structure_features, wt_aa, mut_aa, position)
        
        assert 'ddg_pred' in predictions
        assert 'uncertainty' in predictions
        assert 'wt_aa' in predictions
        assert 'mut_aa' in predictions
        assert 'position' in predictions
        
        assert predictions['ddg_pred'].shape == (batch_size,)
        assert torch.isfinite(predictions['ddg_pred']).all()
    
    def test_predict_single(self):
        """Test single mutation prediction."""
        
        seq_len = 30
        structure_dim, d_model = 96, 64
        
        predictor = DDGPredictor(structure_dim, d_model)
        
        structure_features = torch.randn(seq_len, structure_dim)
        wt_aa, mut_aa, position = 0, 5, 10  # A -> Q at position 10
        
        result = predictor.predict_single(structure_features, wt_aa, mut_aa, position)
        
        assert 'ddg_pred' in result
        assert 'uncertainty' in result
        assert 'confidence' in result
        
        assert isinstance(result['ddg_pred'], float)
        assert isinstance(result['uncertainty'], float)
        assert isinstance(result['confidence'], float)
    
    def test_scan_mutations(self):
        """Test mutation scanning."""
        
        seq_len = 20
        structure_dim, d_model = 96, 64
        
        predictor = DDGPredictor(structure_dim, d_model)
        
        structure_features = torch.randn(seq_len, structure_dim)
        wt_sequence = [amino_acid_to_index(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"]
        positions = [5, 10, 15]
        
        results = predictor.scan_mutations(structure_features, wt_sequence, positions)
        
        assert 'ddg_predictions' in results
        assert 'mutations' in results
        assert 'num_mutations' in results
        
        assert len(results['mutations']) > 0
        assert results['ddg_predictions'].shape[0] == len(results['mutations'])


class TestMutationEffect:
    """Test MutationEffect data class."""
    
    def test_mutation_effect_creation(self):
        """Test creation and categorization of MutationEffect."""
        
        # Destabilizing mutation
        effect1 = MutationEffect(
            position=10, wt_aa='A', mut_aa='P', 
            ddg_pred=3.0, uncertainty=0.5, confidence=0.8
        )
        assert effect1.effect_category == "Destabilizing"
        
        # Stabilizing mutation
        effect2 = MutationEffect(
            position=20, wt_aa='G', mut_aa='A',
            ddg_pred=-1.5, uncertainty=0.3, confidence=0.9
        )
        assert effect2.effect_category == "Mildly Stabilizing"
        
        # Neutral mutation
        effect3 = MutationEffect(
            position=5, wt_aa='L', mut_aa='I',
            ddg_pred=0.1, uncertainty=0.4, confidence=0.7
        )
        assert effect3.effect_category == "Neutral"
    
    def test_mutation_effect_to_dict(self):
        """Test conversion to dictionary."""
        
        effect = MutationEffect(
            position=15, wt_aa='F', mut_aa='Y',
            ddg_pred=-0.8, uncertainty=0.2, confidence=0.9
        )
        
        result_dict = effect.to_dict()
        
        assert 'position' in result_dict
        assert 'mutation' in result_dict
        assert 'ddg_pred' in result_dict
        assert 'effect_category' in result_dict
        
        assert result_dict['mutation'] == "F16Y"  # 1-indexed


class TestMutationScanner:
    """Test mutation scanner."""
    
    def test_mutation_scanner_init(self):
        """Test initialization of MutationScanner."""
        
        predictor = DDGPredictor(96, 64)
        scanner = MutationScanner(predictor)
        
        assert scanner.ddg_predictor == predictor
        assert hasattr(scanner, 'aa_groups')
        assert hasattr(scanner, 'conservation_weights')
    
    def test_scan_position(self):
        """Test scanning mutations at a specific position."""
        
        predictor = DDGPredictor(96, 64)
        scanner = MutationScanner(predictor)
        
        seq_len = 25
        structure_features = torch.randn(seq_len, 96)
        sequence = "ACDEFGHIKLMNPQRSTVWYACDEF"
        position = 10
        
        mutations = scanner.scan_position(structure_features, sequence, position)
        
        assert len(mutations) > 0
        assert all(isinstance(mut, MutationEffect) for mut in mutations)
        assert all(mut.position == position for mut in mutations)
        assert all(mut.wt_aa == sequence[position] for mut in mutations)
        
        # Should be sorted by ΔΔG
        ddg_values = [mut.ddg_pred for mut in mutations]
        assert ddg_values == sorted(ddg_values)
    
    def test_find_stabilizing_mutations(self):
        """Test finding stabilizing mutations."""
        
        predictor = DDGPredictor(96, 64)
        scanner = MutationScanner(predictor)
        
        seq_len = 15
        structure_features = torch.randn(seq_len, 96)
        sequence = "ACDEFGHIKLMNPQR"
        
        stabilizing = scanner.find_stabilizing_mutations(
            structure_features, sequence, ddg_threshold=-0.1, top_k=5
        )
        
        # All should be stabilizing
        assert all(mut.ddg_pred < -0.1 for mut in stabilizing)
        assert len(stabilizing) <= 5


class TestStabilityPredictor:
    """Test stability predictor."""
    
    def test_stability_predictor_init(self):
        """Test initialization of StabilityPredictor."""

        ddg_predictor = DDGPredictor(96, 64)
        stability_predictor = StabilityPredictor(ddg_predictor, structure_dim=96)

        assert stability_predictor.ddg_predictor == ddg_predictor
        assert hasattr(stability_predictor, 'stability_head')
        assert hasattr(stability_predictor, 'condition_encoder')
    
    def test_stability_prediction(self):
        """Test stability prediction."""

        ddg_predictor = DDGPredictor(96, 64)
        stability_predictor = StabilityPredictor(ddg_predictor, structure_dim=96)
        
        seq_len = 20
        structure_features = torch.randn(seq_len, 96)
        sequence = [amino_acid_to_index(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"]
        
        result = stability_predictor(structure_features, sequence)
        
        assert 'stability_free_energy' in result
        assert 'melting_temperature' in result
        assert 'folding_probability' in result
        
        # Check value ranges
        assert result['melting_temperature'] > 273.0  # Above freezing (with some tolerance)
        assert 0 <= result['folding_probability'] <= 1
    
    def test_melting_curve_prediction(self):
        """Test melting curve prediction."""

        ddg_predictor = DDGPredictor(96, 64)
        stability_predictor = StabilityPredictor(ddg_predictor, structure_dim=96)
        
        seq_len = 15
        structure_features = torch.randn(seq_len, 96)
        sequence = [amino_acid_to_index(aa) for aa in "ACDEFGHIKLMNPQR"]
        
        curve = stability_predictor.predict_melting_curve(
            structure_features, sequence, num_points=10
        )
        
        assert 'temperature' in curve
        assert 'fraction_folded' in curve
        assert 'temperature_celsius' in curve
        
        assert len(curve['temperature']) == 10
        assert len(curve['fraction_folded']) == 10
        
        # Fraction folded should decrease with temperature
        assert curve['fraction_folded'][0] > curve['fraction_folded'][-1]


class TestDDGLoss:
    """Test ΔΔG loss function."""
    
    def test_ddg_loss_creation(self):
        """Test creation of ΔΔG loss function."""
        
        loss_fn = create_ddg_loss()
        assert callable(loss_fn)
    
    def test_ddg_loss_computation(self):
        """Test ΔΔG loss computation."""
        
        loss_fn = create_ddg_loss()
        
        batch_size = 8
        predictions = {
            'ddg_pred': torch.randn(batch_size),
            'uncertainty': torch.rand(batch_size) + 0.1  # Positive uncertainty
        }
        targets = torch.randn(batch_size)
        
        losses = loss_fn(predictions, targets)
        
        assert 'total_loss' in losses
        assert 'mse_loss' in losses
        assert 'uncertainty_loss' in losses
        assert 'mae' in losses
        
        assert torch.isfinite(losses['total_loss'])
        assert losses['total_loss'] >= 0


def test_mutation_prediction_integration():
    """Integration test for complete mutation prediction pipeline."""
    
    print("🧪 Testing mutation prediction integration...")
    
    # Create components
    structure_dim, d_model = 96, 64
    ddg_predictor = DDGPredictor(structure_dim, d_model)
    scanner = MutationScanner(ddg_predictor)
    stability_predictor = StabilityPredictor(ddg_predictor, structure_dim=structure_dim)
    
    # Test data
    seq_len = 20
    structure_features = torch.randn(seq_len, structure_dim)
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    
    # Test single mutation prediction
    result = ddg_predictor.predict_single(
        structure_features, 
        amino_acid_to_index('A'), 
        amino_acid_to_index('V'), 
        5
    )
    assert 'ddg_pred' in result
    print(f"Single mutation ΔΔG: {result['ddg_pred']:.3f}")
    
    # Test mutation scanning
    mutations = scanner.scan_position(structure_features, sequence, 10)
    assert len(mutations) > 0
    print(f"Scanned {len(mutations)} mutations at position 10")
    
    # Test stability prediction
    sequence_indices = [amino_acid_to_index(aa) for aa in sequence]
    stability = stability_predictor(structure_features, sequence_indices)
    assert 'stability_free_energy' in stability
    print(f"Stability ΔG: {stability['stability_free_energy']:.3f} kJ/mol")
    
    # Test melting curve
    curve = stability_predictor.predict_melting_curve(
        structure_features, sequence_indices, num_points=5
    )
    assert len(curve['temperature']) == 5
    print(f"Melting curve: {curve['fraction_folded'][0]:.3f} -> {curve['fraction_folded'][-1]:.3f}")
    
    # Test mutation report generation
    report = scanner.generate_mutation_report(structure_features, sequence, "TestProtein")
    assert 'protein_name' in report
    assert 'statistics' in report
    assert 'top_stabilizing' in report
    print(f"Generated report for {report['protein_name']} with {report['total_mutations_scanned']} mutations")
    
    print("✅ Mutation prediction integration test passed!")


if __name__ == "__main__":
    # Run integration test
    test_mutation_prediction_integration()
    
    # Run all tests
    pytest.main([__file__, "-v"])
