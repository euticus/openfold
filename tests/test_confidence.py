#!/usr/bin/env python3
"""
Test Suite for OdinFold Confidence Prediction

Tests TM-score prediction, confidence estimation, and early exit functionality.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openfoldpp.confidence import (
    TMScorePredictor,
    TMScorePredictionHead,
    ConfidenceEstimator,
    SequenceComplexityAnalyzer,
    EarlyExitManager,
    BatchRanker
)
from openfoldpp.confidence.early_exit import EarlyExitConfig


class TestTMScorePredictionHead:
    """Test TM-score prediction head."""
    
    def test_prediction_head_forward(self):
        """Test forward pass of TM-score prediction head."""
        
        d_model = 256
        head = TMScorePredictionHead(d_model)
        
        batch_size, seq_len = 4, 50
        esm_features = torch.randn(batch_size, seq_len, d_model)
        sequence_lengths = torch.tensor([45, 50, 30, 40])
        
        results = head(esm_features, sequence_lengths)
        
        # Check output structure
        assert 'tm_score_pred' in results
        assert 'uncertainty' in results
        assert 'confidence' in results
        assert 'sequence_repr' in results
        
        # Check shapes
        assert results['tm_score_pred'].shape == (batch_size,)
        assert results['uncertainty'].shape == (batch_size,)
        assert results['confidence'].shape == (batch_size,)
        assert results['sequence_repr'].shape == (batch_size, d_model)
        
        # Check value ranges
        assert (results['tm_score_pred'] >= 0).all()
        assert (results['tm_score_pred'] <= 1).all()
        assert (results['uncertainty'] >= 0).all()
        assert (results['confidence'] >= 0).all()
        assert (results['confidence'] <= 1).all()
    
    def test_prediction_head_with_complexity(self):
        """Test prediction head with custom complexity features."""
        
        d_model = 128
        head = TMScorePredictionHead(d_model)
        
        batch_size, seq_len = 2, 30
        esm_features = torch.randn(batch_size, seq_len, d_model)
        sequence_lengths = torch.tensor([25, 30])
        complexity_features = torch.randn(batch_size, 10)
        
        results = head(esm_features, sequence_lengths, complexity_features)
        
        assert results['tm_score_pred'].shape == (batch_size,)
        assert torch.isfinite(results['tm_score_pred']).all()


class TestTMScorePredictor:
    """Test complete TM-score predictor."""
    
    def test_predictor_forward(self):
        """Test forward pass of TM-score predictor."""
        
        # Mock ESM model
        class MockESM:
            pass
        
        predictor = TMScorePredictor(MockESM(), d_model=256)
        
        sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "ACDEFGHIKLMNPQRSTVWY",
            "MKTVQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGGMKTV"
        ]
        
        results = predictor.forward(sequences)
        
        # Check output structure
        assert 'tm_score_pred' in results
        assert 'uncertainty' in results
        assert 'confidence' in results
        assert 'quality_category' in results
        assert 'sequence_lengths' in results
        
        # Check shapes
        batch_size = len(sequences)
        assert results['tm_score_pred'].shape == (batch_size,)
        assert results['quality_category'].shape == (batch_size,)
        assert results['sequence_lengths'].shape == (batch_size,)
        
        # Check value ranges
        assert (results['tm_score_pred'] >= 0).all()
        assert (results['tm_score_pred'] <= 1).all()
        assert (results['quality_category'] >= 0).all()
        assert (results['quality_category'] <= 3).all()
    
    def test_predict_single(self):
        """Test single sequence prediction."""
        
        predictor = TMScorePredictor(None, d_model=128)
        
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        result = predictor.predict_single(sequence)
        
        assert 'tm_score_pred' in result
        assert 'uncertainty' in result
        assert 'confidence' in result
        assert 'quality_category' in result
        assert 'sequence_length' in result
        
        # Check types and ranges
        assert isinstance(result['tm_score_pred'], float)
        assert 0.0 <= result['tm_score_pred'] <= 1.0
        assert 0.0 <= result['confidence'] <= 1.0
        assert 0 <= result['quality_category'] <= 3
    
    def test_rank_sequences(self):
        """Test sequence ranking by TM-score."""
        
        predictor = TMScorePredictor(None, d_model=64)
        
        sequences = [
            "ACDEFGHIKLMNPQRSTVWY",
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "MKTVQERLK",
            "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
        ]
        
        ranking = predictor.rank_sequences(sequences, top_k=3)
        
        assert len(ranking) == 3
        
        # Check ranking structure
        for idx, seq, score in ranking:
            assert isinstance(idx, int)
            assert isinstance(seq, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
            assert seq in sequences
        
        # Check ordering (should be descending)
        scores = [score for _, _, score in ranking]
        assert scores == sorted(scores, reverse=True)


class TestSequenceComplexityAnalyzer:
    """Test sequence complexity analysis."""
    
    def test_analyze_sequence(self):
        """Test comprehensive sequence analysis."""
        
        analyzer = SequenceComplexityAnalyzer()
        
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        analysis = analyzer.analyze_sequence(sequence)
        
        # Check required fields
        required_fields = [
            'hydrophobic_fraction', 'polar_fraction', 'charged_fraction',
            'low_complexity_score', 'repeat_content', 'shannon_entropy',
            'overall_complexity', 'sequence_length'
        ]
        
        for field in required_fields:
            assert field in analysis
            assert isinstance(analysis[field], float)
        
        # Check value ranges
        assert 0.0 <= analysis['hydrophobic_fraction'] <= 1.0
        assert 0.0 <= analysis['polar_fraction'] <= 1.0
        assert 0.0 <= analysis['charged_fraction'] <= 1.0
        assert 0.0 <= analysis['low_complexity_score'] <= 1.0
        assert 0.0 <= analysis['repeat_content'] <= 1.0
        assert 0.0 <= analysis['overall_complexity'] <= 1.0
        assert analysis['sequence_length'] == len(sequence)
    
    def test_analyze_simple_sequences(self):
        """Test analysis of simple sequences."""
        
        analyzer = SequenceComplexityAnalyzer()
        
        # Low complexity sequence
        low_complexity = "AAAAAAAAAA"
        analysis_low = analyzer.analyze_sequence(low_complexity)
        assert analysis_low['low_complexity_score'] > 0.5
        
        # Repeat sequence
        repeat_seq = "ABCABCABCABC"
        analysis_repeat = analyzer.analyze_sequence(repeat_seq)
        assert analysis_repeat['repeat_content'] > 0.0
        
        # Diverse sequence
        diverse_seq = "ACDEFGHIKLMNPQRSTVWY"
        analysis_diverse = analyzer.analyze_sequence(diverse_seq)
        assert analysis_diverse['normalized_entropy'] > 0.8
    
    def test_empty_sequence(self):
        """Test handling of empty sequence."""
        
        analyzer = SequenceComplexityAnalyzer()
        analysis = analyzer.analyze_sequence("")
        
        # Should return default values
        assert analysis['sequence_length'] == 0
        assert analysis['overall_complexity'] == 0.0


class TestConfidenceEstimator:
    """Test confidence estimation."""
    
    def test_estimate_confidence(self):
        """Test confidence estimation for batch of sequences."""
        
        estimator = ConfidenceEstimator()
        
        sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "ACDEFGHIKLMNPQRSTVWY",
            "AAAAAAAAAA",  # Low complexity
            "ABCABCABCABC"  # Repetitive
        ]
        
        results = estimator.estimate_confidence(sequences)
        
        # Check output structure
        assert 'overall_confidence' in results
        assert 'complexity_confidence' in results
        assert 'tm_confidence' in results
        assert 'confidence_categories' in results
        
        # Check shapes
        batch_size = len(sequences)
        assert results['overall_confidence'].shape == (batch_size,)
        assert results['complexity_confidence'].shape == (batch_size,)
        assert results['confidence_categories'].shape == (batch_size,)
        
        # Check value ranges
        assert (results['overall_confidence'] >= 0).all()
        assert (results['overall_confidence'] <= 1).all()
        assert (results['confidence_categories'] >= 0).all()
        assert (results['confidence_categories'] <= 3).all()
    
    def test_should_skip_folding(self):
        """Test early exit decisions."""
        
        estimator = ConfidenceEstimator()
        
        sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "AAAAAAAAAA"  # Should likely be skipped
        ]
        
        skip_flags = estimator.should_skip_folding(sequences, confidence_threshold=0.5)
        
        assert len(skip_flags) == len(sequences)
        assert all(isinstance(flag, bool) for flag in skip_flags)
    
    def test_get_folding_priority(self):
        """Test folding priority calculation."""
        
        estimator = ConfidenceEstimator()
        
        sequences = [
            "ACDEFGHIKLMNPQRSTVWY",
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "AAAAAAAAAA"
        ]
        
        priorities = estimator.get_folding_priority(sequences)
        
        assert len(priorities) == len(sequences)
        
        # Check structure
        for idx, conf in priorities:
            assert isinstance(idx, int)
            assert isinstance(conf, float)
            assert 0 <= idx < len(sequences)
            assert 0.0 <= conf <= 1.0
        
        # Should be sorted by confidence (descending)
        confidences = [conf for _, conf in priorities]
        assert confidences == sorted(confidences, reverse=True)


class TestEarlyExitManager:
    """Test early exit management."""
    
    def test_early_exit_manager_init(self):
        """Test early exit manager initialization."""
        
        config = EarlyExitConfig(min_confidence_threshold=0.4)
        manager = EarlyExitManager(config)
        
        assert manager.config.min_confidence_threshold == 0.4
        assert manager.stats['total_sequences'] == 0
    
    def test_process_batch(self):
        """Test batch processing with early exits."""
        
        manager = EarlyExitManager()
        
        sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "ACDEFGHIKLMNPQRSTVWY",
            "AAAAAAAAAA"  # Low complexity
        ]
        
        # Mock fold function
        def mock_fold(seq):
            return {'sequence': seq, 'tm_score': 0.75, 'folded': True}
        
        results = manager.process_batch(sequences, fold_function=mock_fold)
        
        # Check output structure
        assert 'processing_results' in results
        assert 'exit_decisions' in results
        assert 'statistics' in results
        assert 'batch_time' in results
        
        # Check processing results
        proc_results = results['processing_results']
        assert 'folded_results' in proc_results
        assert 'skipped_results' in proc_results
        assert 'folded_count' in proc_results
        assert 'skipped_count' in proc_results
        
        # Should have processed some sequences
        total_processed = proc_results['folded_count'] + proc_results['skipped_count']
        assert total_processed == len(sequences)
    
    def test_get_statistics_summary(self):
        """Test statistics summary."""
        
        manager = EarlyExitManager()
        
        # Process some sequences to generate stats
        sequences = ["ACDEFGHIKLMNPQRSTVWY", "AAAAAAAAAA"]
        manager.process_batch(sequences)
        
        summary = manager.get_statistics_summary()
        
        assert 'total_sequences_processed' in summary
        assert 'folding_rate' in summary
        assert 'skip_rate' in summary
        assert 'exit_breakdown' in summary
        
        assert summary['total_sequences_processed'] > 0


class TestBatchRanker:
    """Test batch ranking functionality."""
    
    def test_rank_by_confidence(self):
        """Test ranking by confidence."""
        
        ranker = BatchRanker()
        
        sequences = [
            "ACDEFGHIKLMNPQRSTVWY",
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "AAAAAAAAAA"
        ]
        
        ranking = ranker.rank_sequences(sequences, ranking_strategy="confidence")
        
        assert len(ranking) == len(sequences)
        
        # Check ranking structure
        for idx, seq, score in ranking:
            assert isinstance(idx, int)
            assert isinstance(seq, str)
            assert isinstance(score, float)
            assert seq in sequences
        
        # Should be sorted by score (descending)
        scores = [score for _, _, score in ranking]
        assert scores == sorted(scores, reverse=True)
    
    def test_rank_by_combined_score(self):
        """Test ranking by combined score."""
        
        ranker = BatchRanker()
        
        sequences = [
            "ACDEFGHIKLMNPQRSTVWY",
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        ]
        
        ranking = ranker.rank_sequences(sequences, ranking_strategy="combined")
        
        assert len(ranking) == len(sequences)
        
        # Check that all sequences are included
        ranked_sequences = [seq for _, seq, _ in ranking]
        assert set(ranked_sequences) == set(sequences)


def test_confidence_integration():
    """Integration test for complete confidence prediction pipeline."""
    
    print("🧪 Testing confidence prediction integration...")
    
    # Create test sequences
    sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # Good sequence
        "ACDEFGHIKLMNPQRSTVWY",  # Diverse sequence
        "AAAAAAAAAA",  # Low complexity
        "ABCABCABCABC",  # Repetitive
        "MKTVQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGGMKTV"  # Long sequence
    ]
    
    # Test TM-score predictor
    tm_predictor = TMScorePredictor(None, d_model=256)
    tm_results = tm_predictor.forward(sequences)
    
    assert 'tm_score_pred' in tm_results
    assert tm_results['tm_score_pred'].shape == (len(sequences),)
    
    # Test confidence estimator
    confidence_estimator = ConfidenceEstimator(tm_predictor)
    conf_results = confidence_estimator.estimate_confidence(sequences)
    
    assert 'overall_confidence' in conf_results
    assert conf_results['overall_confidence'].shape == (len(sequences),)
    
    # Test early exit manager
    early_exit_manager = EarlyExitManager(
        confidence_estimator=confidence_estimator,
        tm_score_predictor=tm_predictor
    )
    
    def mock_fold(seq):
        return {'sequence': seq, 'tm_score': 0.75}
    
    exit_results = early_exit_manager.process_batch(sequences, fold_function=mock_fold)
    
    assert 'processing_results' in exit_results
    assert 'statistics' in exit_results
    
    # Test batch ranker
    ranker = BatchRanker(confidence_estimator, tm_predictor)
    ranking = ranker.rank_sequences(sequences)
    
    assert len(ranking) == len(sequences)
    
    # Get statistics
    stats = early_exit_manager.get_statistics_summary()
    assert 'total_sequences_processed' in stats
    
    print("✅ Confidence prediction integration test passed!")


if __name__ == "__main__":
    # Run integration test
    test_confidence_integration()
    
    # Run all tests
    pytest.main([__file__, "-v"])
