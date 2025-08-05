"""
Early Exit Manager for OdinFold

Manages early exits and batch ranking based on confidence predictions
to optimize computational efficiency for large-scale folding.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EarlyExitConfig:
    """Configuration for early exit strategies."""
    
    # Confidence thresholds
    min_confidence_threshold: float = 0.3
    high_confidence_threshold: float = 0.8
    
    # TM-score thresholds
    min_tm_score_threshold: float = 0.5
    high_tm_score_threshold: float = 0.75
    
    # Batch processing
    max_batch_size: int = 32
    priority_batch_size: int = 8
    
    # Early exit strategies
    enable_confidence_exit: bool = True
    enable_tm_score_exit: bool = True
    enable_complexity_exit: bool = True
    
    # Resource limits
    max_folding_time_per_sequence: float = 300.0  # seconds
    max_total_folding_time: float = 3600.0  # seconds
    
    # Quality control
    require_validation: bool = True
    validation_sample_rate: float = 0.1


class EarlyExitManager:
    """
    Manages early exits and computational resource allocation for OdinFold.
    
    Uses confidence predictions and TM-score estimates to make intelligent
    decisions about which sequences to fold and in what order.
    """
    
    def __init__(self, config: Optional[EarlyExitConfig] = None,
                 confidence_estimator=None, tm_score_predictor=None):
        self.config = config or EarlyExitConfig()
        self.confidence_estimator = confidence_estimator
        self.tm_score_predictor = tm_score_predictor
        
        # Statistics tracking
        self.stats = {
            'total_sequences': 0,
            'folded_sequences': 0,
            'skipped_sequences': 0,
            'early_exits': 0,
            'total_time_saved': 0.0,
            'confidence_exits': 0,
            'tm_score_exits': 0,
            'complexity_exits': 0
        }
    
    def process_batch(self, sequences: List[str], 
                     fold_function=None,
                     return_predictions: bool = True) -> Dict:
        """
        Process a batch of sequences with early exit logic.
        
        Args:
            sequences: List of amino acid sequences
            fold_function: Function to call for actual folding
            return_predictions: Whether to return confidence predictions
            
        Returns:
            Dictionary with processing results and statistics
        """
        
        start_time = time.time()
        batch_size = len(sequences)
        
        logger.info(f"Processing batch of {batch_size} sequences")
        
        # Get confidence predictions
        if self.confidence_estimator:
            confidence_results = self.confidence_estimator.estimate_confidence(sequences)
        else:
            confidence_results = self._mock_confidence_predictions(sequences)
        
        # Get TM-score predictions
        if self.tm_score_predictor:
            tm_results = self.tm_score_predictor.forward(sequences)
        else:
            tm_results = self._mock_tm_predictions(sequences)
        
        # Make early exit decisions
        exit_decisions = self._make_exit_decisions(confidence_results, tm_results, sequences)
        
        # Process sequences based on decisions
        processing_results = self._process_sequences(
            sequences, exit_decisions, fold_function
        )
        
        # Update statistics
        self._update_statistics(exit_decisions, processing_results, time.time() - start_time)
        
        # Compile results
        results = {
            'processing_results': processing_results,
            'exit_decisions': exit_decisions,
            'statistics': self.stats.copy(),
            'batch_time': time.time() - start_time
        }
        
        if return_predictions:
            results['confidence_predictions'] = confidence_results
            results['tm_predictions'] = tm_results
        
        logger.info(f"Batch processed in {results['batch_time']:.2f}s. "
                   f"Folded: {processing_results['folded_count']}, "
                   f"Skipped: {processing_results['skipped_count']}")
        
        return results
    
    def _make_exit_decisions(self, confidence_results: Dict, 
                           tm_results: Dict, sequences: List[str]) -> Dict:
        """Make early exit decisions for each sequence."""
        
        batch_size = len(sequences)
        decisions = {
            'should_fold': [],
            'exit_reasons': [],
            'priorities': [],
            'confidence_scores': confidence_results['overall_confidence'],
            'tm_scores': tm_results['tm_score_pred']
        }
        
        for i in range(batch_size):
            confidence = float(confidence_results['overall_confidence'][i])
            tm_score = float(tm_results['tm_score_pred'][i])
            sequence = sequences[i]
            
            should_fold, reason, priority = self._evaluate_sequence(
                sequence, confidence, tm_score
            )
            
            decisions['should_fold'].append(should_fold)
            decisions['exit_reasons'].append(reason)
            decisions['priorities'].append(priority)
        
        return decisions
    
    def _evaluate_sequence(self, sequence: str, confidence: float, 
                         tm_score: float) -> Tuple[bool, str, float]:
        """Evaluate whether a single sequence should be folded."""
        
        # Check confidence threshold
        if self.config.enable_confidence_exit and confidence < self.config.min_confidence_threshold:
            return False, "low_confidence", 0.0
        
        # Check TM-score threshold
        if self.config.enable_tm_score_exit and tm_score < self.config.min_tm_score_threshold:
            return False, "low_tm_score", 0.0
        
        # Check sequence complexity
        if self.config.enable_complexity_exit:
            complexity_score = self._assess_sequence_complexity(sequence)
            if complexity_score > 0.9:  # Very high complexity
                return False, "high_complexity", 0.0
        
        # Calculate priority score
        priority = self._calculate_priority(confidence, tm_score, len(sequence))
        
        return True, "fold", priority
    
    def _assess_sequence_complexity(self, sequence: str) -> float:
        """Assess sequence complexity for early exit decisions."""
        
        # Simple complexity assessment
        length = len(sequence)
        
        # Check for low complexity regions
        low_complexity_score = 0.0
        window_size = min(20, length // 4)
        
        if window_size >= 3:
            for i in range(length - window_size + 1):
                window = sequence[i:i + window_size]
                unique_aas = len(set(window))
                if unique_aas <= 3:  # Very low diversity
                    low_complexity_score += 1
            
            low_complexity_score /= (length - window_size + 1)
        
        # Check for repeats
        repeat_score = 0.0
        for repeat_len in range(2, min(6, length // 3)):
            for i in range(length - repeat_len * 3 + 1):
                motif = sequence[i:i + repeat_len]
                if sequence[i + repeat_len:i + 2 * repeat_len] == motif:
                    if sequence[i + 2 * repeat_len:i + 3 * repeat_len] == motif:
                        repeat_score += repeat_len / length
        
        # Overall complexity score
        complexity = max(low_complexity_score, repeat_score)
        
        return min(complexity, 1.0)
    
    def _calculate_priority(self, confidence: float, tm_score: float, length: int) -> float:
        """Calculate folding priority score."""
        
        # Base priority from confidence and TM-score
        priority = 0.6 * confidence + 0.4 * tm_score
        
        # Length penalty (longer sequences are more expensive)
        length_penalty = min(1.0, 100.0 / length)  # Penalty for sequences > 100 residues
        priority *= length_penalty
        
        return priority
    
    def _process_sequences(self, sequences: List[str], decisions: Dict, 
                         fold_function=None) -> Dict:
        """Process sequences based on exit decisions."""
        
        folded_results = []
        skipped_results = []
        
        # Sort by priority for processing order
        sequence_priorities = list(zip(range(len(sequences)), decisions['priorities']))
        sequence_priorities.sort(key=lambda x: x[1], reverse=True)
        
        folded_count = 0
        skipped_count = 0
        
        for seq_idx, priority in sequence_priorities:
            sequence = sequences[seq_idx]
            should_fold = decisions['should_fold'][seq_idx]
            reason = decisions['exit_reasons'][seq_idx]
            
            if should_fold and fold_function is not None:
                # Fold the sequence
                try:
                    fold_result = fold_function(sequence)
                    folded_results.append({
                        'sequence_index': seq_idx,
                        'sequence': sequence,
                        'fold_result': fold_result,
                        'priority': priority
                    })
                    folded_count += 1
                except Exception as e:
                    logger.error(f"Folding failed for sequence {seq_idx}: {e}")
                    skipped_results.append({
                        'sequence_index': seq_idx,
                        'sequence': sequence,
                        'reason': 'folding_error',
                        'error': str(e)
                    })
                    skipped_count += 1
            else:
                # Skip the sequence
                skipped_results.append({
                    'sequence_index': seq_idx,
                    'sequence': sequence,
                    'reason': reason,
                    'priority': priority
                })
                skipped_count += 1
        
        return {
            'folded_results': folded_results,
            'skipped_results': skipped_results,
            'folded_count': folded_count,
            'skipped_count': skipped_count
        }
    
    def _update_statistics(self, decisions: Dict, results: Dict, batch_time: float):
        """Update processing statistics."""
        
        batch_size = len(decisions['should_fold'])
        folded_count = results['folded_count']
        skipped_count = results['skipped_count']
        
        self.stats['total_sequences'] += batch_size
        self.stats['folded_sequences'] += folded_count
        self.stats['skipped_sequences'] += skipped_count
        
        # Count exit reasons
        for reason in decisions['exit_reasons']:
            if reason == 'low_confidence':
                self.stats['confidence_exits'] += 1
            elif reason == 'low_tm_score':
                self.stats['tm_score_exits'] += 1
            elif reason == 'high_complexity':
                self.stats['complexity_exits'] += 1
        
        # Estimate time saved (rough approximation)
        avg_fold_time = 30.0  # Assume 30s per sequence
        time_saved = skipped_count * avg_fold_time
        self.stats['total_time_saved'] += time_saved
    
    def _mock_confidence_predictions(self, sequences: List[str]) -> Dict:
        """Mock confidence predictions when estimator not available."""
        
        batch_size = len(sequences)
        
        # Generate mock confidence based on sequence properties
        confidences = []
        for seq in sequences:
            # Simple heuristic: shorter sequences and balanced composition = higher confidence
            length_factor = min(1.0, 100.0 / len(seq))
            composition_factor = len(set(seq)) / 20.0  # Diversity factor
            confidence = 0.5 + 0.3 * length_factor + 0.2 * composition_factor
            confidence = min(1.0, confidence)
            confidences.append(confidence)
        
        return {
            'overall_confidence': torch.tensor(confidences),
            'batch_size': batch_size
        }
    
    def _mock_tm_predictions(self, sequences: List[str]) -> Dict:
        """Mock TM-score predictions when predictor not available."""
        
        batch_size = len(sequences)
        
        # Generate mock TM-scores based on sequence properties
        tm_scores = []
        for seq in sequences:
            # Simple heuristic based on length and composition
            length_factor = min(1.0, 200.0 / len(seq))  # Penalty for very long sequences
            hydrophobic_count = sum(1 for aa in seq if aa in 'AILMFPWYV')
            hydrophobic_factor = min(1.0, hydrophobic_count / len(seq) / 0.4)  # Optimal ~40%
            
            tm_score = 0.6 + 0.2 * length_factor + 0.2 * hydrophobic_factor
            tm_score = min(1.0, max(0.3, tm_score))
            tm_scores.append(tm_score)
        
        return {
            'tm_score_pred': torch.tensor(tm_scores),
            'batch_size': batch_size
        }
    
    def get_statistics_summary(self) -> Dict:
        """Get a summary of processing statistics."""
        
        total = self.stats['total_sequences']
        if total == 0:
            return {'message': 'No sequences processed yet'}
        
        folding_rate = self.stats['folded_sequences'] / total
        skip_rate = self.stats['skipped_sequences'] / total
        time_saved_per_sequence = self.stats['total_time_saved'] / max(total, 1)
        
        return {
            'total_sequences_processed': total,
            'folding_rate': folding_rate,
            'skip_rate': skip_rate,
            'time_saved_per_sequence': time_saved_per_sequence,
            'total_time_saved': self.stats['total_time_saved'],
            'exit_breakdown': {
                'confidence_exits': self.stats['confidence_exits'],
                'tm_score_exits': self.stats['tm_score_exits'],
                'complexity_exits': self.stats['complexity_exits']
            }
        }


class BatchRanker:
    """
    Ranks sequences in batches for optimal processing order.
    
    Uses confidence and TM-score predictions to prioritize sequences
    that are most likely to fold successfully.
    """
    
    def __init__(self, confidence_estimator=None, tm_score_predictor=None):
        self.confidence_estimator = confidence_estimator
        self.tm_score_predictor = tm_score_predictor
    
    def rank_sequences(self, sequences: List[str], 
                      ranking_strategy: str = "combined") -> List[Tuple[int, str, float]]:
        """
        Rank sequences by folding priority.
        
        Args:
            sequences: List of sequences to rank
            ranking_strategy: "confidence", "tm_score", or "combined"
            
        Returns:
            List of (index, sequence, score) tuples sorted by priority
        """
        
        if ranking_strategy == "confidence":
            return self._rank_by_confidence(sequences)
        elif ranking_strategy == "tm_score":
            return self._rank_by_tm_score(sequences)
        elif ranking_strategy == "combined":
            return self._rank_by_combined_score(sequences)
        else:
            raise ValueError(f"Unknown ranking strategy: {ranking_strategy}")
    
    def _rank_by_confidence(self, sequences: List[str]) -> List[Tuple[int, str, float]]:
        """Rank sequences by confidence score."""
        
        if self.confidence_estimator:
            results = self.confidence_estimator.estimate_confidence(sequences)
            scores = results['overall_confidence'].detach().numpy()
        else:
            # Mock confidence scores
            scores = np.random.uniform(0.3, 0.9, len(sequences))
        
        # Create ranking
        ranking = [(i, seq, float(score)) for i, (seq, score) in enumerate(zip(sequences, scores))]
        ranking.sort(key=lambda x: x[2], reverse=True)
        
        return ranking
    
    def _rank_by_tm_score(self, sequences: List[str]) -> List[Tuple[int, str, float]]:
        """Rank sequences by predicted TM-score."""
        
        if self.tm_score_predictor:
            results = self.tm_score_predictor.forward(sequences)
            scores = results['tm_score_pred'].detach().numpy()
        else:
            # Mock TM-scores
            scores = np.random.uniform(0.4, 0.9, len(sequences))
        
        # Create ranking
        ranking = [(i, seq, float(score)) for i, (seq, score) in enumerate(zip(sequences, scores))]
        ranking.sort(key=lambda x: x[2], reverse=True)
        
        return ranking
    
    def _rank_by_combined_score(self, sequences: List[str]) -> List[Tuple[int, str, float]]:
        """Rank sequences by combined confidence and TM-score."""
        
        # Get confidence scores
        if self.confidence_estimator:
            conf_results = self.confidence_estimator.estimate_confidence(sequences)
            confidence_scores = conf_results['overall_confidence'].detach().numpy()
        else:
            confidence_scores = np.random.uniform(0.3, 0.9, len(sequences))

        # Get TM-score predictions
        if self.tm_score_predictor:
            tm_results = self.tm_score_predictor.forward(sequences)
            tm_scores = tm_results['tm_score_pred'].detach().numpy()
        else:
            tm_scores = np.random.uniform(0.4, 0.9, len(sequences))
        
        # Combine scores (weighted average)
        combined_scores = 0.6 * confidence_scores + 0.4 * tm_scores
        
        # Create ranking
        ranking = [(i, seq, float(score)) for i, (seq, score) in enumerate(zip(sequences, combined_scores))]
        ranking.sort(key=lambda x: x[2], reverse=True)
        
        return ranking
