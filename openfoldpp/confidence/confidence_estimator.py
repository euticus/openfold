"""
Confidence Estimator for OdinFold

Comprehensive confidence estimation combining sequence complexity analysis,
TM-score prediction, and uncertainty quantification.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class SequenceComplexityAnalyzer:
    """
    Analyzes sequence complexity to predict folding difficulty.
    
    Combines multiple sequence-based features to estimate
    how challenging a sequence will be to fold accurately.
    """
    
    def __init__(self):
        # Amino acid properties
        self.hydrophobic_aas = set('AILMFPWYV')
        self.polar_aas = set('STNQH')
        self.charged_aas = set('KRHDE')
        self.aromatic_aas = set('FWY')
        self.small_aas = set('AGCS')
        
        # Secondary structure propensities (simplified)
        self.helix_propensity = {'A': 1.42, 'E': 1.51, 'L': 1.21, 'M': 1.45, 'Q': 1.11, 'K': 1.16, 'R': 0.98}
        self.sheet_propensity = {'V': 1.70, 'I': 1.60, 'Y': 1.47, 'F': 1.38, 'W': 1.37, 'L': 1.30, 'T': 1.19}
        self.loop_propensity = {'G': 0.57, 'P': 0.59, 'S': 0.77, 'D': 1.01, 'N': 0.67, 'K': 1.01}
    
    def analyze_sequence(self, sequence: str) -> Dict[str, float]:
        """
        Comprehensive sequence complexity analysis.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary with complexity metrics
        """
        
        sequence = sequence.upper()
        length = len(sequence)
        
        if length == 0:
            return self._empty_analysis()
        
        # Basic composition
        composition = self._analyze_composition(sequence)
        
        # Sequence patterns
        patterns = self._analyze_patterns(sequence)
        
        # Secondary structure propensity
        ss_propensity = self._analyze_ss_propensity(sequence)
        
        # Disorder prediction (simplified)
        disorder = self._predict_disorder(sequence)
        
        # Complexity scores
        complexity = self._calculate_complexity_scores(sequence, composition, patterns)
        
        return {
            **composition,
            **patterns,
            **ss_propensity,
            **disorder,
            **complexity,
            'sequence_length': length
        }
    
    def _analyze_composition(self, sequence: str) -> Dict[str, float]:
        """Analyze amino acid composition."""
        
        length = len(sequence)
        aa_counts = Counter(sequence)
        
        # Basic composition
        hydrophobic_frac = sum(aa_counts[aa] for aa in self.hydrophobic_aas) / length
        polar_frac = sum(aa_counts[aa] for aa in self.polar_aas) / length
        charged_frac = sum(aa_counts[aa] for aa in self.charged_aas) / length
        aromatic_frac = sum(aa_counts[aa] for aa in self.aromatic_aas) / length
        small_frac = sum(aa_counts[aa] for aa in self.small_aas) / length
        
        # Special amino acids
        proline_frac = aa_counts['P'] / length
        glycine_frac = aa_counts['G'] / length
        cysteine_frac = aa_counts['C'] / length
        
        return {
            'hydrophobic_fraction': hydrophobic_frac,
            'polar_fraction': polar_frac,
            'charged_fraction': charged_frac,
            'aromatic_fraction': aromatic_frac,
            'small_fraction': small_frac,
            'proline_fraction': proline_frac,
            'glycine_fraction': glycine_frac,
            'cysteine_fraction': cysteine_frac
        }
    
    def _analyze_patterns(self, sequence: str) -> Dict[str, float]:
        """Analyze sequence patterns and repeats."""
        
        length = len(sequence)
        
        # Low complexity regions
        low_complexity_score = self._calculate_low_complexity(sequence)
        
        # Repeat content
        repeat_score = self._calculate_repeat_content(sequence)
        
        # Charge clusters
        charge_clustering = self._calculate_charge_clustering(sequence)
        
        # Hydrophobic clusters
        hydrophobic_clustering = self._calculate_hydrophobic_clustering(sequence)
        
        return {
            'low_complexity_score': low_complexity_score,
            'repeat_content': repeat_score,
            'charge_clustering': charge_clustering,
            'hydrophobic_clustering': hydrophobic_clustering
        }
    
    def _analyze_ss_propensity(self, sequence: str) -> Dict[str, float]:
        """Analyze secondary structure propensity."""
        
        length = len(sequence)
        
        helix_score = sum(self.helix_propensity.get(aa, 1.0) for aa in sequence) / length
        sheet_score = sum(self.sheet_propensity.get(aa, 1.0) for aa in sequence) / length
        loop_score = sum(self.loop_propensity.get(aa, 1.0) for aa in sequence) / length
        
        return {
            'helix_propensity': helix_score,
            'sheet_propensity': sheet_score,
            'loop_propensity': loop_score
        }
    
    def _predict_disorder(self, sequence: str) -> Dict[str, float]:
        """Simple disorder prediction based on composition."""
        
        # Disorder-promoting amino acids
        disorder_promoting = set('AGPQSEKR')
        disorder_score = sum(1 for aa in sequence if aa in disorder_promoting) / len(sequence)
        
        # Order-promoting amino acids
        order_promoting = set('WFYILVMC')
        order_score = sum(1 for aa in sequence if aa in order_promoting) / len(sequence)
        
        return {
            'disorder_score': disorder_score,
            'order_score': order_score,
            'disorder_tendency': disorder_score - order_score
        }
    
    def _calculate_complexity_scores(self, sequence: str, composition: Dict, patterns: Dict) -> Dict[str, float]:
        """Calculate overall complexity scores."""
        
        # Shannon entropy
        aa_counts = Counter(sequence)
        probs = np.array(list(aa_counts.values())) / len(sequence)
        shannon_entropy = -np.sum(probs * np.log2(probs + 1e-8))
        
        # Normalized entropy (0-1 scale)
        max_entropy = np.log2(20)  # Maximum for 20 amino acids
        normalized_entropy = shannon_entropy / max_entropy
        
        # Overall complexity score
        complexity_score = (
            normalized_entropy * 0.3 +
            (1 - patterns['low_complexity_score']) * 0.2 +
            (1 - patterns['repeat_content']) * 0.2 +
            composition['aromatic_fraction'] * 0.1 +
            composition['charged_fraction'] * 0.1 +
            abs(composition['hydrophobic_fraction'] - 0.4) * 0.1  # Optimal around 40%
        )
        
        return {
            'shannon_entropy': shannon_entropy,
            'normalized_entropy': normalized_entropy,
            'overall_complexity': complexity_score
        }
    
    def _calculate_low_complexity(self, sequence: str) -> float:
        """Calculate low complexity score using sliding window."""
        
        window_size = min(20, len(sequence) // 4)
        if window_size < 3:
            return 0.0
        
        low_complexity_regions = 0
        total_windows = len(sequence) - window_size + 1
        
        for i in range(total_windows):
            window = sequence[i:i + window_size]
            aa_counts = Counter(window)
            
            # Check if dominated by few amino acids
            max_count = max(aa_counts.values())
            if max_count / window_size > 0.6:  # >60% single amino acid
                low_complexity_regions += 1
        
        return low_complexity_regions / max(total_windows, 1)
    
    def _calculate_repeat_content(self, sequence: str) -> float:
        """Calculate repeat content score."""
        
        repeat_score = 0.0
        length = len(sequence)
        
        # Check for short repeats (2-5 amino acids)
        for repeat_len in range(2, min(6, length // 2)):
            for i in range(length - repeat_len * 2 + 1):
                motif = sequence[i:i + repeat_len]
                
                # Count consecutive repeats
                repeats = 1
                pos = i + repeat_len
                while pos + repeat_len <= length and sequence[pos:pos + repeat_len] == motif:
                    repeats += 1
                    pos += repeat_len
                
                if repeats >= 3:  # At least 3 consecutive repeats
                    repeat_score += (repeats * repeat_len) / length
        
        return min(repeat_score, 1.0)
    
    def _calculate_charge_clustering(self, sequence: str) -> float:
        """Calculate charge clustering score."""
        
        # Convert to charge sequence
        charge_seq = []
        for aa in sequence:
            if aa in 'KRH':
                charge_seq.append(1)  # Positive
            elif aa in 'DE':
                charge_seq.append(-1)  # Negative
            else:
                charge_seq.append(0)  # Neutral
        
        # Calculate clustering using autocorrelation
        clustering_score = 0.0
        if len(charge_seq) > 2:
            for lag in range(1, min(10, len(charge_seq))):
                if len(charge_seq) > lag:
                    try:
                        correlation = np.corrcoef(charge_seq[:-lag], charge_seq[lag:])[0, 1]
                        if not np.isnan(correlation):
                            clustering_score += abs(correlation) / lag
                    except:
                        # Skip if correlation calculation fails
                        continue

        return clustering_score
    
    def _calculate_hydrophobic_clustering(self, sequence: str) -> float:
        """Calculate hydrophobic clustering score."""
        
        # Convert to hydrophobicity sequence
        hydrophobic_seq = [1 if aa in self.hydrophobic_aas else 0 for aa in sequence]
        
        # Calculate clustering using runs
        runs = []
        current_run = 0
        
        for is_hydrophobic in hydrophobic_seq:
            if is_hydrophobic:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        
        if current_run > 0:
            runs.append(current_run)
        
        # Score based on long hydrophobic runs
        clustering_score = sum(run ** 2 for run in runs if run >= 3) / len(sequence)
        
        return min(clustering_score, 1.0)
    
    def _empty_analysis(self) -> Dict[str, float]:
        """Return empty analysis for invalid sequences."""
        
        return {key: 0.0 for key in [
            'hydrophobic_fraction', 'polar_fraction', 'charged_fraction',
            'aromatic_fraction', 'small_fraction', 'proline_fraction',
            'glycine_fraction', 'cysteine_fraction', 'low_complexity_score',
            'repeat_content', 'charge_clustering', 'hydrophobic_clustering',
            'helix_propensity', 'sheet_propensity', 'loop_propensity',
            'disorder_score', 'order_score', 'disorder_tendency',
            'shannon_entropy', 'normalized_entropy', 'overall_complexity',
            'sequence_length'
        ]}


class ConfidenceEstimator:
    """
    Comprehensive confidence estimator combining multiple sources of information.
    
    Integrates TM-score prediction, sequence complexity analysis,
    and uncertainty quantification for robust confidence estimation.
    """
    
    def __init__(self, tm_score_predictor=None):
        self.tm_score_predictor = tm_score_predictor
        self.complexity_analyzer = SequenceComplexityAnalyzer()
        
        # Confidence calibration parameters
        self.complexity_weight = 0.3
        self.tm_score_weight = 0.5
        self.uncertainty_weight = 0.2
    
    def estimate_confidence(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """
        Estimate confidence for a batch of sequences.
        
        Args:
            sequences: List of amino acid sequences
            
        Returns:
            Dictionary with confidence estimates and components
        """
        
        batch_size = len(sequences)
        
        # Analyze sequence complexity
        complexity_scores = []
        for seq in sequences:
            analysis = self.complexity_analyzer.analyze_sequence(seq)
            # Convert complexity to confidence (higher complexity = lower confidence)
            complexity_confidence = 1.0 - analysis['overall_complexity']
            complexity_scores.append(complexity_confidence)
        
        complexity_confidence = torch.tensor(complexity_scores)
        
        # Get TM-score predictions if available
        if self.tm_score_predictor is not None:
            tm_predictions = self.tm_score_predictor.forward(sequences)
            tm_confidence = tm_predictions['confidence']
            tm_scores = tm_predictions['tm_score_pred']
        else:
            # Default values if no predictor
            tm_confidence = torch.ones(batch_size) * 0.7
            tm_scores = torch.ones(batch_size) * 0.7
        
        # Combine confidence sources
        overall_confidence = (
            self.complexity_weight * complexity_confidence +
            self.tm_score_weight * tm_confidence +
            self.uncertainty_weight * tm_confidence  # Use TM confidence for uncertainty too
        )
        
        # Normalize to [0, 1]
        overall_confidence = torch.clamp(overall_confidence, 0.0, 1.0)
        
        # Categorize confidence levels
        confidence_categories = self._categorize_confidence(overall_confidence)
        
        return {
            'overall_confidence': overall_confidence,
            'complexity_confidence': complexity_confidence,
            'tm_confidence': tm_confidence,
            'predicted_tm_scores': tm_scores,
            'confidence_categories': confidence_categories,
            'batch_size': batch_size
        }
    
    def _categorize_confidence(self, confidence_scores: torch.Tensor) -> torch.Tensor:
        """Categorize confidence scores into levels."""
        
        # Confidence categories:
        # 0: Low (< 0.5)
        # 1: Medium (0.5 - 0.7)
        # 2: High (0.7 - 0.85)
        # 3: Very High (>= 0.85)
        
        categories = torch.zeros_like(confidence_scores, dtype=torch.long)
        categories[confidence_scores >= 0.5] = 1
        categories[confidence_scores >= 0.7] = 2
        categories[confidence_scores >= 0.85] = 3
        
        return categories
    
    def should_skip_folding(self, sequences: List[str], 
                          confidence_threshold: float = 0.3) -> List[bool]:
        """
        Determine which sequences should skip folding due to low confidence.
        
        Args:
            sequences: List of sequences to evaluate
            confidence_threshold: Minimum confidence for folding
            
        Returns:
            List of boolean flags indicating whether to skip folding
        """
        
        confidence_results = self.estimate_confidence(sequences)
        overall_confidence = confidence_results['overall_confidence']
        
        skip_flags = (overall_confidence < confidence_threshold).tolist()
        
        return skip_flags
    
    def get_folding_priority(self, sequences: List[str]) -> List[Tuple[int, float]]:
        """
        Get folding priority order based on confidence.
        
        Args:
            sequences: List of sequences to prioritize
            
        Returns:
            List of (index, confidence) tuples sorted by priority
        """
        
        confidence_results = self.estimate_confidence(sequences)
        overall_confidence = confidence_results['overall_confidence']
        
        # Create priority list (higher confidence = higher priority)
        priorities = []
        for i, conf in enumerate(overall_confidence):
            priorities.append((i, float(conf)))
        
        # Sort by confidence (descending)
        priorities.sort(key=lambda x: x[1], reverse=True)
        
        return priorities
