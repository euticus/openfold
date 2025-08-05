"""
TM-Score Predictor for OdinFold

Pre-fold TM-score prediction using ESM encoder features.
Enables batch ranking and early exits for computational efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class TMScorePredictionHead(nn.Module):
    """
    Regression head for predicting TM-score from ESM encoder features.
    
    Takes sequence-level embeddings and predicts expected TM-score
    before running the full folding pipeline.
    """
    
    def __init__(self, d_model: int = 1280, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        
        # Sequence-level feature aggregation
        self.sequence_pooling = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten()
        )
        
        # Additional sequence statistics
        self.length_embedding = nn.Embedding(2048, 64)  # Sequence length embedding
        self.complexity_encoder = nn.Linear(10, 64)  # Sequence complexity features
        
        # Regression head
        input_dim = d_model + 64 + 64  # ESM features + length + complexity
        self.regression_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # TM-score is in [0, 1]
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Positive uncertainty
        )
    
    def forward(self, esm_features: torch.Tensor, 
                sequence_length: torch.Tensor,
                sequence_complexity: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Predict TM-score from ESM features.
        
        Args:
            esm_features: ESM encoder features [batch_size, seq_len, d_model]
            sequence_length: Sequence lengths [batch_size]
            sequence_complexity: Optional complexity features [batch_size, 10]
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        
        batch_size, seq_len, d_model = esm_features.shape
        
        # Global sequence representation
        # Transpose for pooling: [batch_size, d_model, seq_len]
        esm_transposed = esm_features.transpose(1, 2)
        sequence_repr = self.sequence_pooling(esm_transposed)  # [batch_size, d_model]
        
        # Length embedding
        length_clamped = torch.clamp(sequence_length, 0, 2047)
        length_emb = self.length_embedding(length_clamped)  # [batch_size, 64]
        
        # Complexity features
        if sequence_complexity is None:
            # Generate default complexity features
            sequence_complexity = self._compute_default_complexity(esm_features)
        
        complexity_emb = self.complexity_encoder(sequence_complexity)  # [batch_size, 64]
        
        # Combine features
        combined_features = torch.cat([sequence_repr, length_emb, complexity_emb], dim=-1)
        
        # Predict TM-score
        tm_score_pred = self.regression_head(combined_features).squeeze(-1)  # [batch_size]
        
        # Predict uncertainty
        uncertainty = self.uncertainty_head(combined_features).squeeze(-1)  # [batch_size]
        
        return {
            'tm_score_pred': tm_score_pred,
            'uncertainty': uncertainty,
            'confidence': 1.0 / (1.0 + uncertainty),  # Convert uncertainty to confidence
            'sequence_repr': sequence_repr
        }
    
    def _compute_default_complexity(self, esm_features: torch.Tensor) -> torch.Tensor:
        """Compute default sequence complexity features."""
        
        batch_size, seq_len, d_model = esm_features.shape
        
        # Simple complexity metrics based on ESM features
        complexity_features = []
        
        # Feature variance (diversity)
        feature_var = esm_features.var(dim=1).mean(dim=-1)  # [batch_size]
        complexity_features.append(feature_var)
        
        # Feature entropy (approximate)
        feature_norm = F.normalize(esm_features, dim=-1)
        feature_entropy = -(feature_norm * torch.log(feature_norm + 1e-8)).sum(dim=-1).mean(dim=-1)
        complexity_features.append(feature_entropy)
        
        # Sequence length (normalized)
        length_norm = seq_len / 512.0  # Normalize by typical length
        complexity_features.append(torch.full((batch_size,), length_norm, device=esm_features.device))
        
        # Add more features to reach 10 dimensions
        for i in range(7):
            # Additional statistical features
            if i == 0:
                feat = esm_features.mean(dim=1).std(dim=-1)
            elif i == 1:
                feat = esm_features.std(dim=1).mean(dim=-1)
            elif i == 2:
                feat = esm_features.max(dim=1)[0].mean(dim=-1)
            elif i == 3:
                feat = esm_features.min(dim=1)[0].mean(dim=-1)
            elif i == 4:
                feat = (esm_features > 0).float().mean(dim=(1, 2))
            elif i == 5:
                feat = esm_features.abs().mean(dim=(1, 2))
            else:
                feat = torch.randn(batch_size, device=esm_features.device) * 0.1
            
            complexity_features.append(feat)
        
        return torch.stack(complexity_features, dim=-1)  # [batch_size, 10]


class TMScorePredictor(nn.Module):
    """
    Complete TM-score predictor that integrates with ESM encoder.
    
    Provides pre-fold TM-score estimation for batch ranking and early exits.
    """
    
    def __init__(self, esm_model, d_model: int = 1280, hidden_dim: int = 512):
        super().__init__()
        
        self.esm_model = esm_model
        self.prediction_head = TMScorePredictionHead(d_model, hidden_dim)
        
        # Training statistics for calibration
        self.register_buffer('tm_score_mean', torch.tensor(0.7))
        self.register_buffer('tm_score_std', torch.tensor(0.15))
        
        # Calibration parameters
        self.calibration_temperature = nn.Parameter(torch.tensor(1.0))
        self.calibration_bias = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, sequences: List[str], 
                return_esm_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Predict TM-scores for a batch of sequences.
        
        Args:
            sequences: List of amino acid sequences
            return_esm_features: Whether to return ESM features
            
        Returns:
            Dictionary with TM-score predictions and metadata
        """
        
        # Encode sequences with ESM
        esm_output = self._encode_sequences(sequences)
        esm_features = esm_output['representations']  # [batch_size, seq_len, d_model]
        
        # Get sequence lengths
        sequence_lengths = torch.tensor([len(seq) for seq in sequences], 
                                      device=esm_features.device)
        
        # Predict TM-scores
        predictions = self.prediction_head(esm_features, sequence_lengths)
        
        # Apply calibration
        calibrated_tm_scores = self._apply_calibration(predictions['tm_score_pred'])
        
        # Calculate quality categories
        quality_categories = self._categorize_quality(calibrated_tm_scores)
        
        results = {
            'tm_score_pred': calibrated_tm_scores,
            'raw_tm_score_pred': predictions['tm_score_pred'],
            'uncertainty': predictions['uncertainty'],
            'confidence': predictions['confidence'],
            'quality_category': quality_categories,
            'sequence_lengths': sequence_lengths,
            'batch_size': len(sequences)
        }
        
        if return_esm_features:
            results['esm_features'] = esm_features
            results['sequence_repr'] = predictions['sequence_repr']
        
        return results
    
    def _encode_sequences(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """Encode sequences using ESM model."""
        
        # Mock ESM encoding (replace with actual ESM model)
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)
        d_model = self.prediction_head.d_model
        
        # Generate mock ESM features
        esm_features = torch.randn(batch_size, max_len, d_model)
        
        # Add some sequence-dependent variation
        for i, seq in enumerate(sequences):
            seq_hash = hash(seq) % 1000
            esm_features[i] += torch.randn_like(esm_features[i]) * 0.1 * (seq_hash / 1000.0)
        
        return {
            'representations': esm_features,
            'tokens': None  # Not needed for this implementation
        }
    
    def _apply_calibration(self, raw_predictions: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling and bias correction for calibration."""
        
        # Temperature scaling
        calibrated = raw_predictions / self.calibration_temperature
        
        # Bias correction
        calibrated = calibrated + self.calibration_bias
        
        # Ensure valid range [0, 1]
        calibrated = torch.clamp(calibrated, 0.0, 1.0)
        
        return calibrated
    
    def _categorize_quality(self, tm_scores: torch.Tensor) -> torch.Tensor:
        """Categorize predicted TM-scores into quality levels."""
        
        # Quality categories:
        # 0: Poor (TM < 0.5)
        # 1: Fair (0.5 <= TM < 0.65)
        # 2: Good (0.65 <= TM < 0.8)
        # 3: Excellent (TM >= 0.8)
        
        categories = torch.zeros_like(tm_scores, dtype=torch.long)
        categories[tm_scores >= 0.5] = 1
        categories[tm_scores >= 0.65] = 2
        categories[tm_scores >= 0.8] = 3
        
        return categories
    
    def predict_single(self, sequence: str) -> Dict[str, float]:
        """Predict TM-score for a single sequence."""
        
        with torch.no_grad():
            results = self.forward([sequence])
            
            return {
                'tm_score_pred': float(results['tm_score_pred'][0]),
                'uncertainty': float(results['uncertainty'][0]),
                'confidence': float(results['confidence'][0]),
                'quality_category': int(results['quality_category'][0]),
                'sequence_length': int(results['sequence_lengths'][0])
            }
    
    def rank_sequences(self, sequences: List[str], 
                      top_k: Optional[int] = None) -> List[Tuple[int, str, float]]:
        """
        Rank sequences by predicted TM-score.
        
        Args:
            sequences: List of sequences to rank
            top_k: Optional number of top sequences to return
            
        Returns:
            List of (index, sequence, tm_score) tuples sorted by TM-score
        """
        
        with torch.no_grad():
            results = self.forward(sequences)
            tm_scores = results['tm_score_pred'].cpu().numpy()
            
            # Create ranking
            ranked_indices = np.argsort(tm_scores)[::-1]  # Descending order
            
            ranking = []
            for idx in ranked_indices:
                ranking.append((int(idx), sequences[idx], float(tm_scores[idx])))
            
            if top_k is not None:
                ranking = ranking[:top_k]
            
            return ranking
    
    def calibrate(self, sequences: List[str], true_tm_scores: torch.Tensor):
        """
        Calibrate the predictor using true TM-scores.
        
        Args:
            sequences: Calibration sequences
            true_tm_scores: True TM-scores for calibration
        """
        
        with torch.no_grad():
            # Get predictions
            results = self.forward(sequences)
            raw_predictions = results['raw_tm_score_pred']
            
            # Optimize calibration parameters
            from scipy.optimize import minimize_scalar
            
            def calibration_loss(temperature):
                calibrated = torch.clamp(raw_predictions / temperature + self.calibration_bias, 0, 1)
                mse = F.mse_loss(calibrated, true_tm_scores)
                return mse.item()
            
            # Optimize temperature
            temp_result = minimize_scalar(calibration_loss, bounds=(0.1, 10.0), method='bounded')
            self.calibration_temperature.data = torch.tensor(temp_result.x)
            
            # Optimize bias
            def bias_loss(bias):
                calibrated = torch.clamp(raw_predictions / self.calibration_temperature + bias, 0, 1)
                mse = F.mse_loss(calibrated, true_tm_scores)
                return mse.item()
            
            bias_result = minimize_scalar(bias_loss, bounds=(-1.0, 1.0), method='bounded')
            self.calibration_bias.data = torch.tensor(bias_result.x)
            
            logger.info(f"Calibration complete: temperature={self.calibration_temperature.item():.3f}, "
                       f"bias={self.calibration_bias.item():.3f}")
    
    def get_training_targets(self, sequences: List[str], 
                           true_tm_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get training targets for the TM-score predictor.
        
        Args:
            sequences: Training sequences
            true_tm_scores: True TM-scores
            
        Returns:
            Dictionary with training targets
        """
        
        # Get ESM features
        esm_output = self._encode_sequences(sequences)
        esm_features = esm_output['representations']
        
        sequence_lengths = torch.tensor([len(seq) for seq in sequences], 
                                      device=esm_features.device)
        
        return {
            'esm_features': esm_features,
            'sequence_lengths': sequence_lengths,
            'tm_score_targets': true_tm_scores,
            'complexity_features': self.prediction_head._compute_default_complexity(esm_features)
        }
