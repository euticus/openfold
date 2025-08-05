"""
ΔΔG Mutation Predictor for OdinFold

Predicts the change in free energy (ΔΔG) upon amino acid mutations
using protein structure and sequence information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import logging

logger = logging.getLogger(__name__)


class MutationEncoder(nn.Module):
    """
    Encodes mutation information for ΔΔG prediction.
    
    Combines wild-type and mutant amino acid information with
    structural context for accurate stability prediction.
    """
    
    def __init__(self, 
                 d_model: int = 256,
                 num_amino_acids: int = 20,
                 max_position: int = 2048):
        super().__init__()
        
        self.d_model = d_model
        self.num_amino_acids = num_amino_acids
        
        # Amino acid embeddings
        self.aa_embedding = nn.Embedding(num_amino_acids + 1, d_model // 4)  # +1 for unknown
        
        # Position embeddings
        self.position_embedding = nn.Embedding(max_position, d_model // 4)
        
        # Mutation type encoding
        self.mutation_type_proj = nn.Linear(2 * (d_model // 4), d_model // 2)
        
        # Context encoding
        self.context_proj = nn.Linear(d_model, d_model)
        
        # Final projection
        self.output_proj = nn.Linear(d_model + d_model // 2, d_model)
        
        # Amino acid properties (hydrophobicity, charge, etc.)
        self.aa_properties = nn.Parameter(torch.randn(num_amino_acids + 1, 8))
        self.property_proj = nn.Linear(16, d_model // 4)  # 2 * 8 properties
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        nn.init.normal_(self.aa_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.mutation_type_proj.weight, std=0.02)
        nn.init.zeros_(self.mutation_type_proj.bias)
        nn.init.normal_(self.context_proj.weight, std=0.02)
        nn.init.zeros_(self.context_proj.bias)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.normal_(self.aa_properties, std=0.02)
        nn.init.normal_(self.property_proj.weight, std=0.02)
        nn.init.zeros_(self.property_proj.bias)
    
    def forward(self,
                wt_aa: torch.Tensor,
                mut_aa: torch.Tensor,
                position: torch.Tensor,
                context_features: torch.Tensor) -> torch.Tensor:
        """
        Encode mutation information.
        
        Args:
            wt_aa: Wild-type amino acid indices [batch_size]
            mut_aa: Mutant amino acid indices [batch_size]
            position: Position indices [batch_size]
            context_features: Structural context features [batch_size, d_model]
            
        Returns:
            Encoded mutation features [batch_size, d_model]
        """
        
        batch_size = wt_aa.shape[0]
        
        # Amino acid embeddings
        wt_emb = self.aa_embedding(wt_aa)  # [batch_size, d_model//4]
        mut_emb = self.aa_embedding(mut_aa)  # [batch_size, d_model//4]
        
        # Position embeddings
        pos_emb = self.position_embedding(position)  # [batch_size, d_model//4]
        
        # Amino acid properties
        wt_props = self.aa_properties[wt_aa]  # [batch_size, 8]
        mut_props = self.aa_properties[mut_aa]  # [batch_size, 8]
        prop_diff = torch.cat([wt_props, mut_props], dim=-1)  # [batch_size, 16]
        prop_emb = self.property_proj(prop_diff)  # [batch_size, d_model//4]
        
        # Combine mutation information
        mutation_features = torch.cat([wt_emb, mut_emb], dim=-1)  # [batch_size, d_model//2]
        mutation_encoded = self.mutation_type_proj(mutation_features)  # [batch_size, d_model//2]
        
        # Process context
        context_encoded = self.context_proj(context_features)  # [batch_size, d_model]
        
        # Combine all features
        combined_features = torch.cat([context_encoded, mutation_encoded], dim=-1)
        output = self.output_proj(combined_features)
        
        return output


class DDGPredictionHead(nn.Module):
    """
    ΔΔG prediction head for protein mutations.
    
    Predicts the change in free energy upon mutation using
    structural and sequence context.
    """
    
    def __init__(self,
                 d_model: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Multi-layer regression head
        layers = []
        input_dim = d_model
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
            
            # Reduce hidden dimension in later layers
            if i < num_layers - 1:
                hidden_dim = hidden_dim // 2
        
        # Final prediction layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.regression_head = nn.Sequential(*layers)
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Positive uncertainty
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, mutation_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict ΔΔG from mutation features.
        
        Args:
            mutation_features: Encoded mutation features [batch_size, d_model]
            
        Returns:
            Dictionary with ΔΔG prediction and uncertainty
        """
        
        # Predict ΔΔG
        ddg_pred = self.regression_head(mutation_features).squeeze(-1)  # [batch_size]
        
        # Predict uncertainty
        uncertainty = self.uncertainty_head(mutation_features).squeeze(-1)  # [batch_size]
        
        return {
            'ddg_pred': ddg_pred,
            'uncertainty': uncertainty,
            'confidence': 1.0 / (1.0 + uncertainty)
        }


class DDGPredictor(nn.Module):
    """
    Complete ΔΔG predictor combining mutation encoding and prediction.
    
    Integrates with OdinFold's structure module to predict mutation effects
    on protein stability.
    """
    
    def __init__(self,
                 structure_dim: int = 384,
                 d_model: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.structure_dim = structure_dim
        self.d_model = d_model
        
        # Structure feature projection
        self.structure_proj = nn.Linear(structure_dim, d_model)
        
        # Mutation encoder
        self.mutation_encoder = MutationEncoder(d_model)
        
        # ΔΔG prediction head
        self.ddg_head = DDGPredictionHead(d_model, hidden_dim, num_layers, dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Training statistics for normalization
        self.register_buffer('ddg_mean', torch.tensor(0.0))
        self.register_buffer('ddg_std', torch.tensor(1.0))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        nn.init.normal_(self.structure_proj.weight, std=0.02)
        nn.init.zeros_(self.structure_proj.bias)
    
    def forward(self,
                structure_features: torch.Tensor,
                wt_aa: torch.Tensor,
                mut_aa: torch.Tensor,
                position: torch.Tensor,
                normalize: bool = True) -> Dict[str, torch.Tensor]:
        """
        Predict ΔΔG for mutations.
        
        Args:
            structure_features: Structure features from OdinFold [batch_size, seq_len, structure_dim]
            wt_aa: Wild-type amino acid indices [batch_size]
            mut_aa: Mutant amino acid indices [batch_size]
            position: Mutation positions [batch_size]
            normalize: Whether to normalize predictions
            
        Returns:
            Dictionary with ΔΔG predictions and metadata
        """
        
        batch_size = wt_aa.shape[0]
        
        # Extract features at mutation positions
        position_clamped = torch.clamp(position, 0, structure_features.shape[1] - 1)
        context_features = structure_features[torch.arange(batch_size), position_clamped]
        
        # Project structure features
        context_projected = self.structure_proj(context_features)
        context_projected = self.layer_norm(context_projected)
        
        # Encode mutation
        mutation_features = self.mutation_encoder(
            wt_aa, mut_aa, position, context_projected
        )
        
        # Predict ΔΔG
        predictions = self.ddg_head(mutation_features)
        
        # Normalize predictions if requested
        if normalize:
            predictions['ddg_pred'] = (predictions['ddg_pred'] * self.ddg_std) + self.ddg_mean
        
        # Add input information
        predictions.update({
            'wt_aa': wt_aa,
            'mut_aa': mut_aa,
            'position': position,
            'batch_size': batch_size
        })
        
        return predictions
    
    def predict_single(self,
                      structure_features: torch.Tensor,
                      wt_aa: int,
                      mut_aa: int,
                      position: int) -> Dict[str, float]:
        """
        Predict ΔΔG for a single mutation.
        
        Args:
            structure_features: Structure features [seq_len, structure_dim]
            wt_aa: Wild-type amino acid index
            mut_aa: Mutant amino acid index
            position: Mutation position
            
        Returns:
            Dictionary with ΔΔG prediction
        """
        
        # Add batch dimension
        structure_features = structure_features.unsqueeze(0)
        wt_aa_tensor = torch.tensor([wt_aa], device=structure_features.device)
        mut_aa_tensor = torch.tensor([mut_aa], device=structure_features.device)
        position_tensor = torch.tensor([position], device=structure_features.device)
        
        with torch.no_grad():
            predictions = self.forward(
                structure_features, wt_aa_tensor, mut_aa_tensor, position_tensor
            )
        
        return {
            'ddg_pred': float(predictions['ddg_pred'][0]),
            'uncertainty': float(predictions['uncertainty'][0]),
            'confidence': float(predictions['confidence'][0])
        }
    
    def scan_mutations(self,
                      structure_features: torch.Tensor,
                      wt_sequence: List[int],
                      positions: Optional[List[int]] = None,
                      target_aa: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        Scan multiple mutations across positions.
        
        Args:
            structure_features: Structure features [seq_len, structure_dim]
            wt_sequence: Wild-type sequence as amino acid indices
            positions: Positions to mutate (default: all)
            target_aa: Target amino acids (default: all 20)
            
        Returns:
            Dictionary with mutation scan results
        """
        
        seq_len = len(wt_sequence)
        
        if positions is None:
            positions = list(range(seq_len))
        
        if target_aa is None:
            target_aa = list(range(20))  # All 20 amino acids
        
        # Generate all mutation combinations
        mutations = []
        for pos in positions:
            wt_aa = wt_sequence[pos]
            for mut_aa in target_aa:
                if mut_aa != wt_aa:  # Skip identity mutations
                    mutations.append((wt_aa, mut_aa, pos))
        
        if not mutations:
            return {'ddg_predictions': torch.tensor([]), 'mutations': []}
        
        # Batch process mutations
        batch_size = len(mutations)
        wt_aa_batch = torch.tensor([m[0] for m in mutations], device=structure_features.device)
        mut_aa_batch = torch.tensor([m[1] for m in mutations], device=structure_features.device)
        position_batch = torch.tensor([m[2] for m in mutations], device=structure_features.device)
        
        # Expand structure features
        structure_batch = structure_features.unsqueeze(0).expand(batch_size, -1, -1)
        
        with torch.no_grad():
            predictions = self.forward(
                structure_batch, wt_aa_batch, mut_aa_batch, position_batch
            )
        
        return {
            'ddg_predictions': predictions['ddg_pred'],
            'uncertainties': predictions['uncertainty'],
            'confidences': predictions['confidence'],
            'mutations': mutations,
            'num_mutations': batch_size
        }
    
    def update_normalization_stats(self, ddg_values: torch.Tensor):
        """Update normalization statistics from training data."""
        
        self.ddg_mean.data = ddg_values.mean()
        self.ddg_std.data = ddg_values.std()
        
        logger.info(f"Updated ΔΔG normalization: mean={self.ddg_mean:.3f}, std={self.ddg_std:.3f}")


def amino_acid_to_index(aa: str) -> int:
    """Convert single-letter amino acid code to index."""
    aa_map = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
        'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
        'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
    }
    return aa_map.get(aa.upper(), 20)  # 20 for unknown


def index_to_amino_acid(idx: int) -> str:
    """Convert amino acid index to single-letter code."""
    aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    return aa_list[idx] if 0 <= idx < 20 else 'X'


def create_ddg_loss(reduction: str = 'mean') -> nn.Module:
    """
    Create loss function for ΔΔG prediction training.
    
    Args:
        reduction: Loss reduction method
        
    Returns:
        Loss function
    """
    
    class DDGLoss(nn.Module):
        def __init__(self, reduction='mean'):
            super().__init__()
            self.reduction = reduction
            self.mse_loss = nn.MSELoss(reduction='none')
        
        def forward(self, predictions: Dict[str, torch.Tensor], 
                   targets: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Compute ΔΔG prediction loss.
            
            Args:
                predictions: Model predictions
                targets: Target ΔΔG values
                
            Returns:
                Loss dictionary
            """
            
            ddg_pred = predictions['ddg_pred']
            uncertainty = predictions['uncertainty']
            
            # MSE loss
            mse_loss = self.mse_loss(ddg_pred, targets)
            
            # Uncertainty-weighted loss
            weighted_loss = mse_loss / (uncertainty + 1e-8) + torch.log(uncertainty + 1e-8)
            
            # Reduce losses
            if self.reduction == 'mean':
                mse_loss = mse_loss.mean()
                weighted_loss = weighted_loss.mean()
            elif self.reduction == 'sum':
                mse_loss = mse_loss.sum()
                weighted_loss = weighted_loss.sum()
            
            return {
                'total_loss': weighted_loss,
                'mse_loss': mse_loss,
                'uncertainty_loss': weighted_loss - mse_loss,
                'mae': torch.abs(ddg_pred - targets).mean()
            }
    
    return DDGLoss(reduction)
