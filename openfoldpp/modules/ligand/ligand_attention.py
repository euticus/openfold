"""
Ligand-Protein Cross-Attention for OdinFold

Implements cross-attention mechanisms between protein residues and ligand atoms
for ligand-aware protein structure prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import logging

logger = logging.getLogger(__name__)


class LigandProteinCrossAttention(nn.Module):
    """
    Cross-attention between protein residues and ligand atoms.
    
    Allows protein residues to attend to ligand atoms and vice versa,
    enabling ligand-aware structure prediction.
    """
    
    def __init__(self,
                 protein_dim: int = 384,
                 ligand_dim: int = 128,
                 d_model: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_distance: float = 10.0):
        super().__init__()
        
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_distance = max_distance
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Project protein and ligand features to common dimension
        self.protein_proj = nn.Linear(protein_dim, d_model)
        self.ligand_proj = nn.Linear(ligand_dim, d_model)
        
        # Cross-attention projections (protein queries, ligand keys/values)
        self.protein_q_proj = nn.Linear(d_model, d_model)
        self.ligand_k_proj = nn.Linear(d_model, d_model)
        self.ligand_v_proj = nn.Linear(d_model, d_model)
        
        # Reverse cross-attention (ligand queries, protein keys/values)
        self.ligand_q_proj = nn.Linear(d_model, d_model)
        self.protein_k_proj = nn.Linear(d_model, d_model)
        self.protein_v_proj = nn.Linear(d_model, d_model)
        
        # Distance-based attention bias
        self.distance_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, num_heads)
        )
        
        # Output projections
        self.protein_out_proj = nn.Linear(d_model, protein_dim)
        self.ligand_out_proj = nn.Linear(d_model, ligand_dim)
        
        # Layer normalization
        self.protein_norm = nn.LayerNorm(protein_dim)
        self.ligand_norm = nn.LayerNorm(ligand_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self,
                protein_features: torch.Tensor,
                ligand_features: torch.Tensor,
                protein_coords: torch.Tensor,
                ligand_coords: torch.Tensor,
                protein_mask: Optional[torch.Tensor] = None,
                ligand_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Apply cross-attention between protein and ligand.
        
        Args:
            protein_features: Protein features [batch_size, seq_len, protein_dim]
            ligand_features: Ligand features [batch_size, num_atoms, ligand_dim]
            protein_coords: Protein coordinates [batch_size, seq_len, 3]
            ligand_coords: Ligand coordinates [batch_size, num_atoms, 3]
            protein_mask: Protein mask [batch_size, seq_len] (optional)
            ligand_mask: Ligand mask [batch_size, num_atoms] (optional)
            
        Returns:
            Dictionary with updated protein and ligand features
        """
        
        batch_size, seq_len, _ = protein_features.shape
        num_atoms = ligand_features.shape[1]
        
        # Project to common dimension
        protein_proj = self.protein_proj(protein_features)  # [batch, seq_len, d_model]
        ligand_proj = self.ligand_proj(ligand_features)  # [batch, num_atoms, d_model]
        
        # Compute pairwise distances
        distances = self._compute_distances(protein_coords, ligand_coords)  # [batch, seq_len, num_atoms]
        
        # Apply distance cutoff
        distance_mask = distances <= self.max_distance
        
        # Protein-to-ligand attention
        protein_updated = self._protein_to_ligand_attention(
            protein_proj, ligand_proj, distances, distance_mask, protein_mask, ligand_mask
        )
        
        # Ligand-to-protein attention
        ligand_updated = self._ligand_to_protein_attention(
            ligand_proj, protein_proj, distances, distance_mask, ligand_mask, protein_mask
        )
        
        # Project back to original dimensions and apply residual connections
        protein_out = self.protein_out_proj(protein_updated)
        ligand_out = self.ligand_out_proj(ligand_updated)
        
        protein_out = self.protein_norm(protein_features + protein_out)
        ligand_out = self.ligand_norm(ligand_features + ligand_out)
        
        return {
            'protein_features': protein_out,
            'ligand_features': ligand_out,
            'cross_attention_weights': distances,  # For visualization
            'interaction_mask': distance_mask
        }
    
    def _compute_distances(self,
                          protein_coords: torch.Tensor,
                          ligand_coords: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between protein and ligand atoms."""
        
        # protein_coords: [batch, seq_len, 3]
        # ligand_coords: [batch, num_atoms, 3]
        
        protein_expanded = protein_coords.unsqueeze(2)  # [batch, seq_len, 1, 3]
        ligand_expanded = ligand_coords.unsqueeze(1)    # [batch, 1, num_atoms, 3]
        
        distances = torch.norm(protein_expanded - ligand_expanded, dim=-1)  # [batch, seq_len, num_atoms]
        
        return distances
    
    def _protein_to_ligand_attention(self,
                                   protein_features: torch.Tensor,
                                   ligand_features: torch.Tensor,
                                   distances: torch.Tensor,
                                   distance_mask: torch.Tensor,
                                   protein_mask: Optional[torch.Tensor],
                                   ligand_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Protein residues attend to ligand atoms."""
        
        batch_size, seq_len, d_model = protein_features.shape
        num_atoms = ligand_features.shape[1]
        
        # Compute queries, keys, values
        q = self.protein_q_proj(protein_features)  # [batch, seq_len, d_model]
        k = self.ligand_k_proj(ligand_features)    # [batch, num_atoms, d_model]
        v = self.ligand_v_proj(ligand_features)    # [batch, num_atoms, d_model]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        # q: [batch, num_heads, seq_len, head_dim]
        # k, v: [batch, num_heads, num_atoms, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: [batch, num_heads, seq_len, num_atoms]
        
        # Add distance bias
        distance_bias = self.distance_proj(distances.unsqueeze(-1))  # [batch, seq_len, num_atoms, num_heads]
        distance_bias = distance_bias.permute(0, 3, 1, 2)  # [batch, num_heads, seq_len, num_atoms]
        scores = scores + distance_bias
        
        # Apply masks
        if distance_mask is not None:
            mask = distance_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        if ligand_mask is not None:
            ligand_mask_expanded = ligand_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, seq_len, -1)
            scores = scores.masked_fill(~ligand_mask_expanded, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return attended
    
    def _ligand_to_protein_attention(self,
                                   ligand_features: torch.Tensor,
                                   protein_features: torch.Tensor,
                                   distances: torch.Tensor,
                                   distance_mask: torch.Tensor,
                                   ligand_mask: Optional[torch.Tensor],
                                   protein_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Ligand atoms attend to protein residues."""
        
        batch_size, num_atoms, d_model = ligand_features.shape
        seq_len = protein_features.shape[1]
        
        # Compute queries, keys, values
        q = self.ligand_q_proj(ligand_features)    # [batch, num_atoms, d_model]
        k = self.protein_k_proj(protein_features)  # [batch, seq_len, d_model]
        v = self.protein_v_proj(protein_features)  # [batch, seq_len, d_model]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, num_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # q: [batch, num_heads, num_atoms, head_dim]
        # k, v: [batch, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: [batch, num_heads, num_atoms, seq_len]
        
        # Add distance bias (transpose distances for ligand-to-protein)
        distance_bias = self.distance_proj(distances.transpose(-2, -1).unsqueeze(-1))  # [batch, num_atoms, seq_len, num_heads]
        distance_bias = distance_bias.permute(0, 3, 1, 2)  # [batch, num_heads, num_atoms, seq_len]
        scores = scores + distance_bias
        
        # Apply masks
        if distance_mask is not None:
            mask = distance_mask.transpose(-2, -1).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        if protein_mask is not None:
            protein_mask_expanded = protein_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, num_atoms, -1)
            scores = scores.masked_fill(~protein_mask_expanded, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, v)  # [batch, num_heads, num_atoms, head_dim]
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_atoms, d_model)
        
        return attended


class BindingPocketAttention(nn.Module):
    """
    Specialized attention for binding pocket prediction.
    
    Focuses on protein residues that are likely to interact
    with ligands based on distance and chemical properties.
    """
    
    def __init__(self,
                 d_model: int = 256,
                 num_heads: int = 8,
                 pocket_radius: float = 8.0,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.pocket_radius = pocket_radius
        
        # Binding pocket prediction head
        self.pocket_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Pocket-aware attention
        self.pocket_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.pocket_predictor:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
    
    def forward(self,
                protein_features: torch.Tensor,
                ligand_coords: torch.Tensor,
                protein_coords: torch.Tensor,
                protein_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Apply binding pocket attention.
        
        Args:
            protein_features: Protein features [batch_size, seq_len, d_model]
            ligand_coords: Ligand coordinates [batch_size, num_atoms, 3]
            protein_coords: Protein coordinates [batch_size, seq_len, 3]
            protein_mask: Protein mask [batch_size, seq_len] (optional)
            
        Returns:
            Dictionary with pocket predictions and updated features
        """
        
        batch_size, seq_len, _ = protein_features.shape
        
        # Predict binding pocket residues
        pocket_scores = self.pocket_predictor(protein_features)  # [batch, seq_len, 1]
        pocket_scores = pocket_scores.squeeze(-1)  # [batch, seq_len]
        
        # Compute distance-based pocket mask
        if ligand_coords is not None:
            distances = torch.cdist(protein_coords, ligand_coords)  # [batch, seq_len, num_atoms]
            min_distances = distances.min(dim=-1)[0]  # [batch, seq_len]
            distance_mask = min_distances <= self.pocket_radius
        else:
            distance_mask = torch.ones_like(pocket_scores, dtype=torch.bool)
        
        # Combine predicted and distance-based masks
        pocket_mask = (pocket_scores > 0.5) & distance_mask
        
        # Apply pocket-aware self-attention
        if protein_mask is not None:
            attention_mask = protein_mask & pocket_mask
        else:
            attention_mask = pocket_mask
        
        # Convert mask for attention (True = attend, False = ignore)
        attn_mask = ~attention_mask  # Invert for attention mask
        
        attended_features, attention_weights = self.pocket_attention(
            protein_features, protein_features, protein_features,
            key_padding_mask=attn_mask
        )
        
        # Residual connection and layer norm
        output_features = self.layer_norm(protein_features + attended_features)
        
        return {
            'protein_features': output_features,
            'pocket_scores': pocket_scores,
            'pocket_mask': pocket_mask,
            'attention_weights': attention_weights,
            'distance_mask': distance_mask
        }


class LigandAwareFoldingHead(nn.Module):
    """
    Ligand-aware folding head that incorporates ligand information
    into protein structure prediction.
    """

    def __init__(self,
                 protein_dim: int = 384,
                 ligand_dim: int = 128,
                 d_model: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.d_model = d_model

        # Cross-attention between protein and ligand
        self.cross_attention = LigandProteinCrossAttention(
            protein_dim, ligand_dim, d_model, num_heads, dropout
        )

        # Binding pocket attention
        self.pocket_attention = BindingPocketAttention(
            d_model, num_heads, dropout=dropout
        )

        # Ligand-conditioned structure prediction
        self.structure_head = nn.Sequential(
            nn.Linear(protein_dim + ligand_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)  # xyz coordinates
        )

        # Confidence prediction with ligand awareness
        self.confidence_head = nn.Sequential(
            nn.Linear(protein_dim + ligand_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in [self.structure_head, self.confidence_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    nn.init.zeros_(layer.bias)

    def forward(self,
                protein_features: torch.Tensor,
                ligand_features: torch.Tensor,
                protein_coords: torch.Tensor,
                ligand_coords: torch.Tensor,
                protein_mask: Optional[torch.Tensor] = None,
                ligand_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Predict protein structure with ligand awareness.

        Args:
            protein_features: Protein features [batch_size, seq_len, protein_dim]
            ligand_features: Ligand features [batch_size, num_atoms, ligand_dim]
            protein_coords: Initial protein coordinates [batch_size, seq_len, 3]
            ligand_coords: Ligand coordinates [batch_size, num_atoms, 3]
            protein_mask: Protein mask [batch_size, seq_len] (optional)
            ligand_mask: Ligand mask [batch_size, num_atoms] (optional)

        Returns:
            Dictionary with structure predictions and metadata
        """

        batch_size, seq_len, _ = protein_features.shape
        num_atoms = ligand_features.shape[1]

        # Apply cross-attention between protein and ligand
        cross_attn_output = self.cross_attention(
            protein_features, ligand_features,
            protein_coords, ligand_coords,
            protein_mask, ligand_mask
        )

        updated_protein_features = cross_attn_output['protein_features']
        updated_ligand_features = cross_attn_output['ligand_features']

        # Project protein features to d_model for pocket attention
        protein_proj = nn.Linear(self.protein_dim, self.d_model).to(updated_protein_features.device)
        protein_for_pocket = protein_proj(updated_protein_features)

        # Apply binding pocket attention
        pocket_output = self.pocket_attention(
            protein_for_pocket, ligand_coords, protein_coords, protein_mask
        )

        # Global ligand context (mean pooling)
        if ligand_mask is not None:
            ligand_mask_expanded = ligand_mask.unsqueeze(-1).expand_as(updated_ligand_features)
            masked_ligand_features = updated_ligand_features * ligand_mask_expanded
            ligand_context = masked_ligand_features.sum(dim=1) / ligand_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            ligand_context = updated_ligand_features.mean(dim=1)  # [batch_size, ligand_dim]

        # Broadcast ligand context to all residues
        ligand_context_broadcast = ligand_context.unsqueeze(1).expand(-1, seq_len, -1)

        # Combine protein and ligand features
        combined_features = torch.cat([updated_protein_features, ligand_context_broadcast], dim=-1)

        # Predict structure
        coord_updates = self.structure_head(combined_features)  # [batch, seq_len, 3]
        updated_coords = protein_coords + coord_updates

        # Predict confidence
        confidence_scores = self.confidence_head(combined_features)  # [batch, seq_len, 1]
        confidence_scores = confidence_scores.squeeze(-1)  # [batch, seq_len]

        return {
            'coordinates': updated_coords,
            'confidence': confidence_scores,
            'pocket_scores': pocket_output['pocket_scores'],
            'pocket_mask': pocket_output['pocket_mask'],
            'cross_attention_weights': cross_attn_output['cross_attention_weights'],
            'interaction_mask': cross_attn_output['interaction_mask'],
            'ligand_context': ligand_context,
            'coord_updates': coord_updates
        }


class LigandConditionedStructureModule(nn.Module):
    """
    Complete ligand-conditioned structure module that integrates
    ligand awareness into the entire folding process.
    """

    def __init__(self,
                 protein_dim: int = 384,
                 ligand_dim: int = 128,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()

        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.d_model = d_model
        self.num_layers = num_layers

        # Multiple ligand-aware folding layers
        self.folding_layers = nn.ModuleList([
            LigandAwareFoldingHead(protein_dim, ligand_dim, d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(protein_dim) for _ in range(num_layers)
        ])

        # Final structure refinement
        self.final_refinement = nn.Sequential(
            nn.Linear(protein_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)
        )

        # Final confidence prediction
        self.final_confidence = nn.Sequential(
            nn.Linear(protein_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in [self.final_refinement, self.final_confidence]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    nn.init.zeros_(layer.bias)

    def forward(self,
                protein_features: torch.Tensor,
                ligand_features: torch.Tensor,
                initial_coords: torch.Tensor,
                ligand_coords: torch.Tensor,
                protein_mask: Optional[torch.Tensor] = None,
                ligand_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Apply ligand-conditioned structure prediction.

        Args:
            protein_features: Protein features [batch_size, seq_len, protein_dim]
            ligand_features: Ligand features [batch_size, num_atoms, ligand_dim]
            initial_coords: Initial coordinates [batch_size, seq_len, 3]
            ligand_coords: Ligand coordinates [batch_size, num_atoms, 3]
            protein_mask: Protein mask [batch_size, seq_len] (optional)
            ligand_mask: Ligand mask [batch_size, num_atoms] (optional)

        Returns:
            Dictionary with final structure predictions
        """

        current_coords = initial_coords
        current_features = protein_features

        all_pocket_scores = []
        all_confidence_scores = []
        coordinate_trajectory = [current_coords]

        # Apply multiple folding layers
        for i, (folding_layer, layer_norm) in enumerate(zip(self.folding_layers, self.layer_norms)):
            # Apply ligand-aware folding
            folding_output = folding_layer(
                current_features, ligand_features,
                current_coords, ligand_coords,
                protein_mask, ligand_mask
            )

            # Update coordinates and features
            current_coords = folding_output['coordinates']
            coordinate_trajectory.append(current_coords)

            # Update features with residual connection
            feature_updates = folding_output.get('feature_updates', torch.zeros_like(current_features))
            current_features = layer_norm(current_features + feature_updates)

            # Store intermediate predictions
            all_pocket_scores.append(folding_output['pocket_scores'])
            all_confidence_scores.append(folding_output['confidence'])

        # Final refinement
        final_coord_updates = self.final_refinement(current_features)
        final_coords = current_coords + final_coord_updates

        # Final confidence
        final_confidence = self.final_confidence(current_features).squeeze(-1)

        return {
            'final_coordinates': final_coords,
            'final_confidence': final_confidence,
            'coordinate_trajectory': coordinate_trajectory,
            'pocket_scores_trajectory': all_pocket_scores,
            'confidence_trajectory': all_confidence_scores,
            'final_coord_updates': final_coord_updates,
            'num_refinement_steps': self.num_layers
        }
