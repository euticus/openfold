"""
Ligand Cross-Attention for OdinFold

Implements cross-attention between protein residues and ligand atoms
for ligand-aware protein folding and binding pocket prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import math


class LigandCrossAttention(nn.Module):
    """
    Cross-attention mechanism between protein residues and ligand atoms.
    
    Allows protein residues to attend to ligand atoms for better
    binding pocket prediction and ligand-aware folding.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        # Protein query projection
        self.protein_q_proj = nn.Linear(d_model, d_model)
        
        # Ligand key and value projections
        self.ligand_k_proj = nn.Linear(d_model, d_model)
        self.ligand_v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Distance-based bias
        self.distance_bias = nn.Linear(1, num_heads)
        
        # Ligand type bias (different attention for different atom types)
        self.atom_type_bias = nn.Embedding(100, num_heads)  # Support 100 atom types
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, protein_repr: torch.Tensor, ligand_repr: torch.Tensor,
                protein_coords: Optional[torch.Tensor] = None,
                ligand_coords: Optional[torch.Tensor] = None,
                ligand_atom_types: Optional[torch.Tensor] = None,
                ligand_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Cross-attention from protein to ligand.
        
        Args:
            protein_repr: Protein representations [batch_size, seq_len, d_model]
            ligand_repr: Ligand atom representations [batch_size, num_atoms, d_model]
            protein_coords: Protein coordinates [batch_size, seq_len, 3]
            ligand_coords: Ligand coordinates [batch_size, num_atoms, 3]
            ligand_atom_types: Ligand atom types [batch_size, num_atoms]
            ligand_mask: Ligand atom mask [batch_size, num_atoms]
            
        Returns:
            Updated protein representations [batch_size, seq_len, d_model]
        """
        
        batch_size, seq_len, d_model = protein_repr.shape
        num_atoms = ligand_repr.shape[1]
        
        # Project to queries, keys, values
        q = self.protein_q_proj(protein_repr)  # [batch_size, seq_len, d_model]
        k = self.ligand_k_proj(ligand_repr)    # [batch_size, num_atoms, d_model]
        v = self.ligand_v_proj(ligand_repr)    # [batch_size, num_atoms, d_model]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # scores: [batch_size, num_heads, seq_len, num_atoms]
        
        # Add distance bias if coordinates are provided
        if protein_coords is not None and ligand_coords is not None:
            distances = self._compute_distances(protein_coords, ligand_coords)
            dist_bias = self.distance_bias(distances.unsqueeze(-1))  # [batch_size, seq_len, num_atoms, num_heads]
            dist_bias = dist_bias.permute(0, 3, 1, 2)  # [batch_size, num_heads, seq_len, num_atoms]
            scores = scores + dist_bias
        
        # Add atom type bias if provided
        if ligand_atom_types is not None:
            atom_bias = self.atom_type_bias(ligand_atom_types.long())  # [batch_size, num_atoms, num_heads]
            atom_bias = atom_bias.permute(0, 2, 1).unsqueeze(2)  # [batch_size, num_heads, 1, num_atoms]
            scores = scores + atom_bias
        
        # Apply ligand mask
        if ligand_mask is not None:
            mask = ~ligand_mask.bool().unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, num_atoms]
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out_proj(out)
        
        return out
    
    def _compute_distances(self, protein_coords: torch.Tensor, 
                          ligand_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between protein residues and ligand atoms.
        
        Args:
            protein_coords: [batch_size, seq_len, 3]
            ligand_coords: [batch_size, num_atoms, 3]
            
        Returns:
            Distance matrix [batch_size, seq_len, num_atoms]
        """
        
        # Expand dimensions for broadcasting
        protein_expanded = protein_coords.unsqueeze(2)  # [batch_size, seq_len, 1, 3]
        ligand_expanded = ligand_coords.unsqueeze(1)    # [batch_size, 1, num_atoms, 3]
        
        # Compute distances
        distances = torch.norm(protein_expanded - ligand_expanded, dim=-1)
        
        return distances


class ProteinLigandAttention(nn.Module):
    """
    Bidirectional attention between protein and ligand.
    
    Combines protein-to-ligand and ligand-to-protein attention
    for comprehensive protein-ligand interaction modeling.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Protein-to-ligand attention
        self.protein_to_ligand = LigandCrossAttention(d_model, num_heads, dropout)
        
        # Ligand-to-protein attention (reverse)
        self.ligand_to_protein = LigandCrossAttention(d_model, num_heads, dropout)
        
        # Layer norms
        self.protein_norm = nn.LayerNorm(d_model)
        self.ligand_norm = nn.LayerNorm(d_model)
        
        # Gating mechanisms
        self.protein_gate = nn.Linear(d_model * 2, d_model)
        self.ligand_gate = nn.Linear(d_model * 2, d_model)
    
    def forward(self, protein_repr: torch.Tensor, ligand_repr: torch.Tensor,
                protein_coords: Optional[torch.Tensor] = None,
                ligand_coords: Optional[torch.Tensor] = None,
                ligand_atom_types: Optional[torch.Tensor] = None,
                ligand_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional protein-ligand attention.
        
        Args:
            protein_repr: Protein representations [batch_size, seq_len, d_model]
            ligand_repr: Ligand representations [batch_size, num_atoms, d_model]
            protein_coords: Protein coordinates [batch_size, seq_len, 3]
            ligand_coords: Ligand coordinates [batch_size, num_atoms, 3]
            ligand_atom_types: Ligand atom types [batch_size, num_atoms]
            ligand_mask: Ligand mask [batch_size, num_atoms]
            
        Returns:
            Updated protein and ligand representations
        """
        
        # Store original representations
        protein_orig = protein_repr
        ligand_orig = ligand_repr
        
        # Protein-to-ligand attention
        protein_attended = self.protein_to_ligand(
            protein_repr, ligand_repr, protein_coords, ligand_coords,
            ligand_atom_types, ligand_mask
        )
        
        # Ligand-to-protein attention (swap roles)
        ligand_attended = self.ligand_to_protein(
            ligand_repr, protein_repr, ligand_coords, protein_coords,
            None, None  # No protein atom types or mask
        )
        
        # Gated combination
        protein_combined = torch.cat([protein_orig, protein_attended], dim=-1)
        protein_gate = torch.sigmoid(self.protein_gate(protein_combined))
        protein_out = protein_orig + protein_gate * protein_attended
        
        ligand_combined = torch.cat([ligand_orig, ligand_attended], dim=-1)
        ligand_gate = torch.sigmoid(self.ligand_gate(ligand_combined))
        ligand_out = ligand_orig + ligand_gate * ligand_attended
        
        # Layer normalization
        protein_out = self.protein_norm(protein_out)
        ligand_out = self.ligand_norm(ligand_out)
        
        return protein_out, ligand_out


class BindingPocketPredictor(nn.Module):
    """
    Predicts binding pocket residues based on protein-ligand attention.
    
    Uses attention weights and geometric features to identify
    residues likely to be involved in ligand binding.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        self.d_model = d_model
        
        # Binding pocket prediction head (input is d_model + d_model//4 + d_model//4)
        input_dim = d_model + d_model // 4 + d_model // 4
        self.pocket_predictor = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Distance-based features
        self.distance_encoder = nn.Linear(1, d_model // 4)
        
        # Attention-based features
        self.attention_encoder = nn.Linear(1, d_model // 4)
    
    def forward(self, protein_repr: torch.Tensor,
                attention_weights: torch.Tensor,
                protein_coords: torch.Tensor,
                ligand_coords: torch.Tensor,
                ligand_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Predict binding pocket residues.
        
        Args:
            protein_repr: Protein representations [batch_size, seq_len, d_model]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, num_atoms]
            protein_coords: Protein coordinates [batch_size, seq_len, 3]
            ligand_coords: Ligand coordinates [batch_size, num_atoms, 3]
            ligand_mask: Ligand mask [batch_size, num_atoms]
            
        Returns:
            Dictionary with binding pocket predictions
        """
        
        batch_size, seq_len, d_model = protein_repr.shape
        
        # Compute minimum distances to ligand
        distances = self._compute_min_distances(protein_coords, ligand_coords, ligand_mask)
        distance_features = self.distance_encoder(distances.unsqueeze(-1))
        
        # Compute attention-based features
        if ligand_mask is not None:
            # Mask attention weights
            mask = ligand_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, num_atoms]
            masked_weights = attention_weights * mask
            attention_scores = masked_weights.sum(dim=-1).mean(dim=1)  # [batch_size, seq_len]
        else:
            attention_scores = attention_weights.sum(dim=-1).mean(dim=1)
        
        attention_features = self.attention_encoder(attention_scores.unsqueeze(-1))
        
        # Combine features (concatenate instead of add due to dimension mismatch)
        combined_features = torch.cat([
            protein_repr, distance_features, attention_features
        ], dim=-1)
        
        # Predict binding pocket probabilities
        pocket_probs = self.pocket_predictor(combined_features).squeeze(-1)
        
        # Identify binding pocket residues (threshold-based)
        pocket_threshold = 0.5
        pocket_residues = pocket_probs > pocket_threshold
        
        return {
            'pocket_probabilities': pocket_probs,
            'pocket_residues': pocket_residues,
            'min_distances': distances,
            'attention_scores': attention_scores
        }
    
    def _compute_min_distances(self, protein_coords: torch.Tensor,
                              ligand_coords: torch.Tensor,
                              ligand_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute minimum distance from each residue to the ligand.
        
        Args:
            protein_coords: [batch_size, seq_len, 3]
            ligand_coords: [batch_size, num_atoms, 3]
            ligand_mask: [batch_size, num_atoms]
            
        Returns:
            Minimum distances [batch_size, seq_len]
        """
        
        # Compute all pairwise distances
        protein_expanded = protein_coords.unsqueeze(2)  # [batch_size, seq_len, 1, 3]
        ligand_expanded = ligand_coords.unsqueeze(1)    # [batch_size, 1, num_atoms, 3]
        
        distances = torch.norm(protein_expanded - ligand_expanded, dim=-1)
        # distances: [batch_size, seq_len, num_atoms]
        
        # Apply ligand mask
        if ligand_mask is not None:
            mask = ~ligand_mask.bool().unsqueeze(1)  # [batch_size, 1, num_atoms]
            distances = distances.masked_fill(mask, float('inf'))
        
        # Find minimum distance for each residue
        min_distances, _ = distances.min(dim=-1)
        
        return min_distances
