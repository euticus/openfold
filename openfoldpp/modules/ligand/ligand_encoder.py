"""
Ligand Encoder for OdinFold

Encodes molecular graphs of ligands into embeddings that can be used
for ligand-aware protein folding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import logging

logger = logging.getLogger(__name__)


class AtomTypeEmbedding(nn.Module):
    """
    Embedding layer for atom types in molecular graphs.
    
    Supports common atom types found in drug-like molecules
    and biological ligands.
    """
    
    def __init__(self, d_model: int = 128):
        super().__init__()
        
        self.d_model = d_model
        
        # Common atom types (sequential indices)
        self.atom_types = {
            'UNK': 0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'P': 6, 'S': 7, 'Cl': 8,
            'Br': 9, 'I': 10, 'B': 11, 'Si': 12, 'Se': 13, 'As': 14, 'Fe': 15,
            'Zn': 16, 'Ca': 17, 'Mg': 18, 'Na': 19, 'K': 20, 'Mn': 21, 'Cu': 22
        }
        
        # Reverse mapping
        self.idx_to_atom = {v: k for k, v in self.atom_types.items()}
        
        # Atom type embedding
        self.atom_embedding = nn.Embedding(len(self.atom_types), d_model // 4)
        
        # Atom properties (atomic number, electronegativity, etc.)
        self.register_buffer('atomic_numbers', self._create_atomic_numbers())
        self.register_buffer('electronegativities', self._create_electronegativities())
        self.register_buffer('atomic_radii', self._create_atomic_radii())
        
        # Property projections
        self.property_proj = nn.Linear(3, d_model // 4)  # 3 properties
        
        # Hybridization embedding
        self.hybridization_embedding = nn.Embedding(5, d_model // 4)  # SP, SP2, SP3, SP3D, SP3D2
        
        # Formal charge embedding
        self.charge_embedding = nn.Embedding(11, d_model // 4)  # -5 to +5
        
        # Final projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.atom_embedding.weight, std=0.02)
        nn.init.normal_(self.hybridization_embedding.weight, std=0.02)
        nn.init.normal_(self.charge_embedding.weight, std=0.02)
        nn.init.normal_(self.property_proj.weight, std=0.02)
        nn.init.zeros_(self.property_proj.bias)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
    
    def _create_atomic_numbers(self) -> torch.Tensor:
        """Create atomic number lookup."""
        atomic_nums = torch.zeros(len(self.atom_types))
        # Map from symbol to actual atomic number
        atom_to_num = {
            'UNK': 0, 'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17,
            'Br': 35, 'I': 53, 'B': 5, 'Si': 14, 'Se': 34, 'As': 33, 'Fe': 26,
            'Zn': 30, 'Ca': 20, 'Mg': 12, 'Na': 11, 'K': 19, 'Mn': 25, 'Cu': 29
        }
        for symbol, idx in self.atom_types.items():
            atomic_nums[idx] = atom_to_num.get(symbol, 0)
        return atomic_nums
    
    def _create_electronegativities(self) -> torch.Tensor:
        """Create electronegativity lookup (Pauling scale)."""
        electroneg = torch.zeros(len(self.atom_types))
        for symbol, idx in self.atom_types.items():
            # Simplified electronegativity values
            electroneg_map = {
                'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98, 'P': 2.19,
                'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66, 'B': 2.04, 'Si': 1.90,
                'Se': 2.55, 'As': 2.18, 'Fe': 1.83, 'Zn': 1.65, 'Ca': 1.00, 'Mg': 1.31,
                'Na': 0.93, 'K': 0.82, 'Mn': 1.55, 'Cu': 1.90, 'UNK': 2.0
            }
            electroneg[idx] = electroneg_map.get(symbol, 2.0)
        return electroneg
    
    def _create_atomic_radii(self) -> torch.Tensor:
        """Create atomic radii lookup (van der Waals radii in Angstroms)."""
        radii = torch.zeros(len(self.atom_types))
        for symbol, idx in self.atom_types.items():
            # van der Waals radii
            radii_map = {
                'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'P': 1.80,
                'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98, 'B': 1.92, 'Si': 2.10,
                'Se': 1.90, 'As': 1.85, 'Fe': 2.00, 'Zn': 1.39, 'Ca': 2.31, 'Mg': 1.73,
                'Na': 2.27, 'K': 2.75, 'Mn': 2.05, 'Cu': 1.40, 'UNK': 1.70
            }
            radii[idx] = radii_map.get(symbol, 1.70)
        return radii
    
    def forward(self,
                atom_types: torch.Tensor,
                hybridization: Optional[torch.Tensor] = None,
                formal_charges: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode atom types into embeddings.
        
        Args:
            atom_types: Atom type indices [num_atoms]
            hybridization: Hybridization types [num_atoms] (optional)
            formal_charges: Formal charges [num_atoms] (optional)
            
        Returns:
            Atom embeddings [num_atoms, d_model]
        """
        
        num_atoms = atom_types.shape[0]
        device = atom_types.device
        
        # Atom type embeddings
        atom_emb = self.atom_embedding(atom_types)  # [num_atoms, d_model//4]
        
        # Atom properties
        atomic_nums = self.atomic_numbers[atom_types]
        electroneg = self.electronegativities[atom_types]
        radii = self.atomic_radii[atom_types]
        
        properties = torch.stack([atomic_nums, electroneg, radii], dim=-1)  # [num_atoms, 3]
        prop_emb = self.property_proj(properties)  # [num_atoms, d_model//4]
        
        # Hybridization (default to SP3 if not provided)
        if hybridization is None:
            hybridization = torch.full((num_atoms,), 2, device=device)  # SP3
        hyb_emb = self.hybridization_embedding(hybridization)  # [num_atoms, d_model//4]
        
        # Formal charges (default to 0 if not provided)
        if formal_charges is None:
            formal_charges = torch.full((num_atoms,), 5, device=device)  # 0 charge (index 5)
        else:
            # Clamp charges to [-5, 5] and shift to [0, 10]
            formal_charges = torch.clamp(formal_charges + 5, 0, 10)
        charge_emb = self.charge_embedding(formal_charges)  # [num_atoms, d_model//4]
        
        # Combine all embeddings
        combined = torch.cat([atom_emb, prop_emb, hyb_emb, charge_emb], dim=-1)  # [num_atoms, d_model]
        output = self.output_proj(combined)
        
        return output
    
    def get_atom_index(self, atom_symbol: str) -> int:
        """Get index for atom symbol."""
        return self.atom_types.get(atom_symbol.upper(), self.atom_types['UNK'])


class LigandFeatureExtractor(nn.Module):
    """
    Extract additional molecular features for ligands.
    
    Computes molecular descriptors and pharmacophore features
    that are useful for protein-ligand interactions.
    """
    
    def __init__(self, d_model: int = 128):
        super().__init__()
        
        self.d_model = d_model
        
        # Bond type embedding
        self.bond_embedding = nn.Embedding(5, d_model // 8)  # SINGLE, DOUBLE, TRIPLE, AROMATIC, OTHER
        
        # Ring membership embedding
        self.ring_embedding = nn.Embedding(3, d_model // 8)  # NOT_IN_RING, IN_RING, AROMATIC_RING
        
        # Pharmacophore features
        self.pharmacophore_proj = nn.Linear(8, d_model // 4)  # 8 pharmacophore features
        
        # Molecular descriptors
        self.descriptor_proj = nn.Linear(6, d_model // 4)  # 6 molecular descriptors
        
        # Final projection (d_model//4 + d_model//4 + d_model//4 = 3*d_model//4)
        self.output_proj = nn.Linear(3 * d_model // 4, d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.bond_embedding.weight, std=0.02)
        nn.init.normal_(self.ring_embedding.weight, std=0.02)
        nn.init.normal_(self.pharmacophore_proj.weight, std=0.02)
        nn.init.zeros_(self.pharmacophore_proj.bias)
        nn.init.normal_(self.descriptor_proj.weight, std=0.02)
        nn.init.zeros_(self.descriptor_proj.bias)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self,
                bond_types: torch.Tensor,
                ring_info: torch.Tensor,
                pharmacophore_features: torch.Tensor,
                molecular_descriptors: torch.Tensor) -> torch.Tensor:
        """
        Extract ligand features.
        
        Args:
            bond_types: Bond type indices [num_bonds]
            ring_info: Ring membership info [num_atoms]
            pharmacophore_features: Pharmacophore features [num_atoms, 8]
            molecular_descriptors: Molecular descriptors [num_atoms, 6]
            
        Returns:
            Ligand features [num_atoms, d_model]
        """
        
        # Bond embeddings (aggregate to atoms)
        bond_emb = self.bond_embedding(bond_types)  # [num_bonds, d_model//8]
        # For simplicity, use mean bond embedding per atom (would need bond-atom mapping in practice)
        if bond_emb.shape[0] > 0:
            atom_bond_emb = bond_emb.mean(dim=0, keepdim=True).expand(ring_info.shape[0], -1)
        else:
            atom_bond_emb = torch.zeros(ring_info.shape[0], self.d_model // 8, device=ring_info.device)
        
        # Ring embeddings
        ring_emb = self.ring_embedding(ring_info)  # [num_atoms, d_model//8]
        
        # Combine bond and ring features
        structural_features = torch.cat([atom_bond_emb, ring_emb], dim=-1)  # [num_atoms, d_model//4]

        # Pharmacophore features
        pharm_emb = self.pharmacophore_proj(pharmacophore_features)  # [num_atoms, d_model//4]

        # Molecular descriptors
        desc_emb = self.descriptor_proj(molecular_descriptors)  # [num_atoms, d_model//4]

        # Combine all features (3 * d_model//4)
        combined = torch.cat([structural_features, pharm_emb, desc_emb], dim=-1)
        output = self.output_proj(combined)
        
        return output


class MolecularGraphEncoder(nn.Module):
    """
    Graph neural network encoder for molecular graphs.
    
    Uses message passing to encode molecular structure
    and chemical properties.
    """
    
    def __init__(self,
                 d_model: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Residual connections
        self.dropout = nn.Dropout(dropout)
        
        # Global pooling for molecule-level features
        self.global_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Encode molecular graph.
        
        Args:
            node_features: Node features [num_atoms, d_model]
            edge_index: Edge connectivity [2, num_bonds]
            edge_features: Edge features [num_bonds, d_edge] (optional)
            
        Returns:
            Dictionary with node and graph embeddings
        """
        
        x = node_features
        
        # Apply graph attention layers
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            residual = x
            x = gat_layer(x, edge_index, edge_features)
            x = self.dropout(x)
            x = layer_norm(x + residual)
        
        # Global pooling for molecule-level representation
        graph_embedding = self.global_pool(x.mean(dim=0, keepdim=True))  # [1, d_model]
        
        return {
            'node_embeddings': x,  # [num_atoms, d_model]
            'graph_embedding': graph_embedding.squeeze(0),  # [d_model]
            'num_atoms': x.shape[0]
        }


class GraphAttentionLayer(nn.Module):
    """
    Graph attention layer for molecular graphs.
    
    Implements multi-head attention over molecular graph edges.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Edge feature projection (if provided)
        self.edge_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.normal_(self.edge_proj.weight, std=0.02)
        nn.init.zeros_(self.edge_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply graph attention.
        
        Args:
            x: Node features [num_atoms, d_model]
            edge_index: Edge connectivity [2, num_bonds]
            edge_features: Edge features [num_bonds, d_model] (optional)
            
        Returns:
            Updated node features [num_atoms, d_model]
        """
        
        num_atoms = x.shape[0]
        
        # Compute queries, keys, values
        q = self.q_proj(x).view(num_atoms, self.num_heads, self.head_dim)  # [num_atoms, num_heads, head_dim]
        k = self.k_proj(x).view(num_atoms, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(num_atoms, self.num_heads, self.head_dim)
        
        # Initialize output
        out = torch.zeros_like(x)
        
        # For each atom, aggregate information from neighbors
        for i in range(num_atoms):
            # Find neighbors
            neighbor_mask = (edge_index[0] == i) | (edge_index[1] == i)
            if not neighbor_mask.any():
                out[i] = x[i]  # No neighbors, keep original
                continue
            
            # Get neighbor indices
            neighbor_edges = edge_index[:, neighbor_mask]
            neighbors = torch.unique(neighbor_edges.flatten())
            neighbors = neighbors[neighbors != i]  # Remove self
            
            if len(neighbors) == 0:
                out[i] = x[i]
                continue
            
            # Compute attention scores
            q_i = q[i]  # [num_heads, head_dim]
            k_neighbors = k[neighbors]  # [num_neighbors, num_heads, head_dim]
            
            # Attention scores
            scores = torch.einsum('hd,nhd->nh', q_i, k_neighbors)  # [num_neighbors, num_heads]
            scores = scores / math.sqrt(self.head_dim)
            
            # Apply edge features if provided
            if edge_features is not None:
                edge_mask = neighbor_mask
                edge_feats = edge_features[edge_mask]
                if len(edge_feats) > 0:
                    edge_bias = self.edge_proj(edge_feats.mean(dim=0))  # Average edge features
                    edge_bias = edge_bias.view(self.num_heads, self.head_dim).mean(dim=-1)  # [num_heads]
                    scores = scores + edge_bias.unsqueeze(0)  # Broadcast to [num_neighbors, num_heads]
            
            # Softmax attention weights
            attn_weights = F.softmax(scores, dim=0)  # [num_neighbors, num_heads]
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            v_neighbors = v[neighbors]  # [num_neighbors, num_heads, head_dim]
            attended = torch.einsum('nh,nhd->hd', attn_weights, v_neighbors)  # [num_heads, head_dim]
            
            # Flatten and project
            attended_flat = attended.view(-1)  # [d_model]
            out[i] = self.out_proj(attended_flat)
        
        return out


class LigandEncoder(nn.Module):
    """
    Complete ligand encoder combining atom embeddings,
    molecular features, and graph neural networks.
    """
    
    def __init__(self,
                 d_model: int = 128,
                 num_gnn_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Atom type embedding
        self.atom_embedding = AtomTypeEmbedding(d_model)
        
        # Molecular feature extractor
        self.feature_extractor = LigandFeatureExtractor(d_model)
        
        # Molecular graph encoder
        self.graph_encoder = MolecularGraphEncoder(
            d_model, num_gnn_layers, num_heads, dropout
        )
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.fusion_layer:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
    
    def forward(self, ligand_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode ligand molecular graph.
        
        Args:
            ligand_data: Dictionary containing:
                - atom_types: [num_atoms]
                - edge_index: [2, num_bonds]
                - bond_types: [num_bonds]
                - ring_info: [num_atoms]
                - pharmacophore_features: [num_atoms, 8]
                - molecular_descriptors: [num_atoms, 6]
                - hybridization: [num_atoms] (optional)
                - formal_charges: [num_atoms] (optional)
                - edge_features: [num_bonds, d_edge] (optional)
                
        Returns:
            Dictionary with ligand embeddings
        """
        
        # Extract atom embeddings
        atom_embeddings = self.atom_embedding(
            ligand_data['atom_types'],
            ligand_data.get('hybridization'),
            ligand_data.get('formal_charges')
        )
        
        # Extract molecular features
        molecular_features = self.feature_extractor(
            ligand_data['bond_types'],
            ligand_data['ring_info'],
            ligand_data['pharmacophore_features'],
            ligand_data['molecular_descriptors']
        )
        
        # Fuse atom and molecular features
        fused_features = self.fusion_layer(
            torch.cat([atom_embeddings, molecular_features], dim=-1)
        )
        
        # Apply graph neural network
        graph_output = self.graph_encoder(
            fused_features,
            ligand_data['edge_index'],
            ligand_data.get('edge_features')
        )
        
        return {
            'atom_embeddings': graph_output['node_embeddings'],  # [num_atoms, d_model]
            'ligand_embedding': graph_output['graph_embedding'],  # [d_model]
            'num_atoms': graph_output['num_atoms'],
            'raw_atom_embeddings': atom_embeddings,
            'molecular_features': molecular_features,
            'fused_features': fused_features
        }
