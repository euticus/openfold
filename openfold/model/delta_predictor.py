"""
Delta prediction model for real-time mutation effects.

This module implements a GNN/SE(3)-based model to predict structural changes
from point mutations without re-running the full folding pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, TransformerConv
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import k_hop_subgraph
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

try:
    import e3nn
    from e3nn import o3
    from e3nn.nn import Gate
    E3NN_AVAILABLE = True
except ImportError:
    E3NN_AVAILABLE = False

from openfold.np import protein, residue_constants


@dataclass
class MutationInput:
    """Input for mutation prediction."""
    protein_structure: protein.Protein
    mutation_position: int  # 0-indexed residue position
    original_aa: str  # Original amino acid (single letter)
    target_aa: str   # Target amino acid (single letter)
    local_radius: float = 10.0  # Angstrom radius for local environment


@dataclass
class DeltaPrediction:
    """Output of delta prediction."""
    position_deltas: torch.Tensor  # [num_atoms, 3] coordinate changes
    confidence_scores: torch.Tensor  # [num_atoms] confidence per atom
    affected_residues: List[int]  # Residue indices that may be affected
    energy_change: Optional[float] = None  # Predicted energy change


class ProteinGraphBuilder:
    """Builds graph representations of protein structures for GNN processing."""
    
    def __init__(self, 
                 contact_threshold: float = 8.0,
                 include_backbone_only: bool = False):
        """
        Args:
            contact_threshold: Distance threshold for edges (Angstroms)
            include_backbone_only: Whether to include only backbone atoms
        """
        self.contact_threshold = contact_threshold
        self.include_backbone_only = include_backbone_only
    
    def protein_to_graph(self, prot: protein.Protein, mutation_pos: Optional[int] = None) -> Data:
        """
        Convert protein structure to graph representation.
        
        Args:
            prot: Protein structure
            mutation_pos: Position of mutation (for special encoding)
            
        Returns:
            PyTorch Geometric Data object
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric required for graph operations")
        
        # Extract coordinates and features
        positions = prot.atom_positions  # [num_res, 37, 3]
        aatype = prot.aatype  # [num_res]
        atom_mask = prot.atom_mask  # [num_res, 37]
        
        # Select atoms (backbone only or all)
        if self.include_backbone_only:
            # N, CA, C atoms (indices 0, 1, 2)
            selected_atoms = [0, 1, 2]
        else:
            # All atoms
            selected_atoms = list(range(37))
        
        # Build node features and coordinates
        node_coords = []
        node_features = []
        residue_indices = []
        atom_types = []
        
        for res_idx in range(len(aatype)):
            for atom_idx in selected_atoms:
                if atom_mask[res_idx, atom_idx] > 0:  # Atom is present
                    # Coordinates
                    coord = positions[res_idx, atom_idx]
                    node_coords.append(coord)
                    
                    # Features: [aa_type_onehot(21) + atom_type_onehot(37) + is_mutation_site(1)]
                    aa_onehot = F.one_hot(torch.tensor(aatype[res_idx]), num_classes=21).float()
                    atom_onehot = F.one_hot(torch.tensor(atom_idx), num_classes=37).float()
                    is_mutation = torch.tensor([1.0 if res_idx == mutation_pos else 0.0])
                    
                    node_feat = torch.cat([aa_onehot, atom_onehot, is_mutation])
                    node_features.append(node_feat)
                    
                    residue_indices.append(res_idx)
                    atom_types.append(atom_idx)
        
        if len(node_coords) == 0:
            raise ValueError("No valid atoms found in protein structure")
        
        node_coords = torch.tensor(np.array(node_coords), dtype=torch.float32)
        node_features = torch.stack(node_features)
        
        # Build edges based on distance
        edge_indices = []
        edge_features = []
        
        for i in range(len(node_coords)):
            for j in range(i + 1, len(node_coords)):
                dist = torch.norm(node_coords[i] - node_coords[j])
                
                if dist <= self.contact_threshold:
                    # Add both directions
                    edge_indices.extend([[i, j], [j, i]])
                    
                    # Edge features: [distance, same_residue, bond_type]
                    same_residue = 1.0 if residue_indices[i] == residue_indices[j] else 0.0
                    
                    # Simple bond type classification
                    if same_residue and dist < 2.0:
                        bond_type = [1.0, 0.0, 0.0]  # Covalent bond
                    elif dist < 4.0:
                        bond_type = [0.0, 1.0, 0.0]  # Close contact
                    else:
                        bond_type = [0.0, 0.0, 1.0]  # Distant contact
                    
                    edge_feat = torch.tensor([dist.item(), same_residue] + bond_type)
                    edge_features.extend([edge_feat, edge_feat])  # Both directions
        
        if len(edge_indices) == 0:
            # Add self-loops if no edges
            edge_indices = [[i, i] for i in range(len(node_coords))]
            edge_features = [torch.zeros(5) for _ in range(len(node_coords))]
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_features)
        
        # Create graph data
        data = Data(
            x=node_features,
            pos=node_coords,
            edge_index=edge_index,
            edge_attr=edge_attr,
            residue_indices=torch.tensor(residue_indices),
            atom_types=torch.tensor(atom_types)
        )
        
        return data


class SE3EquivariantGNN(nn.Module):
    """SE(3)-equivariant GNN for structure-aware mutation prediction."""
    
    def __init__(self,
                 node_feat_dim: int = 59,  # 21 + 37 + 1
                 edge_feat_dim: int = 5,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 max_degree: int = 2):
        super().__init__()
        
        if not E3NN_AVAILABLE:
            raise ImportError("e3nn required for SE(3)-equivariant operations")
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input embedding
        self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_feat_dim, hidden_dim)
        
        # SE(3)-equivariant layers
        irreps_hidden = o3.Irreps(f"{hidden_dim}x0e + {hidden_dim//4}x1o + {hidden_dim//8}x2e")
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = o3.TensorProduct(
                irreps_hidden, irreps_hidden, irreps_hidden,
                instructions=[
                    (i, j, k, "uvu", True)
                    for i, (mul_i, ir_i) in enumerate(irreps_hidden)
                    for j, (mul_j, ir_j) in enumerate(irreps_hidden)
                    for k, (mul_k, ir_k) in enumerate(irreps_hidden)
                    if ir_k in ir_i * ir_j
                ]
            )
            self.layers.append(layer)
        
        # Output layers
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3D coordinate changes
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of SE(3)-equivariant GNN.
        
        Args:
            data: Graph data with node features, positions, and edges
            
        Returns:
            Tuple of (position_deltas, confidence_scores)
        """
        x = self.node_embedding(data.x)
        edge_attr = self.edge_embedding(data.edge_attr)
        
        # SE(3)-equivariant message passing
        for layer in self.layers:
            # This is a simplified version - full implementation would need
            # proper SE(3)-equivariant message passing
            x_new = x  # Placeholder for SE(3) operations
            x = x + x_new
        
        # Predict coordinate changes and confidence
        position_deltas = self.position_head(x)
        confidence_scores = self.confidence_head(x).squeeze(-1)
        
        return position_deltas, confidence_scores


class SimpleGNN(nn.Module):
    """Simple GNN fallback when SE(3) libraries are not available."""
    
    def __init__(self,
                 node_feat_dim: int = 59,
                 edge_feat_dim: int = 5,
                 hidden_dim: int = 128,
                 num_layers: int = 4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input embedding
        self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)
        
        # GNN layers
        if TORCH_GEOMETRIC_AVAILABLE:
            self.conv_layers = nn.ModuleList([
                GATConv(hidden_dim, hidden_dim, edge_dim=edge_feat_dim, heads=4, concat=False)
                for _ in range(num_layers)
            ])
        else:
            # Fallback to simple MLPs
            self.conv_layers = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
            ])
        
        # Output heads
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of simple GNN."""
        x = self.node_embedding(data.x)
        
        # Graph convolution layers
        for conv in self.conv_layers:
            if TORCH_GEOMETRIC_AVAILABLE and hasattr(conv, 'edge_dim'):
                x_new = conv(x, data.edge_index, data.edge_attr)
            else:
                x_new = conv(x)
            x = F.relu(x + x_new)  # Residual connection
        
        # Predict outputs
        position_deltas = self.position_head(x)
        confidence_scores = self.confidence_head(x).squeeze(-1)
        
        return position_deltas, confidence_scores


class DeltaPredictor(nn.Module):
    """Main delta prediction model for mutations."""
    
    def __init__(self,
                 model_type: str = "simple_gnn",  # "simple_gnn" or "se3_gnn"
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 local_radius: float = 10.0):
        """
        Args:
            model_type: Type of GNN to use
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            local_radius: Radius for local environment extraction
        """
        super().__init__()
        
        self.model_type = model_type
        self.local_radius = local_radius
        
        # Graph builder
        self.graph_builder = ProteinGraphBuilder(
            contact_threshold=8.0,
            include_backbone_only=False
        )
        
        # GNN model
        if model_type == "se3_gnn" and E3NN_AVAILABLE:
            self.gnn = SE3EquivariantGNN(
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )
        else:
            self.gnn = SimpleGNN(
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )
        
        # Mutation encoding
        self.mutation_encoder = nn.Sequential(
            nn.Linear(42, hidden_dim),  # 21 (from) + 21 (to)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def extract_local_environment(self, 
                                 prot: protein.Protein, 
                                 mutation_pos: int) -> protein.Protein:
        """Extract local environment around mutation site."""
        # Get CA position of mutation site
        ca_pos = prot.atom_positions[mutation_pos, 1]  # CA atom
        
        # Find residues within radius
        local_residues = []
        for i, pos in enumerate(prot.atom_positions[:, 1]):  # CA positions
            if np.linalg.norm(pos - ca_pos) <= self.local_radius:
                local_residues.append(i)
        
        if len(local_residues) == 0:
            local_residues = [mutation_pos]
        
        # Extract local protein
        local_prot = protein.Protein(
            atom_positions=prot.atom_positions[local_residues],
            atom_mask=prot.atom_mask[local_residues],
            aatype=prot.aatype[local_residues],
            residue_index=prot.residue_index[local_residues],
            b_factors=prot.b_factors[local_residues]
        )
        
        # Adjust mutation position to local indexing
        local_mutation_pos = local_residues.index(mutation_pos)
        
        return local_prot, local_mutation_pos, local_residues
    
    def forward(self, mutation_input: MutationInput) -> DeltaPrediction:
        """
        Predict structural changes from mutation.
        
        Args:
            mutation_input: Mutation specification
            
        Returns:
            Predicted structural changes
        """
        # Extract local environment
        local_prot, local_mut_pos, affected_residues = self.extract_local_environment(
            mutation_input.protein_structure,
            mutation_input.mutation_position
        )
        
        # Build graph
        graph_data = self.graph_builder.protein_to_graph(local_prot, local_mut_pos)
        
        # Encode mutation
        from_aa = residue_constants.restype_order.get(mutation_input.original_aa, 20)
        to_aa = residue_constants.restype_order.get(mutation_input.target_aa, 20)
        
        from_onehot = F.one_hot(torch.tensor(from_aa), num_classes=21).float()
        to_onehot = F.one_hot(torch.tensor(to_aa), num_classes=21).float()
        mutation_encoding = torch.cat([from_onehot, to_onehot])
        
        # Add mutation encoding to graph (broadcast to all nodes)
        mutation_feat = self.mutation_encoder(mutation_encoding)

        # Update node embedding layer to handle concatenated features
        if not hasattr(self, '_updated_node_embedding'):
            old_dim = self.gnn.node_embedding.in_features
            new_dim = old_dim + mutation_feat.size(0)
            self.gnn.node_embedding = nn.Linear(new_dim, self.gnn.hidden_dim)
            self._updated_node_embedding = True

        graph_data.x = torch.cat([
            graph_data.x,
            mutation_feat.unsqueeze(0).expand(graph_data.x.size(0), -1)
        ], dim=1)
        
        # Predict deltas
        position_deltas, confidence_scores = self.gnn(graph_data)
        
        return DeltaPrediction(
            position_deltas=position_deltas,
            confidence_scores=confidence_scores,
            affected_residues=affected_residues
        )


def create_delta_predictor(model_type: str = "simple_gnn", **kwargs) -> DeltaPredictor:
    """
    Factory function to create delta predictor.
    
    Args:
        model_type: Type of model ("simple_gnn" or "se3_gnn")
        **kwargs: Additional arguments for model
        
    Returns:
        DeltaPredictor instance
    """
    return DeltaPredictor(model_type=model_type, **kwargs)
