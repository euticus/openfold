"""
Ligand Encoder for OdinFold

Encodes ligand molecules into embeddings for protein-ligand interaction modeling.
Supports SMILES, MOL2, and SDF input formats.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. Ligand functionality will be limited.")


logger = logging.getLogger(__name__)


class LigandAtomEmbedding(nn.Module):
    """
    Embedding layer for ligand atoms.
    
    Converts atomic properties (element, hybridization, charge, etc.)
    into dense embeddings for neural network processing.
    """
    
    def __init__(self, d_model: int = 256):
        super().__init__()
        
        self.d_model = d_model
        
        # Atom type embeddings (common elements in drug-like molecules)
        self.atom_type_embedding = nn.Embedding(100, d_model // 4)  # Support up to 100 elements
        
        # Hybridization embeddings
        self.hybridization_embedding = nn.Embedding(8, d_model // 8)  # SP, SP2, SP3, etc.
        
        # Formal charge embeddings
        self.charge_embedding = nn.Embedding(11, d_model // 8)  # -5 to +5
        
        # Aromaticity embedding
        self.aromatic_embedding = nn.Embedding(2, d_model // 8)  # Aromatic or not
        
        # Degree embedding (number of bonds)
        self.degree_embedding = nn.Embedding(7, d_model // 8)  # 0-6 bonds
        
        # Implicit hydrogen embedding
        self.implicit_h_embedding = nn.Embedding(5, d_model // 8)  # 0-4 implicit H
        
        # Ring membership embedding
        self.ring_embedding = nn.Embedding(2, d_model // 8)  # In ring or not
        
        # Combine all embeddings
        embedding_dim = (d_model // 4) + 6 * (d_model // 8)
        self.projection = nn.Linear(embedding_dim, d_model)
        
        # 3D position encoding
        self.position_encoder = nn.Linear(3, d_model // 4)
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, atom_features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Encode ligand atoms into embeddings.
        
        Args:
            atom_features: Atomic features [num_atoms, num_features]
            positions: 3D coordinates [num_atoms, 3]
            
        Returns:
            Atom embeddings [num_atoms, d_model]
        """
        
        # Extract features (assuming specific order)
        atom_types = atom_features[:, 0].long()
        hybridizations = atom_features[:, 1].long()
        charges = atom_features[:, 2].long()  # Already in 0-10 range
        aromaticity = atom_features[:, 3].long()
        degrees = atom_features[:, 4].long()
        implicit_h = atom_features[:, 5].long()
        ring_membership = atom_features[:, 6].long()
        
        # Get embeddings
        atom_emb = self.atom_type_embedding(atom_types)
        hybrid_emb = self.hybridization_embedding(hybridizations)
        charge_emb = self.charge_embedding(charges)
        aromatic_emb = self.aromatic_embedding(aromaticity)
        degree_emb = self.degree_embedding(degrees)
        implicit_h_emb = self.implicit_h_embedding(implicit_h)
        ring_emb = self.ring_embedding(ring_membership)
        
        # Concatenate feature embeddings
        feature_emb = torch.cat([
            atom_emb, hybrid_emb, charge_emb, aromatic_emb,
            degree_emb, implicit_h_emb, ring_emb
        ], dim=-1)
        
        # Project to model dimension
        atom_embeddings = self.projection(feature_emb)
        
        # Add 3D position encoding
        pos_emb = self.position_encoder(positions)

        # Pad position embedding to match atom embedding dimension
        if pos_emb.shape[-1] < atom_embeddings.shape[-1]:
            padding_size = atom_embeddings.shape[-1] - pos_emb.shape[-1]
            pos_padding = torch.zeros(*pos_emb.shape[:-1], padding_size, device=pos_emb.device)
            pos_emb = torch.cat([pos_emb, pos_padding], dim=-1)

        atom_embeddings = atom_embeddings + pos_emb
        
        # Layer norm
        atom_embeddings = self.layer_norm(atom_embeddings)
        
        return atom_embeddings


class LigandEncoder(nn.Module):
    """
    Complete ligand encoder that processes molecular input into embeddings.
    
    Handles SMILES strings, MOL2 files, and SDF files, converting them
    into atom-level embeddings for protein-ligand interaction modeling.
    """
    
    def __init__(self, d_model: int = 256, max_atoms: int = 100):
        super().__init__()
        
        self.d_model = d_model
        self.max_atoms = max_atoms
        
        # Atom embedding layer
        self.atom_embedding = LigandAtomEmbedding(d_model)
        
        # Self-attention for ligand atoms
        self.ligand_self_attention = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Ligand-level pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Ligand property prediction heads (optional)
        self.property_heads = nn.ModuleDict({
            'molecular_weight': nn.Linear(d_model, 1),
            'logp': nn.Linear(d_model, 1),
            'tpsa': nn.Linear(d_model, 1),
            'num_rotatable_bonds': nn.Linear(d_model, 1)
        })
    
    def forward(self, ligand_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode ligand into embeddings.
        
        Args:
            ligand_data: Dictionary containing:
                - atom_features: [num_atoms, num_features]
                - positions: [num_atoms, 3]
                - atom_mask: [num_atoms] (optional)
                
        Returns:
            Dictionary containing ligand embeddings and properties
        """
        
        atom_features = ligand_data['atom_features']
        positions = ligand_data['positions']
        atom_mask = ligand_data.get('atom_mask', None)
        
        num_atoms = atom_features.shape[0]
        
        # Pad or truncate to max_atoms
        if num_atoms > self.max_atoms:
            atom_features = atom_features[:self.max_atoms]
            positions = positions[:self.max_atoms]
            if atom_mask is not None:
                atom_mask = atom_mask[:self.max_atoms]
            num_atoms = self.max_atoms
        elif num_atoms < self.max_atoms:
            padding_atoms = self.max_atoms - num_atoms
            atom_features = torch.cat([
                atom_features,
                torch.zeros(padding_atoms, atom_features.shape[1], device=atom_features.device)
            ], dim=0)
            positions = torch.cat([
                positions,
                torch.zeros(padding_atoms, 3, device=positions.device)
            ], dim=0)
            if atom_mask is None:
                atom_mask = torch.cat([
                    torch.ones(num_atoms, device=atom_features.device),
                    torch.zeros(padding_atoms, device=atom_features.device)
                ], dim=0)
            else:
                atom_mask = torch.cat([
                    atom_mask,
                    torch.zeros(padding_atoms, device=atom_mask.device)
                ], dim=0)
        
        # Get atom embeddings
        atom_embeddings = self.atom_embedding(atom_features, positions)
        
        # Add batch dimension for attention
        atom_embeddings = atom_embeddings.unsqueeze(0)  # [1, max_atoms, d_model]
        
        # Self-attention over ligand atoms
        if atom_mask is not None:
            # Create attention mask
            attn_mask = ~atom_mask.bool().unsqueeze(0)  # [1, max_atoms]
        else:
            attn_mask = None
        
        attended_atoms, _ = self.ligand_self_attention(
            atom_embeddings, atom_embeddings, atom_embeddings,
            key_padding_mask=attn_mask
        )
        
        # Layer norm
        attended_atoms = self.layer_norm(attended_atoms + atom_embeddings)
        
        # Remove batch dimension
        attended_atoms = attended_atoms.squeeze(0)  # [max_atoms, d_model]
        
        # Global ligand embedding (mean pooling over valid atoms)
        if atom_mask is not None:
            valid_atoms = atom_mask.bool()
            ligand_embedding = attended_atoms[valid_atoms].mean(dim=0)
        else:
            ligand_embedding = attended_atoms.mean(dim=0)
        
        # Predict molecular properties
        properties = {}
        for prop_name, head in self.property_heads.items():
            properties[prop_name] = head(ligand_embedding)
        
        return {
            'atom_embeddings': attended_atoms,
            'ligand_embedding': ligand_embedding,
            'atom_mask': atom_mask,
            'predicted_properties': properties,
            'num_atoms': num_atoms
        }
    
    def encode_smiles(self, smiles: str, conformer_id: int = 0) -> Optional[Dict[str, torch.Tensor]]:
        """
        Encode a SMILES string into ligand embeddings.
        
        Args:
            smiles: SMILES string
            conformer_id: Conformer ID to use for 3D coordinates
            
        Returns:
            Ligand data dictionary or None if parsing fails
        """
        
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for SMILES processing")
        
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Failed to parse SMILES: {smiles}")
                return None
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.OptimizeMoleculeConfs(mol)
            
            # Extract features
            ligand_data = self._extract_mol_features(mol, conformer_id)
            
            return ligand_data
            
        except Exception as e:
            logger.error(f"Error processing SMILES {smiles}: {e}")
            return None
    
    def _extract_mol_features(self, mol, conformer_id: int = 0) -> Dict[str, torch.Tensor]:
        """
        Extract features from RDKit molecule.
        
        Args:
            mol: RDKit molecule object
            conformer_id: Conformer ID for 3D coordinates
            
        Returns:
            Dictionary with atom features and positions
        """
        
        num_atoms = mol.GetNumAtoms()
        
        # Initialize feature arrays
        atom_features = []
        positions = []
        
        # Get conformer for 3D coordinates
        conf = mol.GetConformer(conformer_id)
        
        for atom in mol.GetAtoms():
            # Atomic number
            atomic_num = atom.GetAtomicNum()
            
            # Hybridization
            hybridization = int(atom.GetHybridization())
            
            # Formal charge
            formal_charge = atom.GetFormalCharge()
            
            # Aromaticity
            is_aromatic = int(atom.GetIsAromatic())
            
            # Degree (number of bonds)
            degree = atom.GetDegree()
            
            # Implicit hydrogens
            implicit_h = atom.GetTotalNumHs()
            
            # Ring membership
            is_in_ring = int(atom.IsInRing())
            
            # Combine features
            features = [
                atomic_num, hybridization, formal_charge,
                is_aromatic, degree, implicit_h, is_in_ring
            ]
            atom_features.append(features)
            
            # 3D position
            pos = conf.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
        
        # Convert to tensors
        atom_features = torch.tensor(atom_features, dtype=torch.float32)
        positions = torch.tensor(positions, dtype=torch.float32)
        
        return {
            'atom_features': atom_features,
            'positions': positions,
            'smiles': Chem.MolToSmiles(mol),
            'num_atoms': num_atoms
        }


def create_mock_ligand(num_atoms: int = 20, d_model: int = 256) -> Dict[str, torch.Tensor]:
    """
    Create mock ligand data for testing when RDKit is not available.

    Args:
        num_atoms: Number of atoms in mock ligand
        d_model: Model dimension

    Returns:
        Mock ligand data dictionary
    """

    # Mock atom features with proper ranges
    atom_features = torch.zeros(num_atoms, 7)

    # Atomic numbers (1-20 for common elements)
    atom_features[:, 0] = torch.randint(1, 20, (num_atoms,))

    # Hybridization (0-7)
    atom_features[:, 1] = torch.randint(0, 8, (num_atoms,))

    # Formal charge (-5 to +5, but stored as 0-10)
    atom_features[:, 2] = torch.randint(0, 11, (num_atoms,))

    # Aromaticity (0 or 1)
    atom_features[:, 3] = torch.randint(0, 2, (num_atoms,))

    # Degree (0-6)
    atom_features[:, 4] = torch.randint(0, 7, (num_atoms,))

    # Implicit hydrogens (0-4)
    atom_features[:, 5] = torch.randint(0, 5, (num_atoms,))

    # Ring membership (0 or 1)
    atom_features[:, 6] = torch.randint(0, 2, (num_atoms,))

    # Mock 3D positions (small molecule-like)
    positions = torch.randn(num_atoms, 3) * 5.0

    # All atoms are valid
    atom_mask = torch.ones(num_atoms)

    return {
        'atom_features': atom_features,
        'positions': positions,
        'atom_mask': atom_mask,
        'smiles': 'CCO',  # Mock SMILES
        'num_atoms': num_atoms
    }
