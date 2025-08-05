"""
Ligand Utilities for OdinFold

Utility functions for processing ligands, converting molecular formats,
and computing ligand-protein interactions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)

# Try to import RDKit for molecular processing
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available. Some ligand processing features will be limited.")
    RDKIT_AVAILABLE = False


def smiles_to_graph(smiles: str, add_hydrogens: bool = True) -> Dict[str, torch.Tensor]:
    """
    Convert SMILES string to molecular graph representation.
    
    Args:
        smiles: SMILES string
        add_hydrogens: Whether to add explicit hydrogens
        
    Returns:
        Dictionary with graph data
    """
    
    if not RDKIT_AVAILABLE:
        return _mock_ligand_graph()
    
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Failed to parse SMILES: {smiles}")
            return _mock_ligand_graph()
        
        # Add hydrogens if requested
        if add_hydrogens:
            mol = Chem.AddHs(mol)
        
        return mol_to_graph(mol)
    
    except Exception as e:
        logger.error(f"Error processing SMILES {smiles}: {e}")
        return _mock_ligand_graph()


def mol_to_graph(mol) -> Dict[str, torch.Tensor]:
    """
    Convert RDKit molecule to graph representation.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary with graph data
    """
    
    if not RDKIT_AVAILABLE:
        return _mock_ligand_graph()
    
    try:
        # Get atoms
        atoms = mol.GetAtoms()
        num_atoms = len(atoms)
        
        if num_atoms == 0:
            return _mock_ligand_graph()
        
        # Extract atom features
        atom_types = []
        hybridizations = []
        formal_charges = []
        ring_info = []
        
        for atom in atoms:
            # Atom type
            atom_types.append(_get_atom_type_index(atom.GetSymbol()))
            
            # Hybridization
            hyb = atom.GetHybridization()
            hyb_idx = _get_hybridization_index(hyb)
            hybridizations.append(hyb_idx)
            
            # Formal charge (clamp to [-5, 5] and shift to [0, 10])
            charge = max(-5, min(5, atom.GetFormalCharge())) + 5
            formal_charges.append(charge)
            
            # Ring information
            if atom.IsInRing():
                if atom.GetIsAromatic():
                    ring_info.append(2)  # Aromatic ring
                else:
                    ring_info.append(1)  # Aliphatic ring
            else:
                ring_info.append(0)  # Not in ring
        
        # Extract bonds
        bonds = mol.GetBonds()
        edge_indices = []
        bond_types = []
        
        for bond in bonds:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions for undirected graph
            edge_indices.extend([[i, j], [j, i]])
            
            # Bond type
            bond_type = _get_bond_type_index(bond.GetBondType())
            bond_types.extend([bond_type, bond_type])
        
        # Convert to tensors
        atom_types = torch.tensor(atom_types, dtype=torch.long)
        hybridizations = torch.tensor(hybridizations, dtype=torch.long)
        formal_charges = torch.tensor(formal_charges, dtype=torch.long)
        ring_info = torch.tensor(ring_info, dtype=torch.long)
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
            bond_types = torch.tensor(bond_types, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            bond_types = torch.zeros((0,), dtype=torch.long)
        
        # Compute pharmacophore features
        pharmacophore_features = _compute_pharmacophore_features(mol, num_atoms)
        
        # Compute molecular descriptors
        molecular_descriptors = _compute_molecular_descriptors(mol, num_atoms)
        
        # Generate 3D coordinates (simplified - would use conformer generation in practice)
        coordinates = _generate_mock_coordinates(num_atoms)
        
        return {
            'atom_types': atom_types,
            'edge_index': edge_index,
            'bond_types': bond_types,
            'hybridization': hybridizations,
            'formal_charges': formal_charges,
            'ring_info': ring_info,
            'pharmacophore_features': pharmacophore_features,
            'molecular_descriptors': molecular_descriptors,
            'coordinates': coordinates,
            'num_atoms': num_atoms,
            'smiles': Chem.MolToSmiles(mol) if RDKIT_AVAILABLE else "unknown"
        }
    
    except Exception as e:
        logger.error(f"Error converting molecule to graph: {e}")
        return _mock_ligand_graph()


def _mock_ligand_graph() -> Dict[str, torch.Tensor]:
    """Create a mock ligand graph for testing when RDKit is not available."""
    
    num_atoms = 10  # Mock small molecule
    
    return {
        'atom_types': torch.randint(0, 10, (num_atoms,)),
        'edge_index': torch.randint(0, num_atoms, (2, num_atoms * 2)),
        'bond_types': torch.randint(0, 4, (num_atoms * 2,)),
        'hybridization': torch.randint(0, 4, (num_atoms,)),
        'formal_charges': torch.full((num_atoms,), 5),  # Neutral charges
        'ring_info': torch.randint(0, 3, (num_atoms,)),
        'pharmacophore_features': torch.randn(num_atoms, 8),
        'molecular_descriptors': torch.randn(num_atoms, 6),
        'coordinates': torch.randn(num_atoms, 3),
        'num_atoms': num_atoms,
        'smiles': "mock_molecule"
    }


def _get_atom_type_index(symbol: str) -> int:
    """Get atom type index for symbol."""
    atom_types = {
        'UNK': 0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'P': 6, 'S': 7, 'Cl': 8,
        'Br': 9, 'I': 10, 'B': 11, 'Si': 12, 'Se': 13, 'As': 14, 'Fe': 15,
        'Zn': 16, 'Ca': 17, 'Mg': 18, 'Na': 19, 'K': 20, 'Mn': 21, 'Cu': 22
    }
    return atom_types.get(symbol, 0)  # 0 for unknown


def _get_hybridization_index(hybridization) -> int:
    """Get hybridization index."""
    if not RDKIT_AVAILABLE:
        return 2  # Default to SP3
    
    from rdkit.Chem.rdchem import HybridizationType
    
    hyb_map = {
        HybridizationType.SP: 0,
        HybridizationType.SP2: 1,
        HybridizationType.SP3: 2,
        HybridizationType.SP3D: 3,
        HybridizationType.SP3D2: 4
    }
    return hyb_map.get(hybridization, 2)  # Default to SP3


def _get_bond_type_index(bond_type) -> int:
    """Get bond type index."""
    if not RDKIT_AVAILABLE:
        return 0  # Default to single
    
    from rdkit.Chem.rdchem import BondType
    
    bond_map = {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
        BondType.AROMATIC: 3
    }
    return bond_map.get(bond_type, 4)  # 4 for other


def _compute_pharmacophore_features(mol, num_atoms: int) -> torch.Tensor:
    """Compute pharmacophore features for each atom."""
    
    if not RDKIT_AVAILABLE:
        return torch.randn(num_atoms, 8)
    
    try:
        features = []
        
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            
            # Basic pharmacophore features
            feat = [
                float(atom.GetIsAromatic()),  # Aromatic
                float(atom.GetTotalDegree() > 2),  # High connectivity
                float(atom.GetSymbol() in ['N', 'O']),  # H-bond acceptor
                float(atom.GetSymbol() in ['N', 'O'] and atom.GetTotalNumHs() > 0),  # H-bond donor
                float(atom.GetSymbol() in ['C'] and atom.GetIsAromatic()),  # Hydrophobic aromatic
                float(atom.GetSymbol() in ['C'] and not atom.GetIsAromatic()),  # Hydrophobic aliphatic
                float(atom.GetFormalCharge() > 0),  # Positive charge
                float(atom.GetFormalCharge() < 0)   # Negative charge
            ]
            
            features.append(feat)
        
        return torch.tensor(features, dtype=torch.float32)
    
    except Exception as e:
        logger.warning(f"Error computing pharmacophore features: {e}")
        return torch.randn(num_atoms, 8)


def _compute_molecular_descriptors(mol, num_atoms: int) -> torch.Tensor:
    """Compute molecular descriptors for each atom."""
    
    if not RDKIT_AVAILABLE:
        return torch.randn(num_atoms, 6)
    
    try:
        # Global molecular descriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotatable = Descriptors.NumRotatableBonds(mol)
        
        # Normalize descriptors
        descriptors = [
            mw / 500.0,  # Normalize by typical drug MW
            (logp + 5) / 10.0,  # Normalize LogP to [0, 1]
            tpsa / 200.0,  # Normalize TPSA
            hbd / 10.0,  # Normalize H-bond donors
            hba / 10.0,  # Normalize H-bond acceptors
            rotatable / 20.0  # Normalize rotatable bonds
        ]
        
        # Broadcast to all atoms
        descriptors_tensor = torch.tensor(descriptors, dtype=torch.float32)
        descriptors_broadcast = descriptors_tensor.unsqueeze(0).expand(num_atoms, -1)
        
        return descriptors_broadcast
    
    except Exception as e:
        logger.warning(f"Error computing molecular descriptors: {e}")
        return torch.randn(num_atoms, 6)


def _generate_mock_coordinates(num_atoms: int) -> torch.Tensor:
    """Generate mock 3D coordinates for atoms."""
    # In practice, would use conformer generation or provided coordinates
    return torch.randn(num_atoms, 3) * 5.0  # Scale to reasonable size


def compute_ligand_protein_distances(ligand_coords: torch.Tensor,
                                   protein_coords: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise distances between ligand atoms and protein residues.
    
    Args:
        ligand_coords: Ligand coordinates [num_atoms, 3]
        protein_coords: Protein coordinates [seq_len, 3]
        
    Returns:
        Distance matrix [seq_len, num_atoms]
    """
    
    # Expand dimensions for broadcasting
    protein_expanded = protein_coords.unsqueeze(1)  # [seq_len, 1, 3]
    ligand_expanded = ligand_coords.unsqueeze(0)    # [1, num_atoms, 3]
    
    # Compute distances
    distances = torch.norm(protein_expanded - ligand_expanded, dim=-1)  # [seq_len, num_atoms]
    
    return distances


def get_binding_pocket_mask(ligand_coords: torch.Tensor,
                          protein_coords: torch.Tensor,
                          cutoff: float = 8.0) -> torch.Tensor:
    """
    Get binding pocket mask based on distance cutoff.
    
    Args:
        ligand_coords: Ligand coordinates [num_atoms, 3]
        protein_coords: Protein coordinates [seq_len, 3]
        cutoff: Distance cutoff in Angstroms
        
    Returns:
        Boolean mask [seq_len] indicating binding pocket residues
    """
    
    distances = compute_ligand_protein_distances(ligand_coords, protein_coords)
    min_distances = distances.min(dim=-1)[0]  # [seq_len]
    
    pocket_mask = min_distances <= cutoff
    
    return pocket_mask


def batch_process_ligands(smiles_list: List[str],
                        add_hydrogens: bool = True) -> Dict[str, torch.Tensor]:
    """
    Process multiple ligands in batch.
    
    Args:
        smiles_list: List of SMILES strings
        add_hydrogens: Whether to add explicit hydrogens
        
    Returns:
        Batched ligand data
    """
    
    ligand_graphs = []
    max_atoms = 0
    
    # Process each ligand
    for smiles in smiles_list:
        graph = smiles_to_graph(smiles, add_hydrogens)
        ligand_graphs.append(graph)
        max_atoms = max(max_atoms, graph['num_atoms'])
    
    # Pad to same size
    batch_size = len(ligand_graphs)
    
    # Initialize batched tensors
    batched_data = {
        'atom_types': torch.zeros(batch_size, max_atoms, dtype=torch.long),
        'hybridization': torch.zeros(batch_size, max_atoms, dtype=torch.long),
        'formal_charges': torch.full((batch_size, max_atoms), 5, dtype=torch.long),
        'ring_info': torch.zeros(batch_size, max_atoms, dtype=torch.long),
        'pharmacophore_features': torch.zeros(batch_size, max_atoms, 8),
        'molecular_descriptors': torch.zeros(batch_size, max_atoms, 6),
        'coordinates': torch.zeros(batch_size, max_atoms, 3),
        'ligand_mask': torch.zeros(batch_size, max_atoms, dtype=torch.bool),
        'num_atoms': torch.zeros(batch_size, dtype=torch.long)
    }
    
    # Fill batched data
    for i, graph in enumerate(ligand_graphs):
        num_atoms = graph['num_atoms']
        
        batched_data['atom_types'][i, :num_atoms] = graph['atom_types']
        batched_data['hybridization'][i, :num_atoms] = graph['hybridization']
        batched_data['formal_charges'][i, :num_atoms] = graph['formal_charges']
        batched_data['ring_info'][i, :num_atoms] = graph['ring_info']
        batched_data['pharmacophore_features'][i, :num_atoms] = graph['pharmacophore_features']
        batched_data['molecular_descriptors'][i, :num_atoms] = graph['molecular_descriptors']
        batched_data['coordinates'][i, :num_atoms] = graph['coordinates']
        batched_data['ligand_mask'][i, :num_atoms] = True
        batched_data['num_atoms'][i] = num_atoms
    
    return batched_data


def validate_ligand_data(ligand_data: Dict[str, torch.Tensor]) -> bool:
    """
    Validate ligand data dictionary.
    
    Args:
        ligand_data: Ligand data dictionary
        
    Returns:
        True if valid, False otherwise
    """
    
    required_keys = [
        'atom_types', 'edge_index', 'bond_types', 'ring_info',
        'pharmacophore_features', 'molecular_descriptors', 'coordinates'
    ]
    
    # Check required keys
    for key in required_keys:
        if key not in ligand_data:
            logger.error(f"Missing required key: {key}")
            return False
    
    # Check tensor shapes
    num_atoms = ligand_data.get('num_atoms', ligand_data['atom_types'].shape[-1])
    
    if ligand_data['atom_types'].shape[-1] != num_atoms:
        logger.error("Inconsistent number of atoms")
        return False
    
    if ligand_data['pharmacophore_features'].shape[-2:] != (num_atoms, 8):
        logger.error("Invalid pharmacophore features shape")
        return False
    
    if ligand_data['molecular_descriptors'].shape[-2:] != (num_atoms, 6):
        logger.error("Invalid molecular descriptors shape")
        return False
    
    if ligand_data['coordinates'].shape[-2:] != (num_atoms, 3):
        logger.error("Invalid coordinates shape")
        return False
    
    return True
