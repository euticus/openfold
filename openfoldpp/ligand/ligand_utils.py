"""
Ligand Utilities for OdinFold

Utility functions for ligand processing, file parsing, and binding pocket analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logger = logging.getLogger(__name__)


def parse_ligand_input(ligand_input: Union[str, Path, Dict]) -> Optional[Dict[str, torch.Tensor]]:
    """
    Parse various ligand input formats into standardized tensor format.
    
    Args:
        ligand_input: Can be:
            - SMILES string
            - Path to MOL2/SDF file
            - Dictionary with pre-computed features
            
    Returns:
        Standardized ligand data dictionary or None if parsing fails
    """
    
    if isinstance(ligand_input, dict):
        # Already processed
        return ligand_input
    
    elif isinstance(ligand_input, str):
        if ligand_input.endswith(('.mol2', '.sdf', '.mol')):
            # File path
            return parse_ligand_file(ligand_input)
        else:
            # Assume SMILES string
            return parse_smiles(ligand_input)
    
    elif isinstance(ligand_input, Path):
        return parse_ligand_file(str(ligand_input))
    
    else:
        logger.error(f"Unsupported ligand input type: {type(ligand_input)}")
        return None


def parse_smiles(smiles: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    Parse SMILES string into ligand features.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Ligand data dictionary or None if parsing fails
    """
    
    if not RDKIT_AVAILABLE:
        logger.warning("RDKit not available. Creating mock ligand data.")
        return create_mock_ligand_from_smiles(smiles)
    
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
        ligand_data = extract_mol_features(mol)
        ligand_data['smiles'] = smiles
        
        return ligand_data
        
    except Exception as e:
        logger.error(f"Error processing SMILES {smiles}: {e}")
        return None


def parse_ligand_file(file_path: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    Parse ligand file (MOL2, SDF, MOL) into features.
    
    Args:
        file_path: Path to ligand file
        
    Returns:
        Ligand data dictionary or None if parsing fails
    """
    
    if not RDKIT_AVAILABLE:
        logger.warning("RDKit not available. Creating mock ligand data.")
        return create_mock_ligand_from_file(file_path)
    
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Ligand file not found: {file_path}")
            return None
        
        # Parse based on file extension
        if file_path.suffix.lower() == '.mol2':
            mol = Chem.MolFromMol2File(str(file_path))
        elif file_path.suffix.lower() in ['.sdf', '.mol']:
            supplier = Chem.SDMolSupplier(str(file_path))
            mol = next(supplier, None)
        else:
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return None
        
        if mol is None:
            logger.error(f"Failed to parse ligand file: {file_path}")
            return None
        
        # Extract features
        ligand_data = extract_mol_features(mol)
        ligand_data['file_path'] = str(file_path)
        
        return ligand_data
        
    except Exception as e:
        logger.error(f"Error processing ligand file {file_path}: {e}")
        return None


def extract_mol_features(mol) -> Dict[str, torch.Tensor]:
    """
    Extract features from RDKit molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary with atom features and positions
    """
    
    num_atoms = mol.GetNumAtoms()
    
    # Initialize feature arrays
    atom_features = []
    positions = []
    
    # Get conformer for 3D coordinates
    conf = mol.GetConformer(0)
    
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
    atom_mask = torch.ones(num_atoms, dtype=torch.float32)
    
    return {
        'atom_features': atom_features,
        'positions': positions,
        'atom_mask': atom_mask,
        'num_atoms': num_atoms
    }


def calculate_binding_pocket(protein_coords: torch.Tensor, 
                           ligand_coords: torch.Tensor,
                           cutoff: float = 5.0) -> Dict[str, torch.Tensor]:
    """
    Calculate binding pocket residues based on distance to ligand.
    
    Args:
        protein_coords: Protein coordinates [seq_len, 3]
        ligand_coords: Ligand coordinates [num_atoms, 3]
        cutoff: Distance cutoff for binding pocket definition
        
    Returns:
        Dictionary with binding pocket information
    """
    
    # Compute pairwise distances
    protein_expanded = protein_coords.unsqueeze(1)  # [seq_len, 1, 3]
    ligand_expanded = ligand_coords.unsqueeze(0)    # [1, num_atoms, 3]
    
    distances = torch.norm(protein_expanded - ligand_expanded, dim=-1)
    # distances: [seq_len, num_atoms]
    
    # Find minimum distance for each residue
    min_distances, closest_atoms = distances.min(dim=-1)
    
    # Identify binding pocket residues
    pocket_residues = min_distances <= cutoff
    pocket_indices = torch.where(pocket_residues)[0]
    
    # Calculate pocket properties
    pocket_center = protein_coords[pocket_residues].mean(dim=0) if pocket_residues.any() else torch.zeros(3)
    pocket_size = pocket_residues.sum().item()
    
    return {
        'pocket_residues': pocket_residues,
        'pocket_indices': pocket_indices,
        'min_distances': min_distances,
        'closest_atoms': closest_atoms,
        'pocket_center': pocket_center,
        'pocket_size': pocket_size
    }


def create_mock_ligand_from_smiles(smiles: str, num_atoms: int = 20) -> Dict[str, torch.Tensor]:
    """
    Create mock ligand data from SMILES when RDKit is not available.
    
    Args:
        smiles: SMILES string (used for reproducible random seed)
        num_atoms: Number of atoms in mock ligand
        
    Returns:
        Mock ligand data dictionary
    """
    
    # Use SMILES hash for reproducible randomness
    seed = hash(smiles) % (2**32)
    torch.manual_seed(seed)
    
    # Mock atom features
    atom_features = torch.randint(0, 10, (num_atoms, 7)).float()
    atom_features[:, 0] = torch.randint(1, 20, (num_atoms,))  # Realistic atomic numbers
    
    # Mock 3D positions (small molecule-like)
    positions = torch.randn(num_atoms, 3) * 3.0
    
    # All atoms are valid
    atom_mask = torch.ones(num_atoms)
    
    return {
        'atom_features': atom_features,
        'positions': positions,
        'atom_mask': atom_mask,
        'smiles': smiles,
        'num_atoms': num_atoms
    }


def create_mock_ligand_from_file(file_path: str, num_atoms: int = 25) -> Dict[str, torch.Tensor]:
    """
    Create mock ligand data from file path when RDKit is not available.
    
    Args:
        file_path: File path (used for reproducible random seed)
        num_atoms: Number of atoms in mock ligand
        
    Returns:
        Mock ligand data dictionary
    """
    
    # Use file path hash for reproducible randomness
    seed = hash(file_path) % (2**32)
    torch.manual_seed(seed)
    
    # Mock atom features
    atom_features = torch.randint(0, 10, (num_atoms, 7)).float()
    atom_features[:, 0] = torch.randint(1, 20, (num_atoms,))  # Realistic atomic numbers
    
    # Mock 3D positions (small molecule-like)
    positions = torch.randn(num_atoms, 3) * 4.0
    
    # All atoms are valid
    atom_mask = torch.ones(num_atoms)
    
    return {
        'atom_features': atom_features,
        'positions': positions,
        'atom_mask': atom_mask,
        'file_path': file_path,
        'num_atoms': num_atoms
    }


def validate_ligand_data(ligand_data: Dict[str, torch.Tensor]) -> Dict[str, bool]:
    """
    Validate ligand data dictionary.
    
    Args:
        ligand_data: Ligand data dictionary
        
    Returns:
        Validation results
    """
    
    validation = {
        'has_atom_features': False,
        'has_positions': False,
        'has_atom_mask': False,
        'consistent_shapes': False,
        'reasonable_size': False,
        'valid_coordinates': False,
        'overall_valid': False
    }
    
    try:
        # Check required fields
        validation['has_atom_features'] = 'atom_features' in ligand_data
        validation['has_positions'] = 'positions' in ligand_data
        validation['has_atom_mask'] = 'atom_mask' in ligand_data
        
        if not all([validation['has_atom_features'], validation['has_positions']]):
            return validation
        
        atom_features = ligand_data['atom_features']
        positions = ligand_data['positions']
        atom_mask = ligand_data.get('atom_mask')
        
        # Check shapes
        num_atoms_features = atom_features.shape[0]
        num_atoms_positions = positions.shape[0]
        
        validation['consistent_shapes'] = (
            num_atoms_features == num_atoms_positions and
            positions.shape[1] == 3 and
            atom_features.shape[1] >= 7
        )
        
        if atom_mask is not None:
            validation['consistent_shapes'] = (
                validation['consistent_shapes'] and
                atom_mask.shape[0] == num_atoms_features
            )
        
        # Check reasonable size
        validation['reasonable_size'] = 5 <= num_atoms_features <= 200
        
        # Check coordinate validity
        validation['valid_coordinates'] = (
            torch.isfinite(positions).all() and
            positions.abs().max() < 100  # Reasonable coordinate range
        )
        
        # Overall validation
        validation['overall_valid'] = all([
            validation['has_atom_features'],
            validation['has_positions'],
            validation['consistent_shapes'],
            validation['reasonable_size'],
            validation['valid_coordinates']
        ])
        
    except Exception as e:
        logger.error(f"Error validating ligand data: {e}")
    
    return validation
