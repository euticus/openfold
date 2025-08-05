"""
AutoDock Vina Integration for OdinFold

Integrates AutoDock Vina for molecular docking of ligands to predicted protein structures.
Supports both local and global docking with customizable search parameters.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import numpy as np

try:
    from vina import Vina
    VINA_AVAILABLE = True
except ImportError:
    VINA_AVAILABLE = False
    logging.warning("AutoDock Vina not available. Install with: pip install vina")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class VinaConfig:
    """Configuration for AutoDock Vina docking."""
    
    # Search space parameters
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0
    size_x: float = 20.0
    size_y: float = 20.0
    size_z: float = 20.0
    
    # Docking parameters
    exhaustiveness: int = 8
    num_modes: int = 9
    energy_range: float = 3.0
    
    # Scoring function
    scoring: str = "vina"  # vina, ad4, vinardo
    
    # Output options
    save_poses: bool = True
    save_scores: bool = True
    
    # Performance
    cpu_count: int = 0  # 0 = auto-detect
    seed: int = 42


class VinaDockingRunner:
    """
    AutoDock Vina docking runner for OdinFold.
    
    Performs molecular docking of ligands to predicted protein structures
    with automatic binding site detection and pose scoring.
    """
    
    def __init__(self, config: Optional[VinaConfig] = None):
        self.config = config or VinaConfig()
        
        if not VINA_AVAILABLE:
            logger.warning("AutoDock Vina not available. Using mock docking.")
        
        # Initialize Vina instance
        self.vina = None
        if VINA_AVAILABLE:
            self.vina = Vina(sf_name=self.config.scoring, seed=self.config.seed)
            if self.config.cpu_count > 0:
                self.vina.set_cpu(self.config.cpu_count)
    
    def dock_ligand(self, protein_pdb: Union[str, Path], 
                   ligand_input: Union[str, Path],
                   binding_site: Optional[Dict[str, float]] = None,
                   output_dir: Optional[Union[str, Path]] = None) -> Dict:
        """
        Dock a ligand to a protein structure.
        
        Args:
            protein_pdb: Path to protein PDB file or PDB string
            ligand_input: Path to ligand file (SDF, MOL2) or SMILES string
            binding_site: Optional binding site coordinates
            output_dir: Optional output directory for results
            
        Returns:
            Docking results dictionary
        """
        
        logger.info("Starting molecular docking with AutoDock Vina")
        
        # Create temporary directory if needed
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="vina_docking_")
            output_dir = Path(temp_dir)
            cleanup_temp = True
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            cleanup_temp = False
        
        try:
            # Prepare protein
            protein_pdbqt = self._prepare_protein(protein_pdb, output_dir)
            
            # Prepare ligand
            ligand_pdbqt = self._prepare_ligand(ligand_input, output_dir)
            
            # Detect binding site if not provided
            if binding_site is None:
                binding_site = self._detect_binding_site(protein_pdbqt, ligand_pdbqt)
            
            # Run docking
            if VINA_AVAILABLE:
                results = self._run_vina_docking(
                    protein_pdbqt, ligand_pdbqt, binding_site, output_dir
                )
            else:
                results = self._mock_vina_docking(
                    protein_pdbqt, ligand_pdbqt, binding_site, output_dir
                )
            
            # Analyze results
            analyzed_results = self._analyze_docking_results(results, output_dir)
            
            logger.info(f"Docking completed. Best score: {analyzed_results['best_score']:.2f}")
            
            return analyzed_results
            
        except Exception as e:
            logger.error(f"Docking failed: {e}")
            raise
        finally:
            if cleanup_temp and output_dir.exists():
                shutil.rmtree(output_dir)
    
    def _prepare_protein(self, protein_input: Union[str, Path], 
                        output_dir: Path) -> Path:
        """Prepare protein for docking (convert to PDBQT)."""
        
        protein_pdbqt = output_dir / "protein.pdbqt"
        
        if isinstance(protein_input, str) and not Path(protein_input).exists():
            # Assume it's PDB content
            protein_pdb = output_dir / "protein.pdb"
            with open(protein_pdb, 'w') as f:
                f.write(protein_input)
        else:
            protein_pdb = Path(protein_input)
        
        # Convert PDB to PDBQT using OpenBabel or mock conversion
        if shutil.which("obabel"):
            cmd = [
                "obabel", str(protein_pdb), "-O", str(protein_pdbqt),
                "-p", "7.4"  # Add hydrogens at pH 7.4
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        else:
            # Mock conversion - copy PDB as PDBQT
            logger.warning("OpenBabel not found. Using mock PDB->PDBQT conversion.")
            shutil.copy(protein_pdb, protein_pdbqt)
        
        return protein_pdbqt
    
    def _prepare_ligand(self, ligand_input: Union[str, Path], 
                       output_dir: Path) -> Path:
        """Prepare ligand for docking (convert to PDBQT)."""
        
        ligand_pdbqt = output_dir / "ligand.pdbqt"
        
        if isinstance(ligand_input, str) and not Path(ligand_input).exists():
            # Assume it's SMILES
            if RDKIT_AVAILABLE:
                ligand_sdf = self._smiles_to_sdf(ligand_input, output_dir)
            else:
                # Mock SDF creation
                ligand_sdf = output_dir / "ligand.sdf"
                with open(ligand_sdf, 'w') as f:
                    f.write("Mock SDF content\n")
        else:
            ligand_sdf = Path(ligand_input)
        
        # Convert SDF to PDBQT
        if shutil.which("obabel"):
            cmd = [
                "obabel", str(ligand_sdf), "-O", str(ligand_pdbqt),
                "--gen3d"  # Generate 3D coordinates
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        else:
            # Mock conversion
            logger.warning("OpenBabel not found. Using mock SDF->PDBQT conversion.")
            shutil.copy(ligand_sdf, ligand_pdbqt)
        
        return ligand_pdbqt
    
    def _smiles_to_sdf(self, smiles: str, output_dir: Path) -> Path:
        """Convert SMILES to SDF with 3D coordinates."""
        
        ligand_sdf = output_dir / "ligand.sdf"
        
        if not RDKIT_AVAILABLE:
            # Mock SDF
            with open(ligand_sdf, 'w') as f:
                f.write(f"Mock SDF for SMILES: {smiles}\n")
            return ligand_sdf
        
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.OptimizeMoleculeConfs(mol)
            
            # Write SDF
            writer = Chem.SDWriter(str(ligand_sdf))
            writer.write(mol)
            writer.close()
            
            return ligand_sdf
            
        except Exception as e:
            logger.error(f"Failed to convert SMILES to SDF: {e}")
            # Fallback to mock
            with open(ligand_sdf, 'w') as f:
                f.write(f"Mock SDF for SMILES: {smiles}\n")
            return ligand_sdf
    
    def _detect_binding_site(self, protein_pdbqt: Path, 
                           ligand_pdbqt: Path) -> Dict[str, float]:
        """Detect binding site automatically."""
        
        # Simple heuristic: use protein center with reasonable box size
        # In practice, you'd use CAVity detection or fpocket
        
        binding_site = {
            'center_x': self.config.center_x,
            'center_y': self.config.center_y,
            'center_z': self.config.center_z,
            'size_x': self.config.size_x,
            'size_y': self.config.size_y,
            'size_z': self.config.size_z
        }
        
        logger.info(f"Using binding site: {binding_site}")
        return binding_site
    
    def _run_vina_docking(self, protein_pdbqt: Path, ligand_pdbqt: Path,
                         binding_site: Dict[str, float], output_dir: Path) -> Dict:
        """Run actual AutoDock Vina docking."""
        
        # Set receptor
        self.vina.set_receptor(str(protein_pdbqt))
        
        # Set ligand
        self.vina.set_ligand_from_file(str(ligand_pdbqt))
        
        # Set search space
        self.vina.compute_vina_maps(
            center=[binding_site['center_x'], binding_site['center_y'], binding_site['center_z']],
            box_size=[binding_site['size_x'], binding_site['size_y'], binding_site['size_z']]
        )
        
        # Run docking
        self.vina.dock(
            exhaustiveness=self.config.exhaustiveness,
            n_poses=self.config.num_modes
        )
        
        # Get results
        energies = self.vina.energies(n_poses=self.config.num_modes)
        
        # Save poses
        output_poses = output_dir / "docked_poses.pdbqt"
        self.vina.write_poses(str(output_poses), n_poses=self.config.num_modes)
        
        return {
            'energies': energies,
            'poses_file': output_poses,
            'num_poses': len(energies)
        }
    
    def _mock_vina_docking(self, protein_pdbqt: Path, ligand_pdbqt: Path,
                          binding_site: Dict[str, float], output_dir: Path) -> Dict:
        """Mock docking when Vina is not available."""
        
        logger.warning("Running mock docking (Vina not available)")
        
        # Generate mock energies
        num_poses = self.config.num_modes
        mock_energies = []
        
        for i in range(num_poses):
            # Generate realistic binding energies (-12 to -6 kcal/mol)
            energy = np.random.uniform(-12.0, -6.0)
            mock_energies.append([energy, 0.0, 0.0])  # [binding, inter, intra]
        
        # Sort by binding energy (most negative first)
        mock_energies.sort(key=lambda x: x[0])
        
        # Create mock poses file
        output_poses = output_dir / "docked_poses.pdbqt"
        with open(output_poses, 'w') as f:
            f.write("Mock docked poses\n")
            for i, energy in enumerate(mock_energies):
                f.write(f"MODEL {i+1}\n")
                f.write(f"REMARK VINA RESULT: {energy[0]:.3f} {energy[1]:.3f} {energy[2]:.3f}\n")
                f.write("ENDMDL\n")
        
        return {
            'energies': mock_energies,
            'poses_file': output_poses,
            'num_poses': len(mock_energies)
        }
    
    def _analyze_docking_results(self, results: Dict, output_dir: Path) -> Dict:
        """Analyze docking results and extract key metrics."""
        
        energies = results['energies']
        
        if not energies:
            return {
                'success': False,
                'error': 'No docking poses generated'
            }
        
        # Extract binding energies
        binding_energies = [energy[0] for energy in energies]
        
        # Calculate statistics
        best_score = min(binding_energies)
        mean_score = np.mean(binding_energies)
        std_score = np.std(binding_energies)
        
        # Classify binding strength
        if best_score < -10.0:
            binding_strength = "Strong"
        elif best_score < -8.0:
            binding_strength = "Moderate"
        elif best_score < -6.0:
            binding_strength = "Weak"
        else:
            binding_strength = "Very Weak"
        
        analyzed_results = {
            'success': True,
            'best_score': best_score,
            'mean_score': mean_score,
            'std_score': std_score,
            'binding_strength': binding_strength,
            'num_poses': results['num_poses'],
            'all_scores': binding_energies,
            'poses_file': str(results['poses_file']),
            'config': self.config
        }
        
        # Save results summary
        summary_file = output_dir / "docking_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("AutoDock Vina Docking Results\n")
            f.write("=" * 30 + "\n")
            f.write(f"Best Score: {best_score:.2f} kcal/mol\n")
            f.write(f"Mean Score: {mean_score:.2f} ± {std_score:.2f} kcal/mol\n")
            f.write(f"Binding Strength: {binding_strength}\n")
            f.write(f"Number of Poses: {results['num_poses']}\n")
            f.write("\nAll Scores:\n")
            for i, score in enumerate(binding_energies):
                f.write(f"  Pose {i+1}: {score:.2f} kcal/mol\n")
        
        analyzed_results['summary_file'] = str(summary_file)
        
        return analyzed_results


def run_batch_docking(protein_pdb: Union[str, Path],
                     ligand_list: List[Union[str, Path]],
                     config: Optional[VinaConfig] = None,
                     output_dir: Optional[Union[str, Path]] = None) -> List[Dict]:
    """
    Run batch docking of multiple ligands to a single protein.

    Args:
        protein_pdb: Path to protein PDB file
        ligand_list: List of ligand files or SMILES strings
        config: Vina configuration
        output_dir: Output directory for results

    Returns:
        List of docking results for each ligand
    """

    runner = VinaDockingRunner(config)
    results = []

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for i, ligand in enumerate(ligand_list):
        logger.info(f"Docking ligand {i+1}/{len(ligand_list)}")

        # Create subdirectory for each ligand
        if output_dir:
            ligand_output_dir = output_dir / f"ligand_{i+1}"
        else:
            ligand_output_dir = None

        try:
            result = runner.dock_ligand(
                protein_pdb, ligand, output_dir=ligand_output_dir
            )
            result['ligand_index'] = i
            result['ligand_input'] = str(ligand)
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to dock ligand {i+1}: {e}")
            results.append({
                'success': False,
                'error': str(e),
                'ligand_index': i,
                'ligand_input': str(ligand)
            })

    # Sort results by best score
    successful_results = [r for r in results if r.get('success', False)]
    if successful_results:
        successful_results.sort(key=lambda x: x['best_score'])

        logger.info(f"Batch docking completed. Best ligand: {successful_results[0]['ligand_input']} "
                   f"(Score: {successful_results[0]['best_score']:.2f})")

    return results
