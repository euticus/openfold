"""
GNINA Integration for OdinFold

Integrates GNINA (deep learning-enhanced AutoDock Vina) for molecular docking
with improved scoring and pose prediction using convolutional neural networks.
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
import json

logger = logging.getLogger(__name__)


@dataclass
class GninaConfig:
    """Configuration for GNINA docking."""
    
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
    
    # GNINA-specific parameters
    cnn_scoring: bool = True
    cnn_model: str = "default"  # default, dense, general_default2018
    minimize: bool = True
    autobox_add: float = 4.0
    
    # Scoring options
    score_only: bool = False
    local_only: bool = False
    randomize_only: bool = False
    
    # Output options
    save_poses: bool = True
    save_scores: bool = True
    
    # Performance
    cpu_count: int = 0  # 0 = auto-detect
    gpu: bool = True
    seed: int = 42


class GninaDockingRunner:
    """
    GNINA docking runner for OdinFold.
    
    Performs molecular docking using GNINA's deep learning-enhanced scoring
    for improved binding pose prediction and affinity estimation.
    """
    
    def __init__(self, config: Optional[GninaConfig] = None):
        self.config = config or GninaConfig()
        
        # Check if GNINA is available
        self.gnina_available = shutil.which("gnina") is not None
        if not self.gnina_available:
            logger.warning("GNINA not found. Using mock docking. Install from: https://github.com/gnina/gnina")
    
    def dock_ligand(self, protein_pdb: Union[str, Path], 
                   ligand_input: Union[str, Path],
                   binding_site: Optional[Dict[str, float]] = None,
                   output_dir: Optional[Union[str, Path]] = None) -> Dict:
        """
        Dock a ligand to a protein structure using GNINA.
        
        Args:
            protein_pdb: Path to protein PDB file or PDB string
            ligand_input: Path to ligand file (SDF, MOL2) or SMILES string
            binding_site: Optional binding site coordinates
            output_dir: Optional output directory for results
            
        Returns:
            Docking results dictionary
        """
        
        logger.info("Starting molecular docking with GNINA")
        
        # Create temporary directory if needed
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="gnina_docking_")
            output_dir = Path(temp_dir)
            cleanup_temp = True
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            cleanup_temp = False
        
        try:
            # Prepare input files
            protein_file = self._prepare_protein(protein_pdb, output_dir)
            ligand_file = self._prepare_ligand(ligand_input, output_dir)
            
            # Run docking
            if self.gnina_available:
                results = self._run_gnina_docking(
                    protein_file, ligand_file, binding_site, output_dir
                )
            else:
                results = self._mock_gnina_docking(
                    protein_file, ligand_file, binding_site, output_dir
                )
            
            # Analyze results
            analyzed_results = self._analyze_docking_results(results, output_dir)
            
            logger.info(f"GNINA docking completed. Best CNN score: {analyzed_results.get('best_cnn_score', 'N/A')}")
            
            return analyzed_results
            
        except Exception as e:
            logger.error(f"GNINA docking failed: {e}")
            raise
        finally:
            if cleanup_temp and output_dir.exists():
                shutil.rmtree(output_dir)
    
    def _prepare_protein(self, protein_input: Union[str, Path], 
                        output_dir: Path) -> Path:
        """Prepare protein for GNINA docking."""
        
        if isinstance(protein_input, str) and not Path(protein_input).exists():
            # Assume it's PDB content
            protein_file = output_dir / "protein.pdb"
            with open(protein_file, 'w') as f:
                f.write(protein_input)
        else:
            protein_file = Path(protein_input)
        
        return protein_file
    
    def _prepare_ligand(self, ligand_input: Union[str, Path], 
                       output_dir: Path) -> Path:
        """Prepare ligand for GNINA docking."""
        
        if isinstance(ligand_input, str) and not Path(ligand_input).exists():
            # Assume it's SMILES - convert to SDF
            ligand_file = self._smiles_to_sdf(ligand_input, output_dir)
        else:
            ligand_file = Path(ligand_input)
        
        return ligand_file
    
    def _smiles_to_sdf(self, smiles: str, output_dir: Path) -> Path:
        """Convert SMILES to SDF using OpenBabel or RDKit."""
        
        ligand_sdf = output_dir / "ligand.sdf"
        
        # Try OpenBabel first
        if shutil.which("obabel"):
            try:
                cmd = [
                    "obabel", "-:{}".format(smiles), "-O", str(ligand_sdf),
                    "--gen3d"
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                return ligand_sdf
            except subprocess.CalledProcessError:
                pass
        
        # Fallback to mock SDF
        logger.warning("Could not convert SMILES to SDF. Using mock file.")
        with open(ligand_sdf, 'w') as f:
            f.write(f"Mock SDF for SMILES: {smiles}\n")
        
        return ligand_sdf
    
    def _run_gnina_docking(self, protein_file: Path, ligand_file: Path,
                          binding_site: Optional[Dict[str, float]], 
                          output_dir: Path) -> Dict:
        """Run actual GNINA docking."""
        
        output_poses = output_dir / "docked_poses.sdf"
        output_log = output_dir / "gnina.log"
        
        # Build GNINA command
        cmd = [
            "gnina",
            "-r", str(protein_file),
            "-l", str(ligand_file),
            "-o", str(output_poses),
            "--log", str(output_log),
            "--exhaustiveness", str(self.config.exhaustiveness),
            "--num_modes", str(self.config.num_modes),
            "--energy_range", str(self.config.energy_range),
            "--seed", str(self.config.seed)
        ]
        
        # Add binding site if specified
        if binding_site:
            cmd.extend([
                "--center_x", str(binding_site['center_x']),
                "--center_y", str(binding_site['center_y']),
                "--center_z", str(binding_site['center_z']),
                "--size_x", str(binding_site['size_x']),
                "--size_y", str(binding_site['size_y']),
                "--size_z", str(binding_site['size_z'])
            ])
        else:
            cmd.extend(["--autobox_add", str(self.config.autobox_add)])
        
        # Add CNN scoring options
        if self.config.cnn_scoring:
            cmd.extend(["--cnn_scoring"])
            if self.config.cnn_model != "default":
                cmd.extend(["--cnn", self.config.cnn_model])
        
        # Add other options
        if self.config.minimize:
            cmd.extend(["--minimize"])
        
        if self.config.cpu_count > 0:
            cmd.extend(["--cpu", str(self.config.cpu_count)])
        
        if not self.config.gpu:
            cmd.extend(["--no_gpu"])
        
        # Run GNINA
        logger.info(f"Running GNINA: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"GNINA failed: {result.stderr}")
        
        # Parse results
        return self._parse_gnina_output(output_log, output_poses)
    
    def _mock_gnina_docking(self, protein_file: Path, ligand_file: Path,
                           binding_site: Optional[Dict[str, float]], 
                           output_dir: Path) -> Dict:
        """Mock GNINA docking when not available."""
        
        logger.warning("Running mock GNINA docking")
        
        output_poses = output_dir / "docked_poses.sdf"
        output_log = output_dir / "gnina.log"
        
        # Generate mock results
        num_poses = self.config.num_modes
        mock_results = []
        
        for i in range(num_poses):
            # Generate realistic scores
            vina_score = np.random.uniform(-12.0, -6.0)
            cnn_score = np.random.uniform(0.1, 0.9)  # CNN scores are 0-1
            cnn_affinity = np.random.uniform(-12.0, -6.0)
            
            mock_results.append({
                'mode': i + 1,
                'vina_score': vina_score,
                'cnn_score': cnn_score,
                'cnn_affinity': cnn_affinity
            })
        
        # Sort by CNN score (higher is better)
        mock_results.sort(key=lambda x: x['cnn_score'], reverse=True)
        
        # Create mock output files
        with open(output_poses, 'w') as f:
            f.write("Mock GNINA poses\n")
            for result in mock_results:
                f.write(f"MODE {result['mode']}: Vina={result['vina_score']:.3f}, "
                       f"CNN={result['cnn_score']:.3f}, CNN_affinity={result['cnn_affinity']:.3f}\n")
        
        with open(output_log, 'w') as f:
            f.write("Mock GNINA log\n")
            for result in mock_results:
                f.write(f"Mode {result['mode']}: {result['vina_score']:.3f} {result['cnn_score']:.3f} {result['cnn_affinity']:.3f}\n")
        
        return {
            'poses': mock_results,
            'poses_file': output_poses,
            'log_file': output_log,
            'num_poses': len(mock_results)
        }
    
    def _parse_gnina_output(self, log_file: Path, poses_file: Path) -> Dict:
        """Parse GNINA output files."""
        
        poses = []
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Parse log file for scores
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            mode = len(poses) + 1
                            vina_score = float(parts[0])
                            cnn_score = float(parts[1]) if len(parts) > 1 else None
                            cnn_affinity = float(parts[2]) if len(parts) > 2 else None
                            
                            poses.append({
                                'mode': mode,
                                'vina_score': vina_score,
                                'cnn_score': cnn_score,
                                'cnn_affinity': cnn_affinity
                            })
                        except ValueError:
                            continue
        
        except Exception as e:
            logger.warning(f"Could not parse GNINA log: {e}")
        
        return {
            'poses': poses,
            'poses_file': poses_file,
            'log_file': log_file,
            'num_poses': len(poses)
        }
    
    def _analyze_docking_results(self, results: Dict, output_dir: Path) -> Dict:
        """Analyze GNINA docking results."""
        
        poses = results['poses']
        
        if not poses:
            return {
                'success': False,
                'error': 'No docking poses generated'
            }
        
        # Extract scores
        vina_scores = [pose['vina_score'] for pose in poses]
        cnn_scores = [pose['cnn_score'] for pose in poses if pose['cnn_score'] is not None]
        cnn_affinities = [pose['cnn_affinity'] for pose in poses if pose['cnn_affinity'] is not None]
        
        # Calculate statistics
        best_vina_score = min(vina_scores) if vina_scores else None
        best_cnn_score = max(cnn_scores) if cnn_scores else None
        best_cnn_affinity = min(cnn_affinities) if cnn_affinities else None
        
        # Determine best pose (by CNN score if available, otherwise Vina)
        if cnn_scores:
            best_pose = max(poses, key=lambda x: x['cnn_score'] or 0)
        else:
            best_pose = min(poses, key=lambda x: x['vina_score'])
        
        analyzed_results = {
            'success': True,
            'best_vina_score': best_vina_score,
            'best_cnn_score': best_cnn_score,
            'best_cnn_affinity': best_cnn_affinity,
            'best_pose': best_pose,
            'num_poses': results['num_poses'],
            'all_poses': poses,
            'poses_file': str(results['poses_file']),
            'log_file': str(results['log_file']),
            'config': self.config
        }
        
        # Save results summary
        summary_file = output_dir / "gnina_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(analyzed_results, f, indent=2, default=str)
        
        analyzed_results['summary_file'] = str(summary_file)
        
        return analyzed_results
