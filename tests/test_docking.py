#!/usr/bin/env python3
"""
Test Suite for OdinFold Docking Functionality

Tests molecular docking integration with AutoDock Vina and GNINA.
"""

import pytest
import torch
import numpy as np
import tempfile
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openfoldpp.docking import (
    VinaDockingRunner,
    VinaConfig,
    GninaDockingRunner,
    GninaConfig,
    prepare_protein_for_docking,
    prepare_ligand_for_docking,
    DockingPoseAnalyzer,
    calculate_binding_affinity
)
from openfoldpp.docking.docking_utils import detect_binding_site, analyze_binding_pocket
from openfoldpp.ligand.ligand_encoder import create_mock_ligand


class TestVinaConfig:
    """Test Vina configuration."""
    
    def test_default_config(self):
        """Test default Vina configuration."""
        
        config = VinaConfig()
        
        assert config.exhaustiveness == 8
        assert config.num_modes == 9
        assert config.scoring == "vina"
        assert config.seed == 42
    
    def test_custom_config(self):
        """Test custom Vina configuration."""
        
        config = VinaConfig(
            exhaustiveness=16,
            num_modes=20,
            scoring="vinardo",
            size_x=30.0
        )
        
        assert config.exhaustiveness == 16
        assert config.num_modes == 20
        assert config.scoring == "vinardo"
        assert config.size_x == 30.0


class TestVinaDockingRunner:
    """Test AutoDock Vina integration."""
    
    def test_vina_runner_init(self):
        """Test Vina runner initialization."""
        
        config = VinaConfig(exhaustiveness=4)
        runner = VinaDockingRunner(config)
        
        assert runner.config.exhaustiveness == 4
        # Vina may not be available, so just check runner exists
        assert runner is not None
    
    def test_mock_docking(self):
        """Test mock docking when Vina not available."""
        
        runner = VinaDockingRunner()
        
        # Create mock protein and ligand
        protein_coords = torch.randn(50, 3) * 10
        sequence = "A" * 50
        protein_pdb = prepare_protein_for_docking(protein_coords, sequence)
        
        ligand_smiles = "CCO"  # Ethanol
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = runner.dock_ligand(
                protein_pdb, ligand_smiles, output_dir=temp_dir
            )
            
            # Check results structure
            assert 'success' in results
            assert 'best_score' in results
            assert 'num_poses' in results
            
            if results['success']:
                assert results['best_score'] < 0  # Negative binding energy
                assert results['num_poses'] > 0


class TestGninaDockingRunner:
    """Test GNINA integration."""
    
    def test_gnina_runner_init(self):
        """Test GNINA runner initialization."""
        
        config = GninaConfig(cnn_scoring=True)
        runner = GninaDockingRunner(config)
        
        assert runner.config.cnn_scoring == True
        assert runner is not None
    
    def test_mock_gnina_docking(self):
        """Test mock GNINA docking."""
        
        runner = GninaDockingRunner()
        
        # Create mock inputs
        protein_coords = torch.randn(40, 3) * 8
        sequence = "M" * 40
        protein_pdb = prepare_protein_for_docking(protein_coords, sequence)
        
        ligand_smiles = "CC(=O)O"  # Acetic acid
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = runner.dock_ligand(
                protein_pdb, ligand_smiles, output_dir=temp_dir
            )
            
            # Check results structure
            assert 'success' in results
            
            if results['success']:
                assert 'best_vina_score' in results
                assert 'best_cnn_score' in results
                assert 'num_poses' in results


class TestDockingUtils:
    """Test docking utility functions."""
    
    def test_prepare_protein_for_docking(self):
        """Test protein preparation for docking."""
        
        coordinates = torch.randn(20, 3) * 5
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        
        pdb_content = prepare_protein_for_docking(coordinates, sequence)
        
        assert "HEADER" in pdb_content
        assert "ATOM" in pdb_content
        assert "END" in pdb_content
        assert len(pdb_content.split('\n')) > 20  # Should have multiple lines
    
    def test_prepare_ligand_for_docking(self):
        """Test ligand preparation for docking."""
        
        ligand_data = create_mock_ligand(15)
        
        mol2_content = prepare_ligand_for_docking(ligand_data)
        
        assert "@<TRIPOS>MOLECULE" in mol2_content
        assert "@<TRIPOS>ATOM" in mol2_content
        assert "ligand" in mol2_content
    
    def test_detect_binding_site(self):
        """Test binding site detection."""
        
        protein_coords = torch.randn(30, 3) * 10
        
        # Test geometric center method
        binding_site = detect_binding_site(protein_coords, method="geometric_center")
        
        assert 'center_x' in binding_site
        assert 'center_y' in binding_site
        assert 'center_z' in binding_site
        assert 'size_x' in binding_site
        assert 'size_y' in binding_site
        assert 'size_z' in binding_site
        
        # Check reasonable values
        assert 15.0 <= binding_site['size_x'] <= 25.0
        assert 15.0 <= binding_site['size_y'] <= 25.0
        assert 15.0 <= binding_site['size_z'] <= 25.0
    
    def test_detect_binding_site_ligand_based(self):
        """Test ligand-based binding site detection."""
        
        protein_coords = torch.randn(25, 3) * 8
        ligand_coords = torch.randn(10, 3) * 2
        
        binding_site = detect_binding_site(
            protein_coords, ligand_coords, method="ligand_based"
        )
        
        assert 'center_x' in binding_site
        assert binding_site['size_x'] >= 15.0
    
    def test_analyze_binding_pocket(self):
        """Test binding pocket analysis."""
        
        protein_coords = torch.randn(40, 3) * 12
        ligand_coords = torch.randn(8, 3) * 3
        
        # Move ligand close to some protein residues
        ligand_coords += protein_coords[:5].mean(dim=0)
        
        analysis = analyze_binding_pocket(protein_coords, ligand_coords)
        
        assert 'pocket_residues' in analysis
        assert 'pocket_size' in analysis
        assert 'pocket_center' in analysis
        assert 'complementarity_score' in analysis
        
        # Should find some pocket residues
        assert analysis['pocket_size'] > 0
        assert 0.0 <= analysis['complementarity_score'] <= 1.0


class TestDockingPoseAnalyzer:
    """Test docking pose analysis."""
    
    def test_pose_analyzer_init(self):
        """Test pose analyzer initialization."""
        
        analyzer = DockingPoseAnalyzer()
        
        assert analyzer.interaction_cutoffs['hydrogen_bond'] == 3.5
        assert analyzer.interaction_cutoffs['hydrophobic'] == 4.0
    
    def test_analyze_pose(self):
        """Test comprehensive pose analysis."""
        
        analyzer = DockingPoseAnalyzer()
        
        # Create mock data
        protein_coords = torch.randn(30, 3) * 8
        ligand_coords = torch.randn(12, 3) * 3
        sequence = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"
        docking_score = -8.5
        
        # Move ligand close to protein for interactions
        ligand_coords += protein_coords[:10].mean(dim=0)
        
        analysis = analyzer.analyze_pose(
            protein_coords, ligand_coords, sequence, docking_score
        )
        
        # Check analysis structure
        assert 'pose_id' in analysis
        assert 'docking_score' in analysis
        assert 'predicted_affinity' in analysis
        assert 'geometric_analysis' in analysis
        assert 'interactions' in analysis
        assert 'binding_site' in analysis
        assert 'quality_metrics' in analysis
        assert 'summary' in analysis
        
        # Check specific components
        geometric = analysis['geometric_analysis']
        assert 'protein_center' in geometric
        assert 'ligand_center' in geometric
        assert 'min_distance' in geometric
        
        interactions = analysis['interactions']
        assert 'statistics' in interactions
        assert 'total_interactions' in interactions['statistics']
        
        quality = analysis['quality_metrics']
        assert 'quality_score' in quality
        assert 0.0 <= quality['quality_score'] <= 1.0
    
    def test_calculate_binding_affinity(self):
        """Test binding affinity calculation."""
        
        # Mock docking results
        docking_results = [
            {'success': True, 'best_score': -9.2, 'best_cnn_score': 0.85},
            {'success': True, 'best_score': -8.7, 'best_cnn_score': 0.78},
            {'success': True, 'best_score': -8.1, 'best_cnn_score': 0.72},
            {'success': False, 'error': 'Failed'}
        ]
        
        affinity = calculate_binding_affinity(docking_results)
        
        assert 'consensus_vina_score' in affinity
        assert 'best_vina_score' in affinity
        assert 'estimated_affinity_kcal_mol' in affinity
        assert 'confidence' in affinity
        assert 'num_poses' in affinity
        
        # Check values
        assert affinity['best_vina_score'] == -9.2  # Best score
        assert affinity['num_poses'] == 3  # Only successful poses
        assert 0.0 <= affinity['confidence'] <= 1.0


def test_docking_integration():
    """Integration test for complete docking pipeline."""
    
    print("🧪 Testing docking integration pipeline...")
    
    # Create mock protein structure
    sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYN"
    seq_len = len(sequence)
    protein_coords = torch.randn(seq_len, 3) * 10
    
    # Prepare protein
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        protein_pdb_content = prepare_protein_for_docking(protein_coords, sequence)
        f.write(protein_pdb_content)
        protein_pdb_file = f.name

    assert "ATOM" in protein_pdb_content
    
    # Create mock ligand
    ligand_data = create_mock_ligand(20)
    ligand_mol2 = prepare_ligand_for_docking(ligand_data)
    assert "@<TRIPOS>" in ligand_mol2
    
    # Detect binding site
    binding_site = detect_binding_site(protein_coords)
    assert 'center_x' in binding_site
    
    # Run Vina docking (mock)
    vina_runner = VinaDockingRunner()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        vina_results = vina_runner.dock_ligand(
            protein_pdb_file, "CCO", output_dir=temp_dir
        )
        
        if vina_results['success']:
            # Analyze pose
            analyzer = DockingPoseAnalyzer()
            
            # Create mock ligand coordinates for analysis
            ligand_coords = torch.randn(10, 3) * 3
            
            pose_analysis = analyzer.analyze_pose(
                protein_coords, ligand_coords, sequence, 
                vina_results['best_score']
            )
            
            assert 'summary' in pose_analysis
            assert 'binding_strength' in pose_analysis['summary']

    # Clean up temporary file
    Path(protein_pdb_file).unlink()

    print("✅ Docking integration test passed!")


if __name__ == "__main__":
    # Run integration test
    test_docking_integration()
    
    # Run all tests
    pytest.main([__file__, "-v"])
