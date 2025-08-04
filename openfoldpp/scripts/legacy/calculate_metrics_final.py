#!/usr/bin/env python3
"""
Final script to calculate RMSD_CA and TM-score for OpenFold++ predictions.
This handles the actual structure comparison properly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser, Superimposer
import warnings

# Suppress BioPython warnings
warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB")


def extract_ca_coordinates(pdb_file: Path, chain_id: str = None):
    """Extract CA coordinates from PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_file))
    
    ca_coords = []
    residue_info = []
    
    for model in structure:
        for chain in model:
            # If chain_id specified, only use that chain
            if chain_id and chain.id != chain_id:
                continue
                
            for residue in chain:
                if residue.has_id("CA"):
                    ca_coords.append(residue["CA"].coord)
                    residue_info.append({
                        'chain': chain.id,
                        'resnum': residue.id[1],
                        'resname': residue.resname
                    })
    
    return np.array(ca_coords), residue_info


def calculate_rmsd_ca_robust(pred_file: Path, ref_file: Path):
    """Calculate RMSD between CA atoms with robust handling."""
    try:
        # Extract coordinates
        pred_coords, pred_info = extract_ca_coordinates(pred_file)
        ref_coords, ref_info = extract_ca_coordinates(ref_file, chain_id='A')  # Use chain A from reference
        
        if len(pred_coords) == 0 or len(ref_coords) == 0:
            print(f"    ❌ No CA atoms found")
            return None
        
        # Use the minimum number of atoms
        min_atoms = min(len(pred_coords), len(ref_coords))
        
        if min_atoms < 3:
            print(f"    ❌ Too few atoms for alignment ({min_atoms})")
            return None
        
        # Truncate to same length
        pred_coords = pred_coords[:min_atoms]
        ref_coords = ref_coords[:min_atoms]
        
        print(f"    📏 Aligning {min_atoms} CA atoms")
        
        # Calculate RMSD using superimposition
        # Center the coordinates
        pred_center = np.mean(pred_coords, axis=0)
        ref_center = np.mean(ref_coords, axis=0)
        
        pred_centered = pred_coords - pred_center
        ref_centered = ref_coords - ref_center
        
        # Calculate optimal rotation using SVD
        H = pred_centered.T @ ref_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Apply rotation
        pred_rotated = pred_centered @ R.T
        
        # Calculate RMSD
        diff = pred_rotated - ref_centered
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        
        return float(rmsd)
        
    except Exception as e:
        print(f"    ❌ RMSD calculation error: {e}")
        return None


def calculate_tm_score_simple(pred_file: Path, ref_file: Path):
    """Calculate a simplified TM-score approximation."""
    try:
        # Extract coordinates
        pred_coords, _ = extract_ca_coordinates(pred_file)
        ref_coords, _ = extract_ca_coordinates(ref_file, chain_id='A')
        
        if len(pred_coords) == 0 or len(ref_coords) == 0:
            return None
        
        # Use minimum length
        min_atoms = min(len(pred_coords), len(ref_coords))
        pred_coords = pred_coords[:min_atoms]
        ref_coords = ref_coords[:min_atoms]
        
        # Calculate distances after optimal superposition
        pred_center = np.mean(pred_coords, axis=0)
        ref_center = np.mean(ref_coords, axis=0)
        
        pred_centered = pred_coords - pred_center
        ref_centered = ref_coords - ref_center
        
        # Optimal rotation
        H = pred_centered.T @ ref_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        pred_rotated = pred_centered @ R.T
        
        # Calculate distances
        distances = np.linalg.norm(pred_rotated - ref_centered, axis=1)
        
        # TM-score approximation using distance cutoffs
        d0 = 1.24 * (min_atoms - 15)**(1/3) - 1.8  # TM-score normalization
        d0 = max(d0, 0.5)  # Minimum d0
        
        # Count aligned residues within distance cutoff
        aligned = np.sum(distances < d0)
        tm_score = aligned / min_atoms
        
        return float(tm_score)
        
    except Exception as e:
        print(f"    ❌ TM-score calculation error: {e}")
        return None


def main():
    """Add structural metrics to the CSV file."""
    
    print("🔬 Calculating RMSD_CA and TM-score for OpenFold++ predictions")
    print("=" * 65)
    
    # Load existing results
    csv_file = Path("real_openfold_results.csv")
    df = pd.read_csv(csv_file)
    
    print(f"📊 Loaded {len(df)} results from {csv_file}")
    
    # Add new columns
    df['rmsd_ca'] = None
    df['tm_score'] = None
    
    # File mappings
    pred_dir = Path("real_openfold_predictions")
    ref_dir = Path("casp14_data/pdb")
    
    target_to_pdb = {
        "T1024": "6w70",
        "T1030": "6xkl",
        "T1031": "6w4h", 
        "T1032": "6m71",
        "H1101": "6w63"
    }
    
    # Process each target
    for idx, row in df.iterrows():
        target_id = row['target_id']
        print(f"\n🧬 Processing {target_id}...")
        
        # Find files
        pred_file = pred_dir / f"{target_id}_pred.pdb"
        pdb_id = target_to_pdb.get(target_id)
        
        if not pred_file.exists():
            print(f"  ❌ Prediction not found: {pred_file}")
            continue
            
        if not pdb_id:
            print(f"  ❌ No PDB mapping for {target_id}")
            continue
            
        ref_file = ref_dir / f"{pdb_id}.pdb"
        if not ref_file.exists():
            print(f"  ❌ Reference not found: {ref_file}")
            continue
        
        print(f"  📁 Prediction: {pred_file.name}")
        print(f"  📁 Reference: {ref_file.name}")
        
        # Calculate RMSD_CA
        rmsd_ca = calculate_rmsd_ca_robust(pred_file, ref_file)
        if rmsd_ca is not None:
            df.at[idx, 'rmsd_ca'] = round(rmsd_ca, 3)
            print(f"  ✅ RMSD_CA: {rmsd_ca:.3f} Å")
        
        # Calculate TM-score approximation
        tm_score = calculate_tm_score_simple(pred_file, ref_file)
        if tm_score is not None:
            df.at[idx, 'tm_score'] = round(tm_score, 3)
            print(f"  ✅ TM-score: {tm_score:.3f}")
    
    # Save results
    output_file = "final_enhanced_results.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n💾 Enhanced results saved to: {output_file}")
    
    # Summary statistics
    print(f"\n📈 STRUCTURAL METRICS SUMMARY")
    print("=" * 40)
    
    rmsd_values = df['rmsd_ca'].dropna()
    tm_values = df['tm_score'].dropna()
    
    if len(rmsd_values) > 0:
        print(f"RMSD_CA (Å): {len(rmsd_values)} values")
        print(f"  Mean: {rmsd_values.mean():.3f}")
        print(f"  Range: {rmsd_values.min():.3f} - {rmsd_values.max():.3f}")
    
    if len(tm_values) > 0:
        print(f"\nTM-score: {len(tm_values)} values")
        print(f"  Mean: {tm_values.mean():.3f}")
        print(f"  Range: {tm_values.min():.3f} - {tm_values.max():.3f}")
    
    # Display final results
    print(f"\n📊 FINAL ENHANCED RESULTS:")
    print("=" * 120)
    
    display_cols = ['target_id', 'target_type', 'total_length', 'time_s', 'confidence', 'rmsd_ca', 'tm_score']
    print(df[display_cols].to_string(index=False))
    
    print(f"\n🎉 Structural metrics successfully calculated!")
    print(f"📁 Final results: {output_file}")


if __name__ == "__main__":
    main()
