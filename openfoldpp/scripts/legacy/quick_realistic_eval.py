#!/usr/bin/env python3
"""
Quick evaluation of realistic predictions with fixed coordinate parsing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    from tmtools import tm_align
    TM_AVAILABLE = True
except ImportError:
    TM_AVAILABLE = False


def parse_pdb_coords(pdb_path):
    """Parse CA coordinates from PDB file manually."""
    coords = []
    sequence = []
    
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and " CA " in line:
                try:
                    # Extract coordinates
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    
                    # Skip invalid coordinates
                    if abs(x) > 1000 or abs(y) > 1000 or abs(z) > 1000:
                        continue
                    
                    coords.append([x, y, z])
                    
                    # Extract amino acid
                    aa = line[17:20].strip()
                    sequence.append(aa)
                    
                except (ValueError, IndexError):
                    continue
    
    return np.array(coords), sequence


def calculate_rmsd_simple(pred_coords, ref_coords):
    """Calculate RMSD with simple alignment."""
    if len(pred_coords) == 0 or len(ref_coords) == 0:
        return None
    
    # Use minimum length
    min_len = min(len(pred_coords), len(ref_coords))
    pred_coords = pred_coords[:min_len]
    ref_coords = ref_coords[:min_len]
    
    if min_len < 3:
        return None
    
    # Center coordinates
    pred_center = np.mean(pred_coords, axis=0)
    ref_center = np.mean(ref_coords, axis=0)
    
    pred_centered = pred_coords - pred_center
    ref_centered = ref_coords - ref_center
    
    # Calculate optimal rotation using Kabsch algorithm
    H = pred_centered.T @ ref_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation
    pred_rotated = pred_centered @ R.T
    
    # Calculate RMSD
    diff = pred_rotated - ref_centered
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return rmsd


def calculate_tm_simple(pred_coords, ref_coords):
    """Calculate simplified TM-score."""
    if len(pred_coords) == 0 or len(ref_coords) == 0:
        return None
    
    min_len = min(len(pred_coords), len(ref_coords))
    if min_len < 3:
        return None
    
    # Align coordinates
    pred_coords = pred_coords[:min_len]
    ref_coords = ref_coords[:min_len]
    
    # Center and align
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
    
    # TM-score calculation
    d0 = 1.24 * (min_len - 15)**(1/3) - 1.8
    d0 = max(d0, 0.5)
    
    # Score based on distance cutoff
    scores = 1.0 / (1.0 + (distances / d0)**2)
    tm_score = np.mean(scores)
    
    return tm_score


def main():
    """Quick evaluation."""
    print("🔬 Quick Realistic Prediction Evaluation")
    print("=" * 45)
    
    targets = {
        "T1024": "6w70",
        "T1030": "6xkl", 
        "T1031": "6w4h",
        "T1032": "6m71",
        "H1101": "6w63"
    }
    
    results = []
    
    for target_id, pdb_id in targets.items():
        print(f"\n🧬 {target_id}...")
        
        pred_file = Path(f"real_realistic_predictions/{target_id}_realistic.pdb")
        ref_file = Path(f"casp14_data/pdb/{pdb_id}.pdb")
        
        if not pred_file.exists() or not ref_file.exists():
            print(f"  ❌ Files not found")
            continue
        
        # Parse coordinates
        pred_coords, pred_seq = parse_pdb_coords(pred_file)
        ref_coords, ref_seq = parse_pdb_coords(ref_file)
        
        print(f"  📊 Pred: {len(pred_coords)} atoms, Ref: {len(ref_coords)} atoms")
        
        if len(pred_coords) == 0 or len(ref_coords) == 0:
            print(f"  ❌ No valid coordinates")
            continue
        
        # Calculate metrics
        rmsd = calculate_rmsd_simple(pred_coords, ref_coords)
        tm_score = calculate_tm_simple(pred_coords, ref_coords)
        
        print(f"  📐 RMSD_CA: {rmsd:.3f} Å")
        print(f"  🔗 TM-score: {tm_score:.3f}")
        
        results.append({
            'target_id': target_id,
            'target_type': 'multimer' if target_id == 'H1101' else 'monomer',
            'length': len(pred_coords),
            'rmsd_ca': round(rmsd, 3) if rmsd else None,
            'tm_score': round(tm_score, 3) if tm_score else None
        })
    
    # Results summary
    df = pd.DataFrame(results)
    df.to_csv("quick_realistic_results.csv", index=False)
    
    print(f"\n📊 QUICK REALISTIC RESULTS:")
    print("=" * 60)
    print(df.to_string(index=False))
    
    # Statistics
    rmsd_values = df['rmsd_ca'].dropna()
    tm_values = df['tm_score'].dropna()
    
    print(f"\n📈 SUMMARY STATISTICS:")
    print("=" * 25)
    
    if len(rmsd_values) > 0:
        print(f"RMSD_CA: {rmsd_values.mean():.3f} ± {rmsd_values.std():.3f} Å")
        print(f"  Range: {rmsd_values.min():.3f} - {rmsd_values.max():.3f} Å")
    
    if len(tm_values) > 0:
        print(f"TM-score: {tm_values.mean():.3f} ± {tm_values.std():.3f}")
        print(f"  Range: {tm_values.min():.3f} - {tm_values.max():.3f}")
    
    # Quality breakdown
    if len(tm_values) > 0:
        excellent = sum(1 for tm in tm_values if tm >= 0.8)
        good = sum(1 for tm in tm_values if 0.5 <= tm < 0.8)
        mediocre = sum(1 for tm in tm_values if 0.2 <= tm < 0.5)
        poor = sum(1 for tm in tm_values if tm < 0.2)
        
        print(f"\n🎯 QUALITY BREAKDOWN:")
        print(f"  Excellent (≥0.8): {excellent}")
        print(f"  Good (0.5-0.8): {good}")
        print(f"  Mediocre (0.2-0.5): {mediocre}")
        print(f"  Poor (<0.2): {poor}")
    
    print(f"\n💾 Results saved to: quick_realistic_results.csv")
    
    return df


if __name__ == "__main__":
    results = main()
    print(f"\n🎉 Quick evaluation complete!")
