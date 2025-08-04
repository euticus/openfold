#!/usr/bin/env python3
"""
Evaluate Working OpenFold predictions with complete structural metrics.
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
    """Parse CA coordinates from PDB file."""
    coords = []
    sequence = []
    confidences = []
    
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and " CA " in line:
                try:
                    # Extract coordinates
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    
                    # Extract B-factor (confidence)
                    b_factor = float(line[60:66].strip())
                    confidence = 1.0 - (b_factor / 100.0)
                    
                    coords.append([x, y, z])
                    confidences.append(confidence)
                    
                    # Extract amino acid
                    aa = line[17:20].strip()
                    sequence.append(aa)
                    
                except (ValueError, IndexError):
                    continue
    
    return np.array(coords), sequence, np.array(confidences)


def calculate_rmsd_advanced(pred_coords, ref_coords):
    """Calculate RMSD with advanced alignment."""
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
    
    # Kabsch algorithm for optimal rotation
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


def calculate_tm_advanced(pred_coords, ref_coords):
    """Calculate TM-score with advanced algorithm."""
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
    
    # TM-score calculation with proper normalization
    d0 = 1.24 * (min_len - 15)**(1/3) - 1.8
    d0 = max(d0, 0.5)
    
    # TM-score formula
    scores = 1.0 / (1.0 + (distances / d0)**2)
    tm_score = np.mean(scores)
    
    return tm_score


def calculate_gdt_ts(pred_coords, ref_coords):
    """Calculate GDT_TS (Global Distance Test)."""
    if len(pred_coords) == 0 or len(ref_coords) == 0:
        return None
    
    min_len = min(len(pred_coords), len(ref_coords))
    pred_coords = pred_coords[:min_len]
    ref_coords = ref_coords[:min_len]
    
    if min_len < 3:
        return None
    
    # Align structures
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
    
    # GDT_TS calculation (percentage of residues within distance cutoffs)
    cutoffs = [1.0, 2.0, 4.0, 8.0]  # Angstroms
    gdt_scores = []
    
    for cutoff in cutoffs:
        within_cutoff = np.sum(distances <= cutoff) / len(distances)
        gdt_scores.append(within_cutoff)
    
    # GDT_TS is the average of the four cutoffs
    gdt_ts = np.mean(gdt_scores)
    
    return gdt_ts


def extract_metadata_from_pdb(pdb_path):
    """Extract metadata from PDB REMARK lines."""
    metadata = {}
    
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("REMARK   2 MEAN_CONFIDENCE:"):
                metadata['confidence'] = float(line.split()[-1])
            elif line.startswith("REMARK   4 SECONDARY_STRUCTURE:"):
                metadata['secondary_structure'] = line.split()[-1]
    
    return metadata


def main():
    """Evaluate working OpenFold predictions."""
    print("🔬 Evaluating Working OpenFold Predictions")
    print("=" * 50)
    
    # Target mappings
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
        
        pred_file = Path(f"working_openfold_predictions/{target_id}_working.pdb")
        ref_file = Path(f"casp14_data/pdb/{pdb_id}.pdb")
        
        if not pred_file.exists() or not ref_file.exists():
            print(f"  ❌ Files not found")
            continue
        
        # Parse coordinates
        pred_coords, pred_seq, pred_conf = parse_pdb_coords(pred_file)
        ref_coords, ref_seq, _ = parse_pdb_coords(ref_file)
        
        print(f"  📊 Pred: {len(pred_coords)} atoms, Ref: {len(ref_coords)} atoms")
        
        if len(pred_coords) == 0 or len(ref_coords) == 0:
            print(f"  ❌ No valid coordinates")
            continue
        
        # Extract metadata
        metadata = extract_metadata_from_pdb(pred_file)
        
        # Calculate metrics
        rmsd = calculate_rmsd_advanced(pred_coords, ref_coords)
        tm_score = calculate_tm_advanced(pred_coords, ref_coords)
        gdt_ts = calculate_gdt_ts(pred_coords, ref_coords)
        
        print(f"  📐 RMSD_CA: {rmsd:.3f} Å")
        print(f"  🔗 TM-score: {tm_score:.3f}")
        print(f"  🎯 GDT_TS: {gdt_ts:.3f}")
        print(f"  🎲 Confidence: {metadata.get('confidence', 'N/A')}")
        
        # Secondary structure analysis
        ss = metadata.get('secondary_structure', '')
        if ss:
            helix_frac = ss.count('H') / len(ss)
            sheet_frac = ss.count('E') / len(ss)
            coil_frac = ss.count('C') / len(ss)
            print(f"  🧬 SS: {helix_frac:.2f}H {sheet_frac:.2f}E {coil_frac:.2f}C")
        
        results.append({
            'target_id': target_id,
            'target_type': 'multimer' if target_id == 'H1101' else 'monomer',
            'length': len(pred_coords),
            'confidence': metadata.get('confidence'),
            'rmsd_ca': round(rmsd, 3) if rmsd else None,
            'tm_score': round(tm_score, 3) if tm_score else None,
            'gdt_ts': round(gdt_ts, 3) if gdt_ts else None,
            'helix_fraction': round(helix_frac, 3) if ss else None,
            'sheet_fraction': round(sheet_frac, 3) if ss else None,
            'coil_fraction': round(coil_frac, 3) if ss else None,
            'model_type': 'working_openfold'
        })
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    df.to_csv("working_openfold_results.csv", index=False)
    
    print(f"\n📊 WORKING OPENFOLD RESULTS:")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Statistics
    rmsd_values = df['rmsd_ca'].dropna()
    tm_values = df['tm_score'].dropna()
    gdt_values = df['gdt_ts'].dropna()
    conf_values = df['confidence'].dropna()
    
    print(f"\n📈 PERFORMANCE SUMMARY:")
    print("=" * 30)
    
    if len(rmsd_values) > 0:
        print(f"RMSD_CA (Å):")
        print(f"  Mean: {rmsd_values.mean():.3f} ± {rmsd_values.std():.3f}")
        print(f"  Range: {rmsd_values.min():.3f} - {rmsd_values.max():.3f}")
    
    if len(tm_values) > 0:
        print(f"\nTM-score:")
        print(f"  Mean: {tm_values.mean():.3f} ± {tm_values.std():.3f}")
        print(f"  Range: {tm_values.min():.3f} - {tm_values.max():.3f}")
    
    if len(gdt_values) > 0:
        print(f"\nGDT_TS:")
        print(f"  Mean: {gdt_values.mean():.3f} ± {gdt_values.std():.3f}")
        print(f"  Range: {gdt_values.min():.3f} - {gdt_values.max():.3f}")
    
    if len(conf_values) > 0:
        print(f"\nConfidence:")
        print(f"  Mean: {conf_values.mean():.3f} ± {conf_values.std():.3f}")
        print(f"  Range: {conf_values.min():.3f} - {conf_values.max():.3f}")
    
    # Quality assessment
    print(f"\n🎯 QUALITY ASSESSMENT:")
    print("=" * 25)
    
    if len(tm_values) > 0:
        excellent = sum(1 for tm in tm_values if tm >= 0.8)
        good = sum(1 for tm in tm_values if 0.5 <= tm < 0.8)
        mediocre = sum(1 for tm in tm_values if 0.2 <= tm < 0.5)
        poor = sum(1 for tm in tm_values if tm < 0.2)
        
        print(f"Excellent (TM ≥ 0.8): {excellent}")
        print(f"Good (TM 0.5-0.8): {good}")
        print(f"Mediocre (TM 0.2-0.5): {mediocre}")
        print(f"Poor (TM < 0.2): {poor}")
    
    # CASP-style assessment
    if len(gdt_values) > 0:
        print(f"\n🏆 CASP-STYLE ASSESSMENT:")
        print("=" * 30)
        
        high_accuracy = sum(1 for gdt in gdt_values if gdt >= 0.7)
        medium_accuracy = sum(1 for gdt in gdt_values if 0.4 <= gdt < 0.7)
        low_accuracy = sum(1 for gdt in gdt_values if gdt < 0.4)
        
        print(f"High accuracy (GDT ≥ 0.7): {high_accuracy}")
        print(f"Medium accuracy (GDT 0.4-0.7): {medium_accuracy}")
        print(f"Low accuracy (GDT < 0.4): {low_accuracy}")
    
    print(f"\n💾 Results saved to: working_openfold_results.csv")
    
    return df


if __name__ == "__main__":
    results_df = main()
    
    print(f"\n🎉 Working OpenFold evaluation complete!")
    print(f"📁 Results: working_openfold_results.csv")
