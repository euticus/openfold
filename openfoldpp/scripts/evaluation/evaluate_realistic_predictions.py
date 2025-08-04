#!/usr/bin/env python3
"""
Evaluate realistic OpenFold predictions with structural metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser, Superimposer
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB")

try:
    from tmtools import tm_align
    TM_AVAILABLE = True
    print("✅ tmtools available")
except ImportError:
    TM_AVAILABLE = False
    print("❌ tmtools not available")


def get_ca_atoms(pdb_path):
    """Get CA atoms from PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    
    ca_atoms = []
    sequence = []
    
    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id("CA"):
                    ca_atoms.append(residue["CA"])
                    resname = residue.get_resname()
                    sequence.append(aa_map.get(resname, 'X'))
    
    return ca_atoms, ''.join(sequence)


def calculate_rmsd(pred_path, ref_path):
    """Calculate RMSD with proper alignment."""
    try:
        pred_atoms, _ = get_ca_atoms(pred_path)
        ref_atoms, _ = get_ca_atoms(ref_path)
        
        if not pred_atoms or not ref_atoms:
            return None
        
        min_len = min(len(pred_atoms), len(ref_atoms))
        if min_len < 3:
            return None
        
        pred_atoms = pred_atoms[:min_len]
        ref_atoms = ref_atoms[:min_len]
        
        sup = Superimposer()
        sup.set_atoms(ref_atoms, pred_atoms)
        
        return round(sup.rms, 3)
        
    except Exception as e:
        print(f"RMSD error: {e}")
        return None


def calculate_tm(pred_path, ref_path):
    """Calculate TM-score."""
    if not TM_AVAILABLE:
        return None
    
    try:
        pred_atoms, pred_seq = get_ca_atoms(pred_path)
        ref_atoms, ref_seq = get_ca_atoms(ref_path)
        
        if not pred_atoms or not ref_atoms:
            return None
        
        min_len = min(len(pred_atoms), len(ref_atoms))
        
        pred_coords = np.array([a.get_coord() for a in pred_atoms[:min_len]])
        ref_coords = np.array([a.get_coord() for a in ref_atoms[:min_len]])
        
        result = tm_align(pred_coords, ref_coords, pred_seq[:min_len], ref_seq[:min_len])
        
        if hasattr(result, 'tm_norm_chain1'):
            return round(result.tm_norm_chain1, 3)
        elif hasattr(result, 'tm_score'):
            return round(result.tm_score, 3)
        
        return None
        
    except Exception as e:
        print(f"TM-score error: {e}")
        return None


def extract_confidence_from_pdb(pdb_path):
    """Extract confidence from PDB REMARK."""
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("REMARK   2 CONFIDENCE:"):
                    return float(line.split()[-1])
        return None
    except:
        return None


def main():
    """Evaluate realistic predictions."""
    print("🔬 Evaluating Realistic OpenFold Predictions")
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
        
        pred_file = Path(f"real_realistic_predictions/{target_id}_realistic.pdb")
        ref_file = Path(f"casp14_data/pdb/{pdb_id}.pdb")
        
        if not pred_file.exists():
            print(f"  ❌ Prediction not found")
            continue
            
        if not ref_file.exists():
            print(f"  ❌ Reference not found")
            continue
        
        # Extract metadata
        confidence = extract_confidence_from_pdb(pred_file)
        
        # Get sequence length
        pred_atoms, _ = get_ca_atoms(pred_file)
        length = len(pred_atoms)
        
        # Calculate metrics
        rmsd = calculate_rmsd(pred_file, ref_file)
        tm_score = calculate_tm(pred_file, ref_file)
        
        print(f"  📏 Length: {length}")
        print(f"  🎯 Confidence: {confidence}")
        print(f"  📐 RMSD_CA: {rmsd} Å")
        print(f"  🔗 TM-score: {tm_score}")
        
        results.append({
            'target_id': target_id,
            'target_type': 'multimer' if target_id == 'H1101' else 'monomer',
            'length': length,
            'confidence': confidence,
            'rmsd_ca': rmsd,
            'tm_score': tm_score,
            'prediction_type': 'realistic_openfold'
        })
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    df.to_csv("realistic_openfold_results.csv", index=False)
    
    print(f"\n📊 REALISTIC OPENFOLD RESULTS:")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Statistics
    rmsd_values = df['rmsd_ca'].dropna()
    tm_values = df['tm_score'].dropna()
    conf_values = df['confidence'].dropna()
    
    print(f"\n📈 PERFORMANCE SUMMARY:")
    print("=" * 30)
    
    if len(rmsd_values) > 0:
        print(f"RMSD_CA (Å):")
        print(f"  Mean: {rmsd_values.mean():.3f}")
        print(f"  Range: {rmsd_values.min():.3f} - {rmsd_values.max():.3f}")
    
    if len(tm_values) > 0:
        print(f"\nTM-score:")
        print(f"  Mean: {tm_values.mean():.3f}")
        print(f"  Range: {tm_values.min():.3f} - {tm_values.max():.3f}")
    
    if len(conf_values) > 0:
        print(f"\nConfidence:")
        print(f"  Mean: {conf_values.mean():.3f}")
        print(f"  Range: {conf_values.min():.3f} - {conf_values.max():.3f}")
    
    # Quality assessment
    print(f"\n🎯 QUALITY ASSESSMENT:")
    print("=" * 25)
    
    excellent = sum(1 for tm in tm_values if tm >= 0.8)
    good = sum(1 for tm in tm_values if 0.5 <= tm < 0.8)
    mediocre = sum(1 for tm in tm_values if 0.2 <= tm < 0.5)
    poor = sum(1 for tm in tm_values if tm < 0.2)
    
    print(f"Excellent (TM ≥ 0.8): {excellent}")
    print(f"Good (TM 0.5-0.8): {good}")
    print(f"Mediocre (TM 0.2-0.5): {mediocre}")
    print(f"Poor (TM < 0.2): {poor}")
    
    print(f"\n💾 Results saved to: realistic_openfold_results.csv")
    
    return df


if __name__ == "__main__":
    results_df = main()
    
    print(f"\n🎉 Realistic OpenFold evaluation complete!")
    print(f"📁 Results: realistic_openfold_results.csv")
