#!/usr/bin/env python3
"""
Quick fix for structural metrics calculation.
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
except ImportError:
    TM_AVAILABLE = False


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


def main():
    """Calculate metrics for all targets."""
    print("🔧 Quick structural metrics calculation")
    
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
        
        pred_file = Path(f"real_openfold_predictions/{target_id}_pred.pdb")
        ref_file = Path(f"casp14_data/pdb/{pdb_id}.pdb")
        
        if not pred_file.exists():
            print(f"  ❌ Prediction not found")
            continue
            
        if not ref_file.exists():
            print(f"  ❌ Reference not found")
            continue
        
        rmsd = calculate_rmsd(pred_file, ref_file)
        tm_score = calculate_tm(pred_file, ref_file)
        
        print(f"  RMSD_CA: {rmsd} Å")
        print(f"  TM-score: {tm_score}")
        
        results.append({
            'target_id': target_id,
            'rmsd_ca': rmsd,
            'tm_score': tm_score
        })
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    df.to_csv("quick_metrics_results.csv", index=False)
    
    print(f"\n📊 RESULTS SUMMARY:")
    print("=" * 40)
    print(df.to_string(index=False))
    
    # Statistics
    rmsd_values = df['rmsd_ca'].dropna()
    tm_values = df['tm_score'].dropna()
    
    if len(rmsd_values) > 0:
        print(f"\nRMSD_CA: mean={rmsd_values.mean():.3f}, range={rmsd_values.min():.3f}-{rmsd_values.max():.3f}")
    
    if len(tm_values) > 0:
        print(f"TM-score: mean={tm_values.mean():.3f}, range={tm_values.min():.3f}-{tm_values.max():.3f}")
    
    print(f"\n💾 Results saved to: quick_metrics_results.csv")


if __name__ == "__main__":
    main()
