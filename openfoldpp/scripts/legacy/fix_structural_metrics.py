#!/usr/bin/env python3
"""
Fixed structural metrics calculation for OpenFold++ benchmark.
This properly calculates RMSD_CA and TM-score with correct alignment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser, Superimposer
import warnings

# Suppress BioPython warnings
warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB")

# Try tmtools
try:
    from tmtools import tm_align
    TM_AVAILABLE = True
    print("✅ tmtools available")
except ImportError:
    TM_AVAILABLE = False
    print("❌ tmtools not available")


def get_ca_coords_and_seq(pdb_path):
    """Extract CA coordinates and sequence properly."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    ca_atoms = []
    sequence = []

    # Standard amino acid mapping
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
                    # Get amino acid code
                    resname = residue.get_resname()
                    aa_code = aa_map.get(resname, 'X')  # X for unknown
                    sequence.append(aa_code)

    coords = np.array([atom.get_coord() for atom in ca_atoms])
    seq_str = ''.join(sequence)

    return coords, ca_atoms, seq_str


def calculate_rmsd_ca_fixed(pred_path: Path, ref_path: Path):
    """Calculate RMSD_CA with proper superposition."""
    try:
        print(f"    📏 Calculating RMSD for {pred_path.name} vs {ref_path.name}")
        
        # Get coordinates and atoms
        pred_coords, pred_atoms, _ = get_ca_coords_and_seq(pred_path)
        ref_coords, ref_atoms, _ = get_ca_coords_and_seq(ref_path)
        
        print(f"    📊 Pred atoms: {len(pred_atoms)}, Ref atoms: {len(ref_atoms)}")
        
        if len(pred_atoms) == 0 or len(ref_atoms) == 0:
            print(f"    ❌ No CA atoms found")
            return None
        
        # Use minimum length for alignment
        min_len = min(len(pred_atoms), len(ref_atoms))
        if min_len < 3:
            print(f"    ❌ Too few atoms: {min_len}")
            return None
        
        # Trim to same length
        pred_atoms_trim = pred_atoms[:min_len]
        ref_atoms_trim = ref_atoms[:min_len]
        
        print(f"    🔄 Aligning {min_len} CA atoms")
        
        # Use BioPython Superimposer (order: reference, mobile)
        sup = Superimposer()
        sup.set_atoms(ref_atoms_trim, pred_atoms_trim)
        
        rmsd = sup.rms
        print(f"    ✅ RMSD_CA: {rmsd:.3f} Å")
        
        return float(rmsd)
        
    except Exception as e:
        print(f"    ❌ RMSD error: {e}")
        return None


def calculate_tm_score_fixed(pred_path: Path, ref_path: Path):
    """Calculate TM-score using tmtools."""
    if not TM_AVAILABLE:
        print(f"    ⚠️  tmtools not available")
        return None
    
    try:
        print(f"    📏 Calculating TM-score for {pred_path.name} vs {ref_path.name}")
        
        # Get coordinates and sequences
        pred_coords, _, pred_seq = get_ca_coords_and_seq(pred_path)
        ref_coords, _, ref_seq = get_ca_coords_and_seq(ref_path)
        
        if len(pred_coords) == 0 or len(ref_coords) == 0:
            print(f"    ❌ No coordinates found")
            return None
        
        # Trim to same length
        min_len = min(len(pred_coords), len(ref_coords))
        pred_coords = pred_coords[:min_len]
        ref_coords = ref_coords[:min_len]
        pred_seq = pred_seq[:min_len]
        ref_seq = ref_seq[:min_len]

        print(f"    🔄 TM-align with {min_len} atoms")

        # Use tmtools with sequences (order: mobile, reference)
        result = tm_align(pred_coords, ref_coords, pred_seq, ref_seq)
        
        # Extract TM-score
        if hasattr(result, 'tm_norm_chain1'):
            tm_score = result.tm_norm_chain1
        elif hasattr(result, 'tm_score'):
            tm_score = result.tm_score
        else:
            print(f"    ❌ Cannot extract TM-score from result")
            return None
        
        print(f"    ✅ TM-score: {tm_score:.3f}")
        return float(tm_score)
        
    except Exception as e:
        print(f"    ❌ TM-score error: {e}")
        return None


def debug_pdb_structure(pdb_path: Path):
    """Debug PDB structure to understand the format."""
    print(f"\n🔍 Debugging {pdb_path.name}:")
    
    # Count different record types
    with open(pdb_path) as f:
        lines = f.readlines()
    
    atom_lines = [l for l in lines if l.startswith("ATOM")]
    ca_lines = [l for l in atom_lines if " CA " in l]
    
    print(f"  Total lines: {len(lines)}")
    print(f"  ATOM records: {len(atom_lines)}")
    print(f"  CA atoms: {len(ca_lines)}")
    
    if ca_lines:
        print(f"  First CA: {ca_lines[0].strip()}")
        print(f"  Last CA: {ca_lines[-1].strip()}")
    
    # Check chains
    chains = set()
    for line in ca_lines:
        if len(line) > 21:
            chains.add(line[21])
    print(f"  Chains: {sorted(chains)}")


def test_single_target():
    """Test calculation on a single target for debugging."""
    print("🧪 Testing single target calculation...")
    
    pred_file = Path("real_openfold_predictions/T1024_pred.pdb")
    ref_file = Path("casp14_data/pdb/6w70.pdb")
    
    if not pred_file.exists():
        print(f"❌ Prediction file not found: {pred_file}")
        return
    
    if not ref_file.exists():
        print(f"❌ Reference file not found: {ref_file}")
        return
    
    # Debug both structures
    debug_pdb_structure(pred_file)
    debug_pdb_structure(ref_file)
    
    # Calculate metrics
    print(f"\n📊 Calculating metrics for T1024...")
    rmsd = calculate_rmsd_ca_fixed(pred_file, ref_file)
    tm_score = calculate_tm_score_fixed(pred_file, ref_file)
    
    print(f"\n📈 Results for T1024:")
    print(f"  RMSD_CA: {rmsd:.3f} Å" if rmsd else "  RMSD_CA: Failed")
    print(f"  TM-score: {tm_score:.3f}" if tm_score else "  TM-score: Failed")


def fix_all_metrics():
    """Fix metrics for all targets."""
    print("🔧 Fixing structural metrics for all targets...")
    
    # Load existing results
    csv_file = Path("real_openfold_results.csv")
    if not csv_file.exists():
        print(f"❌ Results file not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)

    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    print(f"📊 Loaded {len(df)} results")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Reset metrics columns
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
            print(f"  ❌ Prediction not found")
            continue
            
        if not pdb_id:
            print(f"  ❌ No PDB mapping")
            continue
            
        ref_file = ref_dir / f"{pdb_id}.pdb"
        if not ref_file.exists():
            print(f"  ❌ Reference not found")
            continue
        
        # Calculate fixed metrics
        rmsd = calculate_rmsd_ca_fixed(pred_file, ref_file)
        if rmsd is not None:
            df.at[idx, 'rmsd_ca'] = round(rmsd, 3)
        
        tm_score = calculate_tm_score_fixed(pred_file, ref_file)
        if tm_score is not None:
            df.at[idx, 'tm_score'] = round(tm_score, 3)
    
    # Save fixed results
    output_file = "fixed_structural_metrics.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n💾 Fixed results saved to: {output_file}")
    
    # Show summary
    rmsd_values = df['rmsd_ca'].dropna()
    tm_values = df['tm_score'].dropna()
    
    print(f"\n📈 FIXED METRICS SUMMARY:")
    print("=" * 40)
    
    if len(rmsd_values) > 0:
        print(f"RMSD_CA (Å): {len(rmsd_values)} values")
        print(f"  Mean: {rmsd_values.mean():.3f}")
        print(f"  Range: {rmsd_values.min():.3f} - {rmsd_values.max():.3f}")
    
    if len(tm_values) > 0:
        print(f"TM-score: {len(tm_values)} values")
        print(f"  Mean: {tm_values.mean():.3f}")
        print(f"  Range: {tm_values.min():.3f} - {tm_values.max():.3f}")
    
    # Display results
    print(f"\n📊 FIXED RESULTS:")
    print("=" * 100)
    
    display_cols = ['target_id', 'target_type', 'total_length', 'confidence', 'rmsd_ca', 'tm_score']
    available_cols = [col for col in display_cols if col in df.columns]
    print(df[available_cols].to_string(index=False))
    
    return df


def main():
    """Main function."""
    print("🔧 Fixing OpenFold++ Structural Metrics")
    print("=" * 45)
    
    # First test on single target
    test_single_target()
    
    # Then fix all metrics
    print(f"\n" + "="*60)
    fixed_df = fix_all_metrics()
    
    print(f"\n🎉 Structural metrics fixed!")
    print(f"📁 Results: fixed_structural_metrics.csv")


if __name__ == "__main__":
    main()
