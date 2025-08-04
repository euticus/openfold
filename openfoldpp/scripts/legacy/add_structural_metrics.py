#!/usr/bin/env python3
"""
Add TM-score and RMSD_CA metrics to existing benchmark results.
This script recalculates structural metrics for the OpenFold++ predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional

# Import the enhanced metrics from our benchmark script
from enhanced_benchmark_script import compute_rmsd_ca, compute_tm_score

# Try to import tmtools directly
try:
    from tmtools import tm_align
    TM_AVAILABLE = True
    print("✅ tmtools available for TM-score calculation")
except ImportError:
    TM_AVAILABLE = False
    print("❌ tmtools not available")

# Try BioPython for RMSD
try:
    from Bio.PDB import PDBParser, Superimposer
    BIOPYTHON_AVAILABLE = True
    print("✅ BioPython available for RMSD calculation")
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("❌ BioPython not available")


def calculate_rmsd_ca_direct(pred_path: Path, ref_path: Path) -> Optional[float]:
    """Calculate CA RMSD directly using BioPython."""
    if not BIOPYTHON_AVAILABLE:
        return None
    
    try:
        parser = PDBParser(QUIET=True)
        
        # Parse structures
        pred_structure = parser.get_structure("pred", str(pred_path))
        ref_structure = parser.get_structure("ref", str(ref_path))
        
        # Extract CA atoms
        pred_atoms = []
        ref_atoms = []
        
        for model in pred_structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id("CA"):
                        pred_atoms.append(residue["CA"])
        
        for model in ref_structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id("CA"):
                        ref_atoms.append(residue["CA"])
        
        # Check if we have matching number of atoms
        min_atoms = min(len(pred_atoms), len(ref_atoms))
        if min_atoms == 0:
            logging.warning(f"No CA atoms found in structures")
            return None
        
        # Use only the matching atoms
        pred_atoms = pred_atoms[:min_atoms]
        ref_atoms = ref_atoms[:min_atoms]
        
        # Superimpose and calculate RMSD
        superimposer = Superimposer()
        superimposer.set_atoms(ref_atoms, pred_atoms)
        
        rmsd = superimposer.rms
        logging.info(f"RMSD calculated: {rmsd:.3f} Å ({min_atoms} CA atoms)")
        
        return float(rmsd)
        
    except Exception as e:
        logging.error(f"RMSD calculation failed: {e}")
        return None


def calculate_tm_score_direct(pred_path: Path, ref_path: Path) -> Optional[float]:
    """Calculate TM-score directly using tmtools."""
    if not TM_AVAILABLE:
        logging.warning("tmtools not available for TM-score calculation")
        return None

    try:
        # Calculate TM-score using tm_align
        result = tm_align(str(pred_path), str(ref_path))

        # tmtools returns a TMResult object
        if hasattr(result, 'tm_score'):
            tm_score = result.tm_score
        elif hasattr(result, 'tm_norm_chain1'):
            tm_score = result.tm_norm_chain1  # TM-score normalized by chain 1
        elif hasattr(result, 'tm_norm_chain2'):
            tm_score = result.tm_norm_chain2  # TM-score normalized by chain 2
        else:
            logging.error(f"Could not extract TM-score from result: {dir(result)}")
            return None

        if tm_score is not None:
            logging.info(f"TM-score calculated: {tm_score:.3f}")
            return float(tm_score)
        else:
            logging.error("TM-score is None")
            return None

    except Exception as e:
        logging.error(f"TM-score calculation failed: {e}")
        # Print more details for debugging
        print(f"    Debug: TM-score error for {pred_path.name} vs {ref_path.name}: {e}")
        return None


def add_structural_metrics_to_results(csv_file: Path, 
                                    predictions_dir: Path,
                                    reference_dir: Path) -> pd.DataFrame:
    """Add structural metrics to existing benchmark results."""
    
    # Load existing results
    df = pd.read_csv(csv_file)
    print(f"📊 Loaded {len(df)} results from {csv_file}")
    
    # Add new columns
    df['rmsd_ca'] = None
    df['tm_score'] = None
    
    # Process each target
    for idx, row in df.iterrows():
        target_id = row['target_id']
        print(f"\n🧬 Processing {target_id}...")
        
        # Find prediction file
        pred_file = predictions_dir / f"{target_id}_pred.pdb"
        if not pred_file.exists():
            print(f"  ❌ Prediction file not found: {pred_file}")
            continue
        
        # Find reference file - try multiple naming conventions
        ref_candidates = [
            reference_dir / f"{target_id.lower()}.pdb",
            reference_dir / f"{target_id}.pdb",
        ]
        
        # For CASP targets, also try PDB IDs
        if target_id == "T1024":
            ref_candidates.append(reference_dir / "6w70.pdb")
        elif target_id == "T1030":
            ref_candidates.append(reference_dir / "6xkl.pdb")
        elif target_id == "T1031":
            ref_candidates.append(reference_dir / "6w4h.pdb")
        elif target_id == "T1032":
            ref_candidates.append(reference_dir / "6m71.pdb")
        elif target_id == "H1101":
            ref_candidates.append(reference_dir / "6w63.pdb")
        
        ref_file = None
        for candidate in ref_candidates:
            if candidate.exists():
                ref_file = candidate
                break
        
        if not ref_file:
            print(f"  ❌ Reference file not found for {target_id}")
            print(f"     Tried: {[str(c) for c in ref_candidates]}")
            continue
        
        print(f"  📁 Prediction: {pred_file.name}")
        print(f"  📁 Reference: {ref_file.name}")
        
        # Calculate RMSD_CA
        rmsd_ca = calculate_rmsd_ca_direct(pred_file, ref_file)
        if rmsd_ca is not None:
            df.at[idx, 'rmsd_ca'] = round(rmsd_ca, 3)
            print(f"  ✅ RMSD_CA: {rmsd_ca:.3f} Å")
        else:
            print(f"  ❌ RMSD_CA calculation failed")
        
        # Calculate TM-score
        tm_score = calculate_tm_score_direct(pred_file, ref_file)
        if tm_score is not None:
            df.at[idx, 'tm_score'] = round(tm_score, 3)
            print(f"  ✅ TM-score: {tm_score:.3f}")
        else:
            print(f"  ❌ TM-score calculation failed")
    
    return df


def main():
    """Main function to add structural metrics."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🔬 Adding TM-score and RMSD_CA to OpenFold++ benchmark results")
    print("=" * 60)
    
    # File paths
    csv_file = Path("real_openfold_results.csv")
    predictions_dir = Path("real_openfold_predictions")
    reference_dir = Path("casp14_data/pdb")
    
    # Check if files exist
    if not csv_file.exists():
        print(f"❌ Results file not found: {csv_file}")
        return
    
    if not predictions_dir.exists():
        print(f"❌ Predictions directory not found: {predictions_dir}")
        return
    
    if not reference_dir.exists():
        print(f"❌ Reference directory not found: {reference_dir}")
        return
    
    print(f"📊 Results file: {csv_file}")
    print(f"🧬 Predictions: {predictions_dir}")
    print(f"📚 References: {reference_dir}")
    
    # List available files
    pred_files = list(predictions_dir.glob("*.pdb"))
    ref_files = list(reference_dir.glob("*.pdb"))
    
    print(f"\n📁 Available files:")
    print(f"  Predictions: {len(pred_files)} PDB files")
    for f in pred_files:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name}: {size_kb:.1f} KB")
    
    print(f"  References: {len(ref_files)} PDB files")
    for f in ref_files:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name}: {size_kb:.1f} KB")
    
    # Add structural metrics
    enhanced_df = add_structural_metrics_to_results(csv_file, predictions_dir, reference_dir)
    
    # Save enhanced results
    output_file = csv_file.parent / f"enhanced_{csv_file.name}"
    enhanced_df.to_csv(output_file, index=False)
    
    print(f"\n💾 Enhanced results saved to: {output_file}")
    
    # Display summary
    print(f"\n📈 STRUCTURAL METRICS SUMMARY")
    print("=" * 40)
    
    # RMSD statistics
    rmsd_values = enhanced_df['rmsd_ca'].dropna()
    if len(rmsd_values) > 0:
        print(f"RMSD_CA (Å):")
        print(f"  Count: {len(rmsd_values)}")
        print(f"  Mean: {rmsd_values.mean():.3f}")
        print(f"  Median: {rmsd_values.median():.3f}")
        print(f"  Min: {rmsd_values.min():.3f}")
        print(f"  Max: {rmsd_values.max():.3f}")
    else:
        print(f"RMSD_CA: No values calculated")
    
    # TM-score statistics
    tm_values = enhanced_df['tm_score'].dropna()
    if len(tm_values) > 0:
        print(f"\nTM-score:")
        print(f"  Count: {len(tm_values)}")
        print(f"  Mean: {tm_values.mean():.3f}")
        print(f"  Median: {tm_values.median():.3f}")
        print(f"  Min: {tm_values.min():.3f}")
        print(f"  Max: {tm_values.max():.3f}")
    else:
        print(f"TM-score: No values calculated")
    
    # Show enhanced results
    print(f"\n📊 ENHANCED RESULTS:")
    print("=" * 80)
    
    # Display key columns
    display_cols = ['target_id', 'target_type', 'total_length', 'time_s', 'confidence', 'rmsd_ca', 'tm_score']
    available_cols = [col for col in display_cols if col in enhanced_df.columns]
    
    print(enhanced_df[available_cols].to_string(index=False))
    
    print(f"\n🎉 Structural metrics successfully added!")
    print(f"📁 Full results available in: {output_file}")


if __name__ == "__main__":
    main()
