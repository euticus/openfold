#!/usr/bin/env python3
"""
Simple script to add RMSD_CA and TM-score to benchmark results.
Uses the functions from enhanced_benchmark_script.py
"""

import pandas as pd
from pathlib import Path
import sys

# Import the metric functions from our benchmark script
sys.path.append('.')
from enhanced_benchmark_script import compute_rmsd_ca, compute_tm_score


def add_metrics_to_csv():
    """Add RMSD_CA and TM-score to the existing CSV."""
    
    # Load the existing results
    csv_file = Path("real_openfold_results.csv")
    df = pd.read_csv(csv_file)
    
    print(f"📊 Loaded {len(df)} results from {csv_file}")
    
    # Add new columns
    df['rmsd_ca'] = None
    df['tm_score'] = None
    
    # Directories
    pred_dir = Path("real_openfold_predictions")
    ref_dir = Path("casp14_data/pdb")
    
    # Mapping of target IDs to PDB IDs
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
        
        # Find prediction file
        pred_file = pred_dir / f"{target_id}_pred.pdb"
        if not pred_file.exists():
            print(f"  ❌ Prediction not found: {pred_file}")
            continue
        
        # Find reference file
        pdb_id = target_to_pdb.get(target_id)
        if not pdb_id:
            print(f"  ❌ No PDB ID mapping for {target_id}")
            continue
        
        ref_file = ref_dir / f"{pdb_id}.pdb"
        if not ref_file.exists():
            print(f"  ❌ Reference not found: {ref_file}")
            continue
        
        print(f"  📁 Prediction: {pred_file.name}")
        print(f"  📁 Reference: {ref_file.name}")
        
        # Calculate RMSD_CA
        rmsd_ca = compute_rmsd_ca(pred_file, ref_file)
        if rmsd_ca is not None:
            df.at[idx, 'rmsd_ca'] = round(rmsd_ca, 3)
            print(f"  ✅ RMSD_CA: {rmsd_ca:.3f} Å")
        else:
            print(f"  ❌ RMSD_CA calculation failed")
        
        # Calculate TM-score
        tm_score = compute_tm_score(pred_file, ref_file)
        if tm_score is not None:
            df.at[idx, 'tm_score'] = round(tm_score, 3)
            print(f"  ✅ TM-score: {tm_score:.3f}")
        else:
            print(f"  ❌ TM-score calculation failed")
    
    # Save enhanced results
    output_file = "enhanced_real_openfold_results.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n💾 Enhanced results saved to: {output_file}")
    
    # Show summary
    print(f"\n📈 STRUCTURAL METRICS SUMMARY")
    print("=" * 40)
    
    # RMSD statistics
    rmsd_values = df['rmsd_ca'].dropna()
    if len(rmsd_values) > 0:
        print(f"RMSD_CA (Å):")
        print(f"  Count: {len(rmsd_values)}")
        print(f"  Mean: {rmsd_values.mean():.3f}")
        print(f"  Median: {rmsd_values.median():.3f}")
        print(f"  Range: {rmsd_values.min():.3f} - {rmsd_values.max():.3f}")
    
    # TM-score statistics  
    tm_values = df['tm_score'].dropna()
    if len(tm_values) > 0:
        print(f"\nTM-score:")
        print(f"  Count: {len(tm_values)}")
        print(f"  Mean: {tm_values.mean():.3f}")
        print(f"  Median: {tm_values.median():.3f}")
        print(f"  Range: {tm_values.min():.3f} - {tm_values.max():.3f}")
    
    # Display enhanced results
    print(f"\n📊 ENHANCED RESULTS:")
    print("=" * 100)
    
    # Show key columns
    display_cols = ['target_id', 'target_type', 'total_length', 'time_s', 'confidence', 'rmsd_ca', 'tm_score']
    print(df[display_cols].to_string(index=False))
    
    return df


if __name__ == "__main__":
    print("🔬 Adding RMSD_CA and TM-score to OpenFold++ results")
    print("=" * 55)
    
    enhanced_df = add_metrics_to_csv()
    
    print(f"\n🎉 Structural metrics successfully added!")
    print(f"📁 Enhanced results: enhanced_real_openfold_results.csv")
