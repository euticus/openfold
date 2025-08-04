#!/usr/bin/env python3
"""
Ultimate Comprehensive Benchmark - Every Single Component.
This runs ALL OpenFold++ approaches and compares them comprehensively.
"""

import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

# Import all our pipeline components
import sys
sys.path.append('.')

try:
    from quick_metrics_fix import calculate_rmsd, calculate_tm
    from working_openfold_pipeline import WorkingOpenFoldPipeline
    from trained_openfold_pipeline import TrainedOpenFoldPipeline
    from full_infrastructure_pipeline import FullInfrastructurePipeline
except ImportError as e:
    print(f"⚠️  Import warning: {e}")


class UltimateComprehensiveBenchmark:
    """Ultimate benchmark testing every single component."""
    
    def __init__(self):
        """Initialize the ultimate benchmark."""
        
        print("🚀 ULTIMATE COMPREHENSIVE OPENFOLD++ BENCHMARK")
        print("=" * 55)
        
        # Initialize all pipelines
        self.pipelines = {}
        self.results = {}
        
        # Target mappings
        self.targets = {
            "T1024": "6w70",
            "T1030": "6xkl", 
            "T1031": "6w4h",
            "T1032": "6m71",
            "H1101": "6w63"
        }
        
        # Setup directories
        self.fasta_dir = Path("casp14_data/fasta")
        self.ref_dir = Path("casp14_data/pdb")
        
        print(f"📁 FASTA directory: {self.fasta_dir}")
        print(f"📁 Reference directory: {self.ref_dir}")
        print(f"🎯 Targets: {list(self.targets.keys())}")
    
    def initialize_all_pipelines(self):
        """Initialize every single pipeline component."""
        
        print(f"\n🔧 INITIALIZING ALL PIPELINE COMPONENTS")
        print("=" * 45)
        
        # 1. Mock Server (baseline)
        print("1️⃣ Mock Server Pipeline...")
        self.pipelines['mock'] = self._create_mock_pipeline()
        
        # 2. Realistic Algorithm
        print("2️⃣ Realistic Algorithm Pipeline...")
        self.pipelines['realistic'] = self._create_realistic_pipeline()
        
        # 3. Working OpenFold
        print("3️⃣ Working OpenFold Pipeline...")
        try:
            self.pipelines['working'] = WorkingOpenFoldPipeline()
            print("✅ Working OpenFold initialized")
        except Exception as e:
            print(f"❌ Working OpenFold failed: {e}")
            self.pipelines['working'] = None
        
        # 4. Trained OpenFold
        print("4️⃣ Trained OpenFold Pipeline...")
        try:
            self.pipelines['trained'] = TrainedOpenFoldPipeline()
            print("✅ Trained OpenFold initialized")
        except Exception as e:
            print(f"❌ Trained OpenFold failed: {e}")
            self.pipelines['trained'] = None
        
        # 5. Full Infrastructure
        print("5️⃣ Full Infrastructure Pipeline...")
        try:
            self.pipelines['full_infrastructure'] = FullInfrastructurePipeline(
                weights_path="openfold_model_1_ptm.pt",
                gpu=True,
                full_msa=True
            )
            print("✅ Full Infrastructure initialized")
        except Exception as e:
            print(f"❌ Full Infrastructure failed: {e}")
            self.pipelines['full_infrastructure'] = None
        
        print(f"\n✅ Initialized {len([p for p in self.pipelines.values() if p is not None])}/5 pipelines")
    
    def _create_mock_pipeline(self):
        """Create mock pipeline."""
        
        class MockPipeline:
            def predict_structure(self, sequence, target_id):
                # Generate random helix coordinates
                coords = np.zeros((len(sequence), 3))
                for i in range(len(sequence)):
                    angle = i * 100 * np.pi / 180
                    radius = 2.3
                    coords[i] = [i * 1.5, radius * np.cos(angle), radius * np.sin(angle)]
                
                # Add noise
                coords += np.random.normal(0, 0.1, coords.shape)
                
                # Create PDB
                pdb_lines = ["HEADER    MOCK SERVER PREDICTION"]
                for i, (aa, coord) in enumerate(zip(sequence, coords)):
                    pdb_lines.append(
                        f"ATOM  {i+1:5d}  CA  {aa} A{i+1:4d}    "
                        f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00 50.00           C"
                    )
                pdb_lines.append("END")
                
                return "\n".join(pdb_lines), 0.5, {
                    'model_type': 'mock_server',
                    'processing_time': 0.1
                }
        
        return MockPipeline()
    
    def _create_realistic_pipeline(self):
        """Create realistic algorithm pipeline."""
        
        class RealisticPipeline:
            def predict_structure(self, sequence, target_id):
                # Secondary structure prediction
                ss_pred = []
                helix_propensity = {
                    'A': 1.42, 'E': 1.51, 'L': 1.21, 'M': 1.45, 'Q': 1.11, 'K': 1.16,
                    'R': 0.98, 'H': 1.00, 'V': 1.06, 'I': 1.08, 'Y': 0.69, 'F': 1.13
                }
                
                for aa in sequence:
                    h_prop = helix_propensity.get(aa, 1.0)
                    ss_pred.append('H' if h_prop > 1.1 else 'C')
                
                # Generate coordinates
                coords = np.zeros((len(sequence), 3))
                for i, (aa, ss) in enumerate(zip(sequence, ss_pred)):
                    if ss == 'H':
                        angle = i * 100 * np.pi / 180
                        radius = 2.3
                        coords[i] = [i * 1.5, radius * np.cos(angle), radius * np.sin(angle)]
                    else:
                        coords[i] = [i * 3.8, 0, 0]
                
                # Create PDB
                pdb_lines = ["HEADER    REALISTIC ALGORITHM PREDICTION"]
                for i, (aa, coord) in enumerate(zip(sequence, coords)):
                    pdb_lines.append(
                        f"ATOM  {i+1:5d}  CA  {aa} A{i+1:4d}    "
                        f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00 30.00           C"
                    )
                pdb_lines.append("END")
                
                return "\n".join(pdb_lines), 0.7, {
                    'model_type': 'realistic_algorithm',
                    'processing_time': 0.2
                }
        
        return RealisticPipeline()
    
    def run_complete_benchmark(self):
        """Run the complete benchmark on all targets with all pipelines."""
        
        print(f"\n🏁 RUNNING COMPLETE BENCHMARK")
        print("=" * 35)
        
        all_results = []
        
        for target_id, pdb_id in self.targets.items():
            print(f"\n{'='*70}")
            print(f"🎯 BENCHMARKING {target_id} WITH ALL PIPELINES")
            print(f"{'='*70}")
            
            # Load sequence
            fasta_file = self.fasta_dir / f"{target_id}.fasta"
            if not fasta_file.exists():
                print(f"❌ FASTA file not found: {fasta_file}")
                continue
            
            with open(fasta_file) as f:
                lines = f.readlines()
            sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
            
            print(f"📏 Sequence length: {len(sequence)}")
            
            # Run each pipeline
            for pipeline_name, pipeline in self.pipelines.items():
                if pipeline is None:
                    print(f"⏭️  Skipping {pipeline_name} (not initialized)")
                    continue
                
                print(f"\n🔄 Running {pipeline_name.upper()}...")
                
                try:
                    start_time = time.time()
                    
                    # Run prediction
                    pdb_content, confidence, metadata = pipeline.predict_structure(sequence, target_id)
                    
                    processing_time = time.time() - start_time
                    
                    # Save prediction
                    output_dir = Path(f"ultimate_benchmark_predictions/{pipeline_name}")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_file = output_dir / f"{target_id}_{pipeline_name}.pdb"
                    with open(output_file, 'w') as f:
                        f.write(pdb_content)
                    
                    print(f"✅ {pipeline_name}: {processing_time:.2f}s, confidence: {confidence:.3f}")
                    print(f"💾 Saved: {output_file}")
                    
                    # Calculate structural metrics
                    ref_file = self.ref_dir / f"{pdb_id}.pdb"
                    if ref_file.exists():
                        rmsd, tm_score = self._calculate_metrics(output_file, ref_file)
                        print(f"📐 RMSD: {rmsd:.3f} Å, TM-score: {tm_score:.3f}")
                    else:
                        rmsd, tm_score = None, None
                        print(f"❌ Reference file not found: {ref_file}")
                    
                    # Store results
                    result = {
                        'target_id': target_id,
                        'pipeline': pipeline_name,
                        'sequence_length': len(sequence),
                        'processing_time': processing_time,
                        'confidence': confidence,
                        'rmsd_ca': rmsd,
                        'tm_score': tm_score,
                        'model_type': metadata.get('model_type', pipeline_name),
                        'output_file': str(output_file)
                    }
                    
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"❌ {pipeline_name} failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Save all results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv("ultimate_comprehensive_results.csv", index=False)
        
        print(f"\n💾 All results saved to: ultimate_comprehensive_results.csv")
        
        return results_df
    
    def _calculate_metrics(self, pred_file: Path, ref_file: Path) -> Tuple[float, float]:
        """Calculate RMSD and TM-score."""
        
        try:
            # Parse coordinates
            pred_coords = self._parse_ca_coords(pred_file)
            ref_coords = self._parse_ca_coords(ref_file)
            
            if len(pred_coords) == 0 or len(ref_coords) == 0:
                return None, None
            
            # Calculate RMSD
            rmsd = self._calculate_rmsd(pred_coords, ref_coords)
            
            # Calculate TM-score
            tm_score = self._calculate_tm_score(pred_coords, ref_coords)
            
            return rmsd, tm_score
            
        except Exception as e:
            print(f"⚠️  Metrics calculation failed: {e}")
            return None, None
    
    def _parse_ca_coords(self, pdb_file: Path) -> np.ndarray:
        """Parse CA coordinates from PDB file."""
        coords = []
        
        with open(pdb_file) as f:
            for line in f:
                if line.startswith("ATOM") and " CA " in line:
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coords.append([x, y, z])
                    except (ValueError, IndexError):
                        continue
        
        return np.array(coords)
    
    def _calculate_rmsd(self, pred_coords: np.ndarray, ref_coords: np.ndarray) -> float:
        """Calculate RMSD with optimal alignment."""
        
        min_len = min(len(pred_coords), len(ref_coords))
        if min_len < 3:
            return float('inf')
        
        pred_coords = pred_coords[:min_len]
        ref_coords = ref_coords[:min_len]
        
        # Center coordinates
        pred_center = np.mean(pred_coords, axis=0)
        ref_center = np.mean(ref_coords, axis=0)
        
        pred_centered = pred_coords - pred_center
        ref_centered = ref_coords - ref_center
        
        # Optimal rotation using SVD
        H = pred_centered.T @ ref_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Apply rotation and calculate RMSD
        pred_rotated = pred_centered @ R.T
        diff = pred_rotated - ref_centered
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        
        return float(rmsd)
    
    def _calculate_tm_score(self, pred_coords: np.ndarray, ref_coords: np.ndarray) -> float:
        """Calculate TM-score."""
        
        min_len = min(len(pred_coords), len(ref_coords))
        if min_len < 3:
            return 0.0
        
        pred_coords = pred_coords[:min_len]
        ref_coords = ref_coords[:min_len]
        
        # Align coordinates
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
        
        # Calculate TM-score
        distances = np.linalg.norm(pred_rotated - ref_centered, axis=1)
        d0 = 1.24 * (min_len - 15)**(1/3) - 1.8
        d0 = max(d0, 0.5)
        
        scores = 1.0 / (1.0 + (distances / d0)**2)
        tm_score = np.mean(scores)
        
        return float(tm_score)
    
    def analyze_comprehensive_results(self, results_df: pd.DataFrame):
        """Analyze comprehensive results across all pipelines."""
        
        print(f"\n📊 COMPREHENSIVE RESULTS ANALYSIS")
        print("=" * 40)
        
        # Results by pipeline
        print(f"\n🔍 RESULTS BY PIPELINE:")
        print("=" * 30)
        
        for pipeline in results_df['pipeline'].unique():
            pipeline_data = results_df[results_df['pipeline'] == pipeline]
            
            rmsd_values = pipeline_data['rmsd_ca'].dropna()
            tm_values = pipeline_data['tm_score'].dropna()
            time_values = pipeline_data['processing_time'].dropna()
            conf_values = pipeline_data['confidence'].dropna()
            
            print(f"\n{pipeline.upper()}:")
            print(f"  Targets: {len(pipeline_data)}")
            
            if len(rmsd_values) > 0:
                print(f"  RMSD_CA: {rmsd_values.mean():.3f} ± {rmsd_values.std():.3f} Å")
            
            if len(tm_values) > 0:
                print(f"  TM-score: {tm_values.mean():.3f} ± {tm_values.std():.3f}")
            
            if len(time_values) > 0:
                print(f"  Time: {time_values.mean():.3f} ± {time_values.std():.3f} s")
            
            if len(conf_values) > 0:
                print(f"  Confidence: {conf_values.mean():.3f} ± {conf_values.std():.3f}")
        
        # Results by target
        print(f"\n🎯 RESULTS BY TARGET:")
        print("=" * 25)
        
        for target in results_df['target_id'].unique():
            target_data = results_df[results_df['target_id'] == target]
            
            print(f"\n{target}:")
            for _, row in target_data.iterrows():
                rmsd = f"{row['rmsd_ca']:.3f}" if pd.notna(row['rmsd_ca']) else "N/A"
                tm = f"{row['tm_score']:.3f}" if pd.notna(row['tm_score']) else "N/A"
                time_val = f"{row['processing_time']:.3f}" if pd.notna(row['processing_time']) else "N/A"
                
                print(f"  {row['pipeline']:15} RMSD: {rmsd:>7} Å  TM: {tm:>7}  Time: {time_val:>7} s")
        
        # Performance ranking
        print(f"\n🏆 PERFORMANCE RANKING:")
        print("=" * 25)
        
        pipeline_scores = []
        for pipeline in results_df['pipeline'].unique():
            pipeline_data = results_df[results_df['pipeline'] == pipeline]
            
            rmsd_values = pipeline_data['rmsd_ca'].dropna()
            tm_values = pipeline_data['tm_score'].dropna()
            
            if len(rmsd_values) > 0 and len(tm_values) > 0:
                avg_rmsd = rmsd_values.mean()
                avg_tm = tm_values.mean()
                
                # Combined score (lower RMSD is better, higher TM is better)
                rmsd_score = 1.0 / (1.0 + avg_rmsd / 10.0)
                combined_score = (rmsd_score + avg_tm) / 2.0
                
                pipeline_scores.append({
                    'pipeline': pipeline,
                    'rmsd': avg_rmsd,
                    'tm_score': avg_tm,
                    'combined_score': combined_score
                })
        
        # Sort by combined score
        pipeline_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        
        for i, score in enumerate(pipeline_scores, 1):
            print(f"{i}. {score['pipeline']:20} Score: {score['combined_score']:.3f}")
            print(f"   RMSD: {score['rmsd']:6.3f} Å, TM: {score['tm_score']:.3f}")
        
        return pipeline_scores


def main():
    """Main benchmark function."""
    
    print("🚀 ULTIMATE COMPREHENSIVE OPENFOLD++ BENCHMARK")
    print("=" * 55)
    print("Testing EVERY SINGLE COMPONENT")
    
    # Initialize benchmark
    benchmark = UltimateComprehensiveBenchmark()
    
    # Initialize all pipelines
    benchmark.initialize_all_pipelines()
    
    # Run complete benchmark
    results_df = benchmark.run_complete_benchmark()
    
    # Analyze results
    pipeline_scores = benchmark.analyze_comprehensive_results(results_df)
    
    print(f"\n🎉 ULTIMATE COMPREHENSIVE BENCHMARK COMPLETE!")
    print(f"📁 Results: ultimate_comprehensive_results.csv")
    print(f"📊 Tested {len(results_df)} predictions across {len(results_df['pipeline'].unique())} pipelines")
    print(f"🎯 Evaluated {len(results_df['target_id'].unique())} CASP targets")
    
    return results_df, pipeline_scores


if __name__ == "__main__":
    results_df, pipeline_scores = main()
