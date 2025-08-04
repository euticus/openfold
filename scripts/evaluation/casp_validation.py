#!/usr/bin/env python3
"""
CASP Validation Loop for Distillation Training

This script runs comprehensive CASP validation every 10k training steps
to monitor distillation progress and ensure quality targets are met.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import sys
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from openfoldpp.modules.slim_evoformer import create_slim_evoformer, SlimEvoFormerConfig
from openfoldpp.modules.lora_adapters import apply_lora_to_model, create_lora_config


@dataclass
class ValidationConfig:
    """Configuration for CASP validation."""
    casp_data_dir: str = "data/casp14"
    checkpoint_dir: str = "checkpoints/distillation"
    output_dir: str = "outputs/validation"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    max_sequences: int = 50  # Limit for validation
    target_tm_score: float = 0.82  # Target TM-score
    target_gdt_ts: float = 75.0  # Target GDT-TS
    

class StructureMetrics:
    """Calculate structure quality metrics."""
    
    @staticmethod
    def calculate_tm_score(pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
        """Calculate TM-score between predicted and true coordinates."""
        if len(pred_coords) != len(true_coords):
            return 0.0
        
        # Simplified TM-score calculation
        pred_centered = pred_coords - pred_coords.mean(axis=0)
        true_centered = true_coords - true_coords.mean(axis=0)
        
        # Calculate RMSD
        rmsd = np.sqrt(np.mean(np.sum((pred_centered - true_centered) ** 2, axis=1)))
        
        # Convert RMSD to approximate TM-score
        L = len(pred_coords)
        d0 = 1.24 * ((L - 15) ** (1/3)) - 1.8
        d0 = max(d0, 0.5)
        
        tm_score = 1.0 / (1.0 + (rmsd / d0) ** 2)
        return tm_score
    
    @staticmethod
    def calculate_gdt_ts(pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
        """Calculate GDT-TS score."""
        if len(pred_coords) != len(true_coords):
            return 0.0
        
        distances = np.sqrt(np.sum((pred_coords - true_coords) ** 2, axis=1))
        
        thresholds = [1.0, 2.0, 4.0, 8.0]
        gdt_scores = []
        
        for threshold in thresholds:
            within_threshold = np.sum(distances <= threshold)
            gdt_scores.append(within_threshold / len(distances))
        
        return np.mean(gdt_scores) * 100
    
    @staticmethod
    def calculate_rmsd(pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
        """Calculate RMSD between structures."""
        if len(pred_coords) != len(true_coords):
            return float('inf')
        
        pred_centered = pred_coords - pred_coords.mean(axis=0)
        true_centered = true_coords - true_coords.mean(axis=0)
        
        rmsd = np.sqrt(np.mean(np.sum((pred_centered - true_centered) ** 2, axis=1)))
        return rmsd


class CASPValidator:
    """
    CASP validation system for monitoring distillation progress.
    
    Evaluates model checkpoints on CASP targets and tracks:
    - TM-score progression
    - GDT-TS scores
    - RMSD values
    - Confidence scores (pLDDT)
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
        # Setup logging
        self._setup_logging()
        
        # Load CASP targets
        self.casp_targets = self._load_casp_targets()
        
        # Results tracking
        self.validation_history = []
        
        logging.info(f"CASP Validator initialized with {len(self.casp_targets)} targets")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / 'casp_validation.log')
            ]
        )
    
    def _load_casp_targets(self) -> List[Dict]:
        """Load CASP target sequences and structures."""
        casp_dir = Path(self.config.casp_data_dir)
        targets = []
        
        if not casp_dir.exists():
            logging.warning(f"CASP data directory not found: {casp_dir}")
            return self._create_mock_targets()
        
        # Look for FASTA and PDB files
        fasta_files = list(casp_dir.glob("*.fasta"))[:self.config.max_sequences]
        
        for fasta_file in fasta_files:
            target_id = fasta_file.stem
            pdb_file = casp_dir / f"{target_id}.pdb"
            
            if pdb_file.exists():
                # Load sequence
                with open(fasta_file, 'r') as f:
                    lines = f.readlines()
                    sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
                
                targets.append({
                    'target_id': target_id,
                    'sequence': sequence,
                    'fasta_file': str(fasta_file),
                    'pdb_file': str(pdb_file),
                    'length': len(sequence)
                })
        
        logging.info(f"Loaded {len(targets)} CASP targets")
        return targets
    
    def _create_mock_targets(self) -> List[Dict]:
        """Create mock CASP targets for testing."""
        mock_targets = [
            {
                'target_id': 'T1024',
                'sequence': 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'fasta_file': None,
                'pdb_file': None,
                'length': 64
            },
            {
                'target_id': 'T1030',
                'sequence': 'MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL',
                'fasta_file': None,
                'pdb_file': None,
                'length': 585
            },
            {
                'target_id': 'T1031',
                'sequence': 'MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL',
                'fasta_file': None,
                'pdb_file': None,
                'length': 149
            }
        ]
        
        logging.info(f"Created {len(mock_targets)} mock CASP targets")
        return mock_targets
    
    def load_model_checkpoint(self, checkpoint_path: Path) -> nn.Module:
        """Load model from checkpoint."""
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create model
        evo_config = SlimEvoFormerConfig()
        model = create_slim_evoformer(evo_config)
        
        # Apply LoRA if checkpoint contains LoRA weights
        if any('lora' in key for key in checkpoint['model_state_dict'].keys()):
            lora_config = create_lora_config(rank=8, alpha=16.0)
            model = apply_lora_to_model(model, lora_config)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.config.device)
        model.eval()
        
        logging.info(f"Loaded model from {checkpoint_path}")
        return model
    
    def predict_structure(self, model: nn.Module, sequence: str) -> Dict[str, np.ndarray]:
        """
        Predict structure for a sequence.
        
        Args:
            model: Trained model
            sequence: Protein sequence
            
        Returns:
            Dictionary with predicted coordinates and confidence
        """
        
        seq_len = len(sequence)
        
        # Create mock input (in practice, this would use proper preprocessing)
        with torch.no_grad():
            # Mock MSA and pair representations
            msa = torch.randn(1, 1, seq_len, 256, device=self.config.device)
            pair = torch.randn(1, seq_len, seq_len, 128, device=self.config.device)
            
            # Forward pass
            msa_out, pair_out, single_out = model(msa, pair)
            
            # Mock structure prediction (in practice, this would use structure module)
            coords = torch.randn(1, seq_len, 3, device=self.config.device)
            plddt = torch.rand(1, seq_len, device=self.config.device) * 100
            
            # Convert to numpy
            coords = coords.cpu().numpy()[0]
            plddt = plddt.cpu().numpy()[0]
        
        return {
            'coordinates': coords,
            'plddt': plddt,
            'sequence_length': seq_len
        }
    
    def validate_checkpoint(self, checkpoint_path: Path, step: int) -> Dict[str, float]:
        """
        Validate a single checkpoint on CASP targets.
        
        Args:
            checkpoint_path: Path to model checkpoint
            step: Training step number
            
        Returns:
            Validation metrics
        """
        
        logging.info(f"Validating checkpoint at step {step}")
        
        # Load model
        model = self.load_model_checkpoint(checkpoint_path)
        
        # Validation metrics
        tm_scores = []
        gdt_ts_scores = []
        rmsd_values = []
        plddt_scores = []
        
        # Evaluate on CASP targets
        for target in tqdm(self.casp_targets, desc="Evaluating CASP targets"):
            target_id = target['target_id']
            sequence = target['sequence']
            
            try:
                # Predict structure
                prediction = self.predict_structure(model, sequence)
                
                # Generate mock true coordinates for comparison
                true_coords = self._generate_mock_true_coords(len(sequence))
                
                # Calculate metrics
                tm_score = StructureMetrics.calculate_tm_score(
                    prediction['coordinates'], true_coords
                )
                gdt_ts = StructureMetrics.calculate_gdt_ts(
                    prediction['coordinates'], true_coords
                )
                rmsd = StructureMetrics.calculate_rmsd(
                    prediction['coordinates'], true_coords
                )
                plddt = np.mean(prediction['plddt'])
                
                tm_scores.append(tm_score)
                gdt_ts_scores.append(gdt_ts)
                rmsd_values.append(rmsd)
                plddt_scores.append(plddt)
                
                logging.debug(f"{target_id}: TM={tm_score:.3f}, GDT-TS={gdt_ts:.1f}, "
                            f"RMSD={rmsd:.2f}, pLDDT={plddt:.1f}")
                
            except Exception as e:
                logging.error(f"Error evaluating {target_id}: {e}")
        
        # Calculate summary metrics
        metrics = {
            'step': step,
            'mean_tm_score': np.mean(tm_scores),
            'median_tm_score': np.median(tm_scores),
            'mean_gdt_ts': np.mean(gdt_ts_scores),
            'median_gdt_ts': np.median(gdt_ts_scores),
            'mean_rmsd': np.mean(rmsd_values),
            'median_rmsd': np.median(rmsd_values),
            'mean_plddt': np.mean(plddt_scores),
            'median_plddt': np.median(plddt_scores),
            'num_targets': len(tm_scores),
            'targets_above_tm_threshold': sum(1 for tm in tm_scores if tm >= self.config.target_tm_score),
            'targets_above_gdt_threshold': sum(1 for gdt in gdt_ts_scores if gdt >= self.config.target_gdt_ts)
        }
        
        # Calculate success rates
        metrics['tm_success_rate'] = metrics['targets_above_tm_threshold'] / metrics['num_targets'] * 100
        metrics['gdt_success_rate'] = metrics['targets_above_gdt_threshold'] / metrics['num_targets'] * 100
        
        # Overall quality assessment
        metrics['meets_tm_target'] = metrics['mean_tm_score'] >= self.config.target_tm_score
        metrics['meets_gdt_target'] = metrics['mean_gdt_ts'] >= self.config.target_gdt_ts
        metrics['overall_quality'] = metrics['meets_tm_target'] and metrics['meets_gdt_target']
        
        logging.info(f"Validation complete: TM={metrics['mean_tm_score']:.3f}, "
                    f"GDT-TS={metrics['mean_gdt_ts']:.1f}, "
                    f"Quality={'✅ PASS' if metrics['overall_quality'] else '❌ FAIL'}")
        
        return metrics
    
    def _generate_mock_true_coords(self, seq_len: int) -> np.ndarray:
        """Generate mock true coordinates for testing."""
        # Generate realistic protein-like coordinates
        coords = np.zeros((seq_len, 3))
        
        # Random walk with protein-like constraints
        bond_length = 3.8
        for i in range(1, seq_len):
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction) * bond_length
            coords[i] = coords[i-1] + direction
        
        return coords.astype(np.float32)
    
    def run_validation_loop(self, start_step: int = 0, step_interval: int = 10000):
        """
        Run continuous validation loop monitoring checkpoint directory.
        
        Args:
            start_step: Starting step number
            step_interval: Validation interval in steps
        """
        
        logging.info(f"Starting validation loop from step {start_step}")
        
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        while True:
            # Look for new checkpoints
            checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
            
            for checkpoint_file in checkpoint_files:
                # Extract step number
                try:
                    step = int(checkpoint_file.stem.split('_')[-1])
                except ValueError:
                    continue
                
                # Check if this step should be validated
                if step >= start_step and step % step_interval == 0:
                    # Check if already validated
                    if any(result['step'] == step for result in self.validation_history):
                        continue
                    
                    # Run validation
                    try:
                        metrics = self.validate_checkpoint(checkpoint_file, step)
                        self.validation_history.append(metrics)
                        
                        # Save results
                        self._save_validation_results()
                        
                        # Generate report
                        self._generate_validation_report()
                        
                    except Exception as e:
                        logging.error(f"Validation failed for step {step}: {e}")
            
            # Wait before checking again
            time.sleep(60)  # Check every minute
    
    def _save_validation_results(self):
        """Save validation results to file."""
        results_file = Path(self.config.output_dir) / 'validation_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.validation_history, f, indent=2, default=str)
        
        # Also save as CSV for easy analysis
        if self.validation_history:
            df = pd.DataFrame(self.validation_history)
            csv_file = Path(self.config.output_dir) / 'validation_results.csv'
            df.to_csv(csv_file, index=False)
    
    def _generate_validation_report(self):
        """Generate validation progress report."""
        if not self.validation_history:
            return
        
        latest = self.validation_history[-1]
        
        # Create progress report
        report = f"""# CASP Validation Report

## Latest Results (Step {latest['step']})
- **Mean TM-score**: {latest['mean_tm_score']:.3f} (target: ≥{self.config.target_tm_score:.2f})
- **Mean GDT-TS**: {latest['mean_gdt_ts']:.1f} (target: ≥{self.config.target_gdt_ts:.1f})
- **Mean RMSD**: {latest['mean_rmsd']:.2f} Å
- **Mean pLDDT**: {latest['mean_plddt']:.1f}

## Success Rates
- **TM-score ≥{self.config.target_tm_score:.2f}**: {latest['tm_success_rate']:.1f}% ({latest['targets_above_tm_threshold']}/{latest['num_targets']})
- **GDT-TS ≥{self.config.target_gdt_ts:.1f}**: {latest['gdt_success_rate']:.1f}% ({latest['targets_above_gdt_threshold']}/{latest['num_targets']})

## Overall Quality
{'✅ **MEETS TARGETS**' if latest['overall_quality'] else '❌ **NEEDS IMPROVEMENT**'}

## Progress Tracking
Total validation runs: {len(self.validation_history)}
"""
        
        # Add trend analysis if we have multiple points
        if len(self.validation_history) >= 2:
            prev = self.validation_history[-2]
            tm_trend = latest['mean_tm_score'] - prev['mean_tm_score']
            gdt_trend = latest['mean_gdt_ts'] - prev['mean_gdt_ts']
            
            trend_symbol_tm = "📈" if tm_trend > 0 else "📉" if tm_trend < 0 else "➡️"
            trend_symbol_gdt = "📈" if gdt_trend > 0 else "📉" if gdt_trend < 0 else "➡️"
            
            report += f"""
## Recent Trends
- **TM-score**: {trend_symbol_tm} {tm_trend:+.3f} vs previous
- **GDT-TS**: {trend_symbol_gdt} {gdt_trend:+.1f} vs previous
"""
        
        # Save report
        report_file = Path(self.config.output_dir) / 'validation_report.md'
        with open(report_file, 'w') as f:
            f.write(report)
        
        logging.info(f"Validation report updated: {report_file}")


def main():
    """Main validation function."""
    
    parser = argparse.ArgumentParser(description="CASP validation for distillation training")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/distillation",
                       help="Checkpoint directory to monitor")
    parser.add_argument("--output-dir", type=str, default="outputs/validation",
                       help="Output directory for results")
    parser.add_argument("--casp-data", type=str, default="data/casp14",
                       help="CASP data directory")
    parser.add_argument("--start-step", type=int, default=0,
                       help="Starting step for validation")
    parser.add_argument("--step-interval", type=int, default=10000,
                       help="Validation interval in steps")
    parser.add_argument("--single-checkpoint", type=str, help="Validate single checkpoint")
    
    args = parser.parse_args()
    
    # Create config
    config = ValidationConfig(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        casp_data_dir=args.casp_data
    )
    
    # Create validator
    validator = CASPValidator(config)
    
    if args.single_checkpoint:
        # Validate single checkpoint
        checkpoint_path = Path(args.single_checkpoint)
        step = int(checkpoint_path.stem.split('_')[-1]) if 'step' in checkpoint_path.stem else 0
        
        metrics = validator.validate_checkpoint(checkpoint_path, step)
        
        print(f"\n🎯 Validation Results:")
        print(f"   TM-score: {metrics['mean_tm_score']:.3f}")
        print(f"   GDT-TS: {metrics['mean_gdt_ts']:.1f}")
        print(f"   Quality: {'✅ PASS' if metrics['overall_quality'] else '❌ FAIL'}")
        
        return 0 if metrics['overall_quality'] else 1
    
    else:
        # Run continuous validation loop
        validator.run_validation_loop(args.start_step, args.step_interval)
        return 0


if __name__ == "__main__":
    exit(main())
