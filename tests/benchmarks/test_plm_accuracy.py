#!/usr/bin/env python3
"""
PLM vs MSA Accuracy Benchmark

This script benchmarks the accuracy drop when replacing MSA with PLM embeddings.
Target: TM-score loss ≤ 0.04 on CASP targets.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import time
from typing import Dict, List, Tuple, Optional
import argparse
from dataclasses import dataclass
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from openfoldpp.models.esm_wrapper import create_esm_wrapper
from openfoldpp.modules.plm_projection import create_plm_projector
from openfoldpp.pipelines.complete_pipeline import FullOpenFoldModel

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import DSSP
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    logging.warning("BioPython not available for structure analysis")


@dataclass
class BenchmarkConfig:
    """Configuration for PLM accuracy benchmark."""
    casp_data_dir: str = "data/casp14"
    output_dir: str = "results/benchmarks/plm_accuracy"
    model_weights: str = "data/weights/openfold_model.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    max_sequences: int = 10  # Limit for testing
    plm_model: str = "esm2_t33_650M_UR50D"
    projection_type: str = "linear"
    quantize_plm: bool = True


class StructureMetrics:
    """Calculate structure comparison metrics."""

    @staticmethod
    def calculate_tm_score(pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
        """
        Calculate TM-score between predicted and true coordinates.

        Args:
            pred_coords: Predicted coordinates [N, 3]
            true_coords: True coordinates [N, 3]

        Returns:
            TM-score (0-1, higher is better)
        """
        # Simplified TM-score calculation
        # In practice, you'd use TMalign or similar

        if len(pred_coords) != len(true_coords):
            return 0.0

        # Align structures (simplified)
        pred_centered = pred_coords - pred_coords.mean(axis=0)
        true_centered = true_coords - true_coords.mean(axis=0)

        # Calculate RMSD
        rmsd = np.sqrt(np.mean(np.sum((pred_centered - true_centered) ** 2, axis=1)))

        # Convert RMSD to approximate TM-score
        # TM-score ≈ 1 / (1 + (RMSD/d0)^2) where d0 ≈ 1.24 * (L-15)^(1/3) - 1.8
        L = len(pred_coords)
        d0 = 1.24 * ((L - 15) ** (1/3)) - 1.8
        d0 = max(d0, 0.5)  # Minimum d0

        tm_score = 1.0 / (1.0 + (rmsd / d0) ** 2)

        return tm_score

    @staticmethod
    def calculate_gdt_ts(pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
        """
        Calculate GDT-TS score.

        Args:
            pred_coords: Predicted coordinates [N, 3]
            true_coords: True coordinates [N, 3]

        Returns:
            GDT-TS score (0-100)
        """
        if len(pred_coords) != len(true_coords):
            return 0.0

        # Calculate distances
        distances = np.sqrt(np.sum((pred_coords - true_coords) ** 2, axis=1))

        # Count residues within distance thresholds
        thresholds = [1.0, 2.0, 4.0, 8.0]  # Angstroms
        gdt_scores = []

        for threshold in thresholds:
            within_threshold = np.sum(distances <= threshold)
            gdt_scores.append(within_threshold / len(distances))

        # GDT-TS is average of the four scores
        gdt_ts = np.mean(gdt_scores) * 100

        return gdt_ts

    @staticmethod
    def calculate_rmsd(pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
        """Calculate RMSD between structures."""
        if len(pred_coords) != len(true_coords):
            return float('inf')

        # Center structures
        pred_centered = pred_coords - pred_coords.mean(axis=0)
        true_centered = true_coords - true_coords.mean(axis=0)

        # Calculate RMSD
        rmsd = np.sqrt(np.mean(np.sum((pred_centered - true_centered) ** 2, axis=1)))

        return rmsd


class PLMAccuracyBenchmark:
    """Benchmark PLM vs MSA accuracy on CASP targets."""

    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.results = []
        self._setup_logging()
        self._load_models()

    def _setup_logging(self):
        """Setup logging for benchmark."""
        log_dir = Path(self.config.output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / 'plm_benchmark.log')
            ]
        )

    def _load_models(self):
        """Load ESM and projector models."""
        logging.info("Loading models...")

        # Load ESM wrapper
        self.esm_wrapper = create_esm_wrapper(
            model_name=self.config.plm_model,
            device=self.config.device,
            quantize=self.config.quantize_plm
        )

        # Load PLM projector
        self.plm_projector = create_plm_projector(
            projection_type=self.config.projection_type
        ).to(self.config.device)

        logging.info("Models loaded successfully")

    def run_benchmark(self) -> Dict:
        """Run the complete PLM accuracy benchmark."""
        logging.info("🚀 Starting PLM Accuracy Benchmark")

        # Load test sequences
        test_sequences = self._load_test_sequences()

        # Run benchmarks
        results = {
            'embedding_quality': self._test_embedding_quality(test_sequences),
            'projection_accuracy': self._test_projection_accuracy(test_sequences),
            'memory_efficiency': self._test_memory_efficiency(test_sequences),
            'inference_speed': self._test_inference_speed(test_sequences)
        }

        # Calculate overall metrics
        results['summary'] = self._calculate_summary_metrics(results)

        # Save results
        self._save_results(results)

        logging.info("✅ PLM Accuracy Benchmark Complete")
        return results

    def load_casp_targets(self) -> List[Dict]:
        """Load CASP target sequences and structures."""
        casp_dir = Path(self.config.casp_data_dir)
        targets = []

        if not casp_dir.exists():
            logging.warning(f"CASP data directory not found: {casp_dir}")
            return self._create_mock_targets()

        # Look for FASTA and PDB files
        fasta_files = list(casp_dir.glob("*.fasta"))
        pdb_files = list(casp_dir.glob("*.pdb"))

        for fasta_file in fasta_files[:self.config.max_sequences]:
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
                    'pdb_file': str(pdb_file)
                })

        logging.info(f"Loaded {len(targets)} CASP targets")
        return targets

    def _create_mock_targets(self) -> List[Dict]:
        """Create mock targets for testing."""
        mock_targets = [
            {
                'target_id': 'T1024',
                'sequence': 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'fasta_file': None,
                'pdb_file': None
            },
            {
                'target_id': 'T1030',
                'sequence': 'MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL',
                'fasta_file': None,
                'pdb_file': None
            }
        ]

        logging.info(f"Created {len(mock_targets)} mock targets")
        return mock_targets