"""
MMseqs2-GPU integration for OpenFold++.

This module provides fast preprocessing and homology search capabilities
using MMseqs2-GPU to replace traditional MSA construction methods.
"""

import os
import subprocess
import tempfile
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import json
import time

# Try to import BioPython for sequence handling
try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    logging.warning("BioPython not available. Some features may be limited.")


@dataclass
class MMseqs2Config:
    """Configuration for MMseqs2-GPU operations."""
    mmseqs_binary: str = "mmseqs"
    database_path: str = ""
    sensitivity: float = 7.5
    coverage: float = 0.8
    max_seqs: int = 10000
    min_seq_id: float = 0.0
    max_seq_id: float = 1.0
    e_value: float = 0.001
    iterations: int = 3
    use_gpu: bool = True
    gpu_id: int = 0
    threads: int = 8
    tmp_dir: str = "/tmp/mmseqs2"
    keep_tmp_files: bool = False


@dataclass
class HomologyHit:
    """Single homology search hit."""
    target_id: str
    query_id: str
    sequence: str
    e_value: float
    bit_score: float
    seq_identity: float
    query_coverage: float
    target_coverage: float
    alignment_length: int
    query_start: int
    query_end: int
    target_start: int
    target_end: int


@dataclass
class HomologySearchResult:
    """Results from homology search."""
    query_id: str
    query_sequence: str
    hits: List[HomologyHit]
    search_time_seconds: float
    total_hits: int
    filtered_hits: int


class MMseqs2GPU:
    """MMseqs2-GPU integration for fast homology search."""
    
    def __init__(self, config: MMseqs2Config = None):
        """
        Args:
            config: MMseqs2 configuration
        """
        self.config = config or MMseqs2Config()
        self.logger = logging.getLogger(__name__)
        
        # Validate MMseqs2 installation
        self._validate_installation()
        
        # Setup temporary directory
        os.makedirs(self.config.tmp_dir, exist_ok=True)
    
    def _validate_installation(self):
        """Validate MMseqs2 installation and GPU support."""
        try:
            # Check if MMseqs2 is installed
            result = subprocess.run(
                [self.config.mmseqs_binary, "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"MMseqs2 not found at {self.config.mmseqs_binary}")
            
            version_info = result.stdout.strip()
            self.logger.info(f"MMseqs2 version: {version_info}")
            
            # Check GPU support if requested
            if self.config.use_gpu:
                gpu_result = subprocess.run(
                    [self.config.mmseqs_binary, "search", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if "--gpu" not in gpu_result.stdout:
                    self.logger.warning("GPU support not detected in MMseqs2. Falling back to CPU.")
                    self.config.use_gpu = False
                else:
                    self.logger.info("MMseqs2 GPU support detected")
            
        except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError) as e:
            self.logger.error(f"MMseqs2 validation failed: {e}")
            raise RuntimeError(f"MMseqs2 installation invalid: {e}")
    
    def create_database(self, 
                       sequences: Union[str, List[str], Dict[str, str]],
                       database_name: str) -> str:
        """
        Create MMseqs2 database from sequences.
        
        Args:
            sequences: Input sequences (FASTA file path, list of sequences, or dict)
            database_name: Name for the database
            
        Returns:
            Path to created database
        """
        db_path = os.path.join(self.config.tmp_dir, database_name)
        
        # Create temporary FASTA file if needed
        if isinstance(sequences, str) and os.path.exists(sequences):
            # Input is a FASTA file path
            fasta_path = sequences
        else:
            # Create FASTA file from sequences
            fasta_path = os.path.join(self.config.tmp_dir, f"{database_name}.fasta")
            self._write_fasta(sequences, fasta_path)
        
        # Create MMseqs2 database
        cmd = [
            self.config.mmseqs_binary,
            "createdb",
            fasta_path,
            db_path
        ]
        
        self.logger.info(f"Creating MMseqs2 database: {database_name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Database creation failed: {result.stderr}")
        
        self.logger.info(f"Database created successfully: {db_path}")
        return db_path
    
    def search_homologs(self, 
                       query_sequences: Union[str, List[str], Dict[str, str]],
                       database_path: Optional[str] = None) -> List[HomologySearchResult]:
        """
        Search for homologous sequences using MMseqs2-GPU.
        
        Args:
            query_sequences: Query sequences (FASTA file, list, or dict)
            database_path: Path to search database (uses config default if None)
            
        Returns:
            List of homology search results
        """
        if database_path is None:
            database_path = self.config.database_path
        
        if not database_path or not os.path.exists(database_path):
            raise ValueError(f"Database path not found: {database_path}")
        
        start_time = time.time()
        
        # Create query database
        query_db = self.create_database(query_sequences, "query_db")
        
        # Create result database path
        result_db = os.path.join(self.config.tmp_dir, "search_results")
        
        # Build search command
        cmd = [
            self.config.mmseqs_binary,
            "search",
            query_db,
            database_path,
            result_db,
            self.config.tmp_dir,
            "--sensitivity", str(self.config.sensitivity),
            "--coverage", str(self.config.coverage),
            "--max-seqs", str(self.config.max_seqs),
            "--min-seq-id", str(self.config.min_seq_id),
            "--max-seq-id", str(self.config.max_seq_id),
            "-e", str(self.config.e_value),
            "--num-iterations", str(self.config.iterations),
            "--threads", str(self.config.threads)
        ]
        
        # Add GPU support if available
        if self.config.use_gpu:
            cmd.extend(["--gpu", str(self.config.gpu_id)])
        
        self.logger.info("Starting MMseqs2 homology search...")
        self.logger.debug(f"Search command: {' '.join(cmd)}")
        
        # Run search
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"MMseqs2 search failed: {result.stderr}")
        
        search_time = time.time() - start_time
        self.logger.info(f"Search completed in {search_time:.2f} seconds")
        
        # Convert results to readable format
        results = self._parse_search_results(result_db, query_sequences, search_time)
        
        # Cleanup temporary files if requested
        if not self.config.keep_tmp_files:
            self._cleanup_tmp_files([query_db, result_db])
        
        return results
    
    def _write_fasta(self, sequences: Union[List[str], Dict[str, str]], output_path: str):
        """Write sequences to FASTA file."""
        with open(output_path, 'w') as f:
            if isinstance(sequences, dict):
                for seq_id, sequence in sequences.items():
                    f.write(f">{seq_id}\n{sequence}\n")
            elif isinstance(sequences, list):
                for i, sequence in enumerate(sequences):
                    f.write(f">seq_{i}\n{sequence}\n")
            else:
                raise ValueError("Sequences must be dict or list")
    
    def _parse_search_results(self, 
                             result_db: str, 
                             query_sequences: Union[str, List[str], Dict[str, str]],
                             search_time: float) -> List[HomologySearchResult]:
        """Parse MMseqs2 search results."""
        # Convert results to TSV format
        tsv_path = os.path.join(self.config.tmp_dir, "results.tsv")
        
        cmd = [
            self.config.mmseqs_binary,
            "convertalis",
            os.path.join(self.config.tmp_dir, "query_db"),
            self.config.database_path,
            result_db,
            tsv_path,
            "--format-output", "query,target,evalue,bitscore,pident,qcov,tcov,alnlen,qstart,qend,tstart,tend"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            self.logger.warning(f"Result conversion failed: {result.stderr}")
            return []
        
        # Parse TSV results
        results = []
        current_query = None
        current_hits = []
        
        try:
            with open(tsv_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split('\t')
                        if len(parts) >= 12:
                            query_id = parts[0]
                            
                            # Start new query result
                            if current_query != query_id:
                                if current_query is not None:
                                    # Save previous query results
                                    query_seq = self._get_query_sequence(current_query, query_sequences)
                                    results.append(HomologySearchResult(
                                        query_id=current_query,
                                        query_sequence=query_seq,
                                        hits=current_hits,
                                        search_time_seconds=search_time,
                                        total_hits=len(current_hits),
                                        filtered_hits=len(current_hits)
                                    ))
                                
                                current_query = query_id
                                current_hits = []
                            
                            # Parse hit
                            hit = HomologyHit(
                                target_id=parts[1],
                                query_id=parts[0],
                                sequence="",  # Would need additional lookup
                                e_value=float(parts[2]),
                                bit_score=float(parts[3]),
                                seq_identity=float(parts[4]),
                                query_coverage=float(parts[5]),
                                target_coverage=float(parts[6]),
                                alignment_length=int(parts[7]),
                                query_start=int(parts[8]),
                                query_end=int(parts[9]),
                                target_start=int(parts[10]),
                                target_end=int(parts[11])
                            )
                            
                            current_hits.append(hit)
                
                # Add final query
                if current_query is not None:
                    query_seq = self._get_query_sequence(current_query, query_sequences)
                    results.append(HomologySearchResult(
                        query_id=current_query,
                        query_sequence=query_seq,
                        hits=current_hits,
                        search_time_seconds=search_time,
                        total_hits=len(current_hits),
                        filtered_hits=len(current_hits)
                    ))
        
        except FileNotFoundError:
            self.logger.warning("Results file not found")
            return []
        
        return results
    
    def _get_query_sequence(self, query_id: str, query_sequences) -> str:
        """Get query sequence by ID."""
        if isinstance(query_sequences, dict):
            return query_sequences.get(query_id, "")
        elif isinstance(query_sequences, list):
            try:
                idx = int(query_id.split('_')[1])
                return query_sequences[idx] if idx < len(query_sequences) else ""
            except (ValueError, IndexError):
                return ""
        else:
            return ""
    
    def _cleanup_tmp_files(self, paths: List[str]):
        """Clean up temporary files."""
        for path in paths:
            try:
                if os.path.exists(path):
                    if os.path.isdir(path):
                        import shutil
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
            except OSError as e:
                self.logger.warning(f"Failed to cleanup {path}: {e}")
    
    def create_msa_from_homologs(self, 
                                homology_results: List[HomologySearchResult],
                                max_sequences: int = 1000) -> Dict[str, List[str]]:
        """
        Create MSA-like data from homology search results.
        
        Args:
            homology_results: Results from homology search
            max_sequences: Maximum sequences per query
            
        Returns:
            Dictionary mapping query IDs to lists of homologous sequences
        """
        msa_data = {}
        
        for result in homology_results:
            sequences = [result.query_sequence]  # Start with query
            
            # Add homologous sequences
            sorted_hits = sorted(result.hits, key=lambda x: x.e_value)
            
            for hit in sorted_hits[:max_sequences-1]:
                if hit.sequence:  # Only add if sequence is available
                    sequences.append(hit.sequence)
            
            msa_data[result.query_id] = sequences
        
        return msa_data
    
    def get_statistics(self) -> Dict[str, any]:
        """Get MMseqs2 integration statistics."""
        return {
            "mmseqs_binary": self.config.mmseqs_binary,
            "gpu_enabled": self.config.use_gpu,
            "database_path": self.config.database_path,
            "sensitivity": self.config.sensitivity,
            "max_seqs": self.config.max_seqs,
            "threads": self.config.threads,
            "tmp_dir": self.config.tmp_dir
        }


def create_mmseqs2_pipeline(database_path: str, 
                           use_gpu: bool = True,
                           sensitivity: float = 7.5) -> MMseqs2GPU:
    """
    Create a configured MMseqs2-GPU pipeline.
    
    Args:
        database_path: Path to MMseqs2 database
        use_gpu: Whether to use GPU acceleration
        sensitivity: Search sensitivity (higher = more sensitive)
        
    Returns:
        Configured MMseqs2GPU instance
    """
    config = MMseqs2Config(
        database_path=database_path,
        use_gpu=use_gpu,
        sensitivity=sensitivity
    )
    
    return MMseqs2GPU(config)


def benchmark_mmseqs2_performance(sequences: List[str], 
                                 database_path: str,
                                 num_iterations: int = 3) -> Dict[str, float]:
    """
    Benchmark MMseqs2-GPU performance.
    
    Args:
        sequences: Test sequences
        database_path: Database to search against
        num_iterations: Number of benchmark iterations
        
    Returns:
        Performance statistics
    """
    mmseqs = create_mmseqs2_pipeline(database_path, use_gpu=True)
    
    times = []
    
    for i in range(num_iterations):
        start_time = time.time()
        results = mmseqs.search_homologs(sequences)
        elapsed = time.time() - start_time
        times.append(elapsed)
    
    return {
        "avg_time_seconds": np.mean(times),
        "std_time_seconds": np.std(times),
        "min_time_seconds": np.min(times),
        "max_time_seconds": np.max(times),
        "sequences_per_second": len(sequences) / np.mean(times)
    }


# Integration with OpenFold++ pipeline
class OpenFoldMMseqs2Preprocessor:
    """MMseqs2-based preprocessor for OpenFold++ pipeline."""

    def __init__(self,
                 database_path: str,
                 config: MMseqs2Config = None):
        """
        Args:
            database_path: Path to MMseqs2 database
            config: MMseqs2 configuration
        """
        self.mmseqs = MMseqs2GPU(config or MMseqs2Config(database_path=database_path))
        self.logger = logging.getLogger(__name__)

    def preprocess_sequences(self,
                           sequences: Union[str, List[str], Dict[str, str]],
                           max_msa_depth: int = 1000) -> Dict[str, any]:
        """
        Preprocess sequences for OpenFold++ using MMseqs2.

        Args:
            sequences: Input protein sequences
            max_msa_depth: Maximum MSA depth per sequence

        Returns:
            Preprocessed data ready for OpenFold++
        """
        self.logger.info("Starting MMseqs2 preprocessing for OpenFold++")

        # Search for homologs
        homology_results = self.mmseqs.search_homologs(sequences)

        # Create MSA-like data
        msa_data = self.mmseqs.create_msa_from_homologs(homology_results, max_msa_depth)

        # Format for OpenFold++
        processed_data = {
            "msa_data": msa_data,
            "homology_results": homology_results,
            "preprocessing_stats": {
                "total_queries": len(homology_results),
                "total_hits": sum(len(r.hits) for r in homology_results),
                "avg_hits_per_query": np.mean([len(r.hits) for r in homology_results]) if homology_results else 0,
                "total_search_time": sum(r.search_time_seconds for r in homology_results)
            }
        }

        self.logger.info(f"Preprocessing completed: {processed_data['preprocessing_stats']}")

        return processed_data
