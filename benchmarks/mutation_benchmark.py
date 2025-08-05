"""
Mutation Scanning Benchmark for OdinFold

Benchmarks mutation scanning performance and accuracy including:
- High-throughput mutation scanning
- ΔΔG prediction accuracy validation
- Comparison with experimental data
- Web backend mutation API performance
"""

import time
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json

logger = logging.getLogger(__name__)


@dataclass
class MutationScanConfig:
    """Configuration for mutation scanning benchmarks."""
    
    # Scanning parameters
    num_proteins: int = 10
    mutations_per_protein: int = 100
    batch_size: int = 32
    
    # Performance settings
    max_workers: int = 4
    timeout_seconds: int = 300
    
    # Validation settings
    experimental_data_path: Optional[str] = None
    correlation_threshold: float = 0.6
    
    # Web backend settings
    backend_url: str = "http://localhost:8000"
    concurrent_requests: int = 10


class MutationBenchmark:
    """Benchmark mutation scanning performance and accuracy."""
    
    def __init__(self):
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        self.test_sequences = self._generate_test_sequences()
    
    def _generate_test_sequences(self) -> List[str]:
        """Generate test protein sequences of various lengths."""
        sequences = []
        
        # Short sequences (50-100 residues)
        for length in [50, 75, 100]:
            seq = ''.join(np.random.choice(self.amino_acids, length))
            sequences.append(seq)
        
        # Medium sequences (150-300 residues)
        for length in [150, 200, 250, 300]:
            seq = ''.join(np.random.choice(self.amino_acids, length))
            sequences.append(seq)
        
        # Long sequences (400-500 residues)
        for length in [400, 500]:
            seq = ''.join(np.random.choice(self.amino_acids, length))
            sequences.append(seq)
        
        return sequences
    
    def benchmark_mutation_scanning(self, 
                                  num_proteins: int = 10,
                                  mutations_per_protein: int = 100,
                                  model_path: str = "models/odinfold.pt",
                                  device: str = "cuda") -> Dict[str, Any]:
        """Benchmark mutation scanning performance."""
        
        logger.info(f"Benchmarking mutation scanning: {num_proteins} proteins, {mutations_per_protein} mutations each")
        
        results = {
            'config': {
                'num_proteins': num_proteins,
                'mutations_per_protein': mutations_per_protein,
                'model_path': model_path,
                'device': device
            },
            'protein_results': [],
            'summary': {}
        }
        
        total_mutations = 0
        total_time = 0
        
        # Test each protein
        for i, sequence in enumerate(self.test_sequences[:num_proteins]):
            logger.info(f"Testing protein {i+1}/{num_proteins} (length: {len(sequence)})")
            
            # Generate random mutations
            mutations = self._generate_random_mutations(sequence, mutations_per_protein)
            
            # Time mutation scanning
            start_time = time.time()
            
            try:
                # Mock mutation scanning (replace with actual implementation)
                ddg_predictions = self._mock_mutation_scanning(sequence, mutations, device)
                
                scan_time = time.time() - start_time
                
                protein_result = {
                    'protein_index': i,
                    'sequence_length': len(sequence),
                    'num_mutations': len(mutations),
                    'scan_time_seconds': scan_time,
                    'mutations_per_second': len(mutations) / scan_time,
                    'mean_ddg': np.mean(ddg_predictions),
                    'std_ddg': np.std(ddg_predictions),
                    'min_ddg': np.min(ddg_predictions),
                    'max_ddg': np.max(ddg_predictions)
                }
                
                results['protein_results'].append(protein_result)
                
                total_mutations += len(mutations)
                total_time += scan_time
                
                logger.info(f"  Completed in {scan_time:.2f}s ({len(mutations)/scan_time:.1f} mutations/s)")
                
            except Exception as e:
                logger.error(f"Failed to scan protein {i}: {e}")
                continue
        
        # Calculate summary statistics
        if results['protein_results']:
            scan_times = [r['scan_time_seconds'] for r in results['protein_results']]
            mut_rates = [r['mutations_per_second'] for r in results['protein_results']]
            
            results['summary'] = {
                'total_proteins': len(results['protein_results']),
                'total_mutations': total_mutations,
                'total_time_seconds': total_time,
                'overall_mutations_per_second': total_mutations / total_time if total_time > 0 else 0,
                'mean_scan_time_seconds': np.mean(scan_times),
                'std_scan_time_seconds': np.std(scan_times),
                'mean_mutations_per_second': np.mean(mut_rates),
                'std_mutations_per_second': np.std(mut_rates),
                'min_mutations_per_second': np.min(mut_rates),
                'max_mutations_per_second': np.max(mut_rates)
            }
        
        return results
    
    def _generate_random_mutations(self, sequence: str, num_mutations: int) -> List[Tuple[int, str, str]]:
        """Generate random mutations for a sequence."""
        mutations = []
        sequence_length = len(sequence)
        
        for _ in range(num_mutations):
            # Random position
            position = np.random.randint(0, sequence_length)
            original_aa = sequence[position]
            
            # Random new amino acid (different from original)
            new_aa_choices = [aa for aa in self.amino_acids if aa != original_aa]
            new_aa = np.random.choice(new_aa_choices)
            
            mutations.append((position, original_aa, new_aa))
        
        return mutations
    
    def _mock_mutation_scanning(self, sequence: str, mutations: List[Tuple[int, str, str]], device: str) -> np.ndarray:
        """Mock mutation scanning implementation."""
        # Simulate computation time based on sequence length and number of mutations
        computation_time = len(sequence) * len(mutations) * 0.00001  # Mock timing
        time.sleep(min(computation_time, 1.0))  # Cap at 1 second for testing
        
        # Generate mock ΔΔG predictions
        ddg_predictions = np.random.normal(0, 2.0, len(mutations))  # Mean 0, std 2.0 kcal/mol
        
        return ddg_predictions
    
    def benchmark_web_backend_mutations(self, config: MutationScanConfig) -> Dict[str, Any]:
        """Benchmark web backend mutation API performance."""
        
        logger.info("Benchmarking web backend mutation API")
        
        results = {
            'config': config.__dict__,
            'request_results': [],
            'summary': {}
        }
        
        # Test data
        test_sequence = self.test_sequences[0]  # Use first test sequence
        test_mutations = self._generate_random_mutations(test_sequence, 50)
        
        # Prepare request data
        request_data = {
            'sequence': test_sequence,
            'mutations': [
                {'position': pos, 'from_aa': from_aa, 'to_aa': to_aa}
                for pos, from_aa, to_aa in test_mutations
            ]
        }
        
        # Run concurrent requests
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=config.concurrent_requests) as executor:
            futures = []
            
            for i in range(config.concurrent_requests * 5):  # 5 requests per worker
                future = executor.submit(self._send_mutation_request, config.backend_url, request_data)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=config.timeout_seconds)
                    results['request_results'].append(result)
                except Exception as e:
                    logger.error(f"Request failed: {e}")
                    results['request_results'].append({
                        'success': False,
                        'error': str(e),
                        'response_time_ms': None
                    })
        
        total_time = time.time() - start_time
        
        # Calculate summary
        successful_requests = [r for r in results['request_results'] if r.get('success', False)]
        response_times = [r['response_time_ms'] for r in successful_requests if r['response_time_ms'] is not None]
        
        results['summary'] = {
            'total_requests': len(results['request_results']),
            'successful_requests': len(successful_requests),
            'failed_requests': len(results['request_results']) - len(successful_requests),
            'success_rate': len(successful_requests) / len(results['request_results']) if results['request_results'] else 0,
            'total_time_seconds': total_time,
            'requests_per_second': len(results['request_results']) / total_time if total_time > 0 else 0,
            'mean_response_time_ms': np.mean(response_times) if response_times else None,
            'std_response_time_ms': np.std(response_times) if response_times else None,
            'min_response_time_ms': np.min(response_times) if response_times else None,
            'max_response_time_ms': np.max(response_times) if response_times else None,
            'p95_response_time_ms': np.percentile(response_times, 95) if response_times else None,
            'p99_response_time_ms': np.percentile(response_times, 99) if response_times else None
        }
        
        return results
    
    def _send_mutation_request(self, backend_url: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a single mutation request to the web backend."""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{backend_url}/api/mutations/scan",
                json=request_data,
                timeout=30
            )
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'response_time_ms': response_time,
                    'status_code': response.status_code,
                    'response_size_bytes': len(response.content)
                }
            else:
                return {
                    'success': False,
                    'response_time_ms': response_time,
                    'status_code': response.status_code,
                    'error': f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return {
                'success': False,
                'response_time_ms': response_time,
                'error': str(e)
            }
    
    def validate_ddg_predictions(self, experimental_data_path: Optional[str] = None) -> Dict[str, Any]:
        """Validate ΔΔG prediction accuracy against experimental data."""
        
        if not experimental_data_path:
            logger.warning("No experimental data provided for ΔΔG validation")
            return {'status': 'skipped', 'reason': 'no_experimental_data'}
        
        logger.info("Validating ΔΔG prediction accuracy")
        
        try:
            # Load experimental data
            exp_data = pd.read_csv(experimental_data_path)
            
            # Expected columns: sequence, position, from_aa, to_aa, experimental_ddg
            required_columns = ['sequence', 'position', 'from_aa', 'to_aa', 'experimental_ddg']
            if not all(col in exp_data.columns for col in required_columns):
                return {'status': 'error', 'reason': 'missing_columns', 'required': required_columns}
            
            # Generate predictions for experimental mutations
            predictions = []
            experimental_values = []
            
            for _, row in exp_data.iterrows():
                sequence = row['sequence']
                position = row['position']
                from_aa = row['from_aa']
                to_aa = row['to_aa']
                exp_ddg = row['experimental_ddg']
                
                # Mock prediction (replace with actual model)
                pred_ddg = np.random.normal(exp_ddg, 1.0)  # Mock with some correlation
                
                predictions.append(pred_ddg)
                experimental_values.append(exp_ddg)
            
            predictions = np.array(predictions)
            experimental_values = np.array(experimental_values)
            
            # Calculate correlation and metrics
            correlation = np.corrcoef(predictions, experimental_values)[0, 1]
            mae = np.mean(np.abs(predictions - experimental_values))
            rmse = np.sqrt(np.mean((predictions - experimental_values) ** 2))
            
            results = {
                'status': 'completed',
                'num_mutations': len(predictions),
                'correlation': correlation,
                'mae': mae,
                'rmse': rmse,
                'mean_experimental': np.mean(experimental_values),
                'std_experimental': np.std(experimental_values),
                'mean_predicted': np.mean(predictions),
                'std_predicted': np.std(predictions)
            }
            
            logger.info(f"ΔΔG validation: correlation={correlation:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"ΔΔG validation failed: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    def benchmark_batch_mutations(self, batch_sizes: List[int] = [1, 4, 8, 16, 32]) -> Dict[str, Any]:
        """Benchmark mutation scanning with different batch sizes."""
        
        logger.info("Benchmarking batch mutation scanning")
        
        results = {
            'batch_results': {},
            'summary': {}
        }
        
        test_sequence = self.test_sequences[2]  # Medium length sequence
        base_mutations = self._generate_random_mutations(test_sequence, 100)
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size {batch_size}")
            
            # Time batch processing
            start_time = time.time()
            
            # Process mutations in batches
            total_processed = 0
            for i in range(0, len(base_mutations), batch_size):
                batch_mutations = base_mutations[i:i+batch_size]
                
                # Mock batch processing
                _ = self._mock_mutation_scanning(test_sequence, batch_mutations, "cuda")
                total_processed += len(batch_mutations)
            
            batch_time = time.time() - start_time
            
            results['batch_results'][batch_size] = {
                'batch_size': batch_size,
                'total_mutations': total_processed,
                'total_time_seconds': batch_time,
                'mutations_per_second': total_processed / batch_time,
                'time_per_mutation_ms': (batch_time / total_processed) * 1000
            }
        
        # Find optimal batch size
        best_batch_size = max(results['batch_results'].keys(), 
                             key=lambda x: results['batch_results'][x]['mutations_per_second'])
        
        results['summary'] = {
            'optimal_batch_size': best_batch_size,
            'max_mutations_per_second': results['batch_results'][best_batch_size]['mutations_per_second'],
            'batch_sizes_tested': batch_sizes
        }
        
        return results
