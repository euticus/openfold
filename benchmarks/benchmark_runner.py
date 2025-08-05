"""
Main Benchmark Runner for OdinFold

Orchestrates comprehensive benchmarking including performance, accuracy,
and comparative analysis against reference methods.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import torch
import psutil

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

from .casp_validator import CASPValidator
from .performance_profiler import PerformanceProfiler
from .mutation_benchmark import MutationBenchmark
from .web_backend_test import WebBackendTester
from .comparative_analysis import ComparativeAnalyzer
from .metrics import calculate_tm_score, calculate_rmsd

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    # General settings
    output_dir: str = "benchmark_results"
    model_path: str = "models/odinfold.pt"
    device: str = "auto"
    batch_size: int = 1
    max_sequence_length: int = 1024
    
    # Test datasets
    casp_dataset_path: Optional[str] = None
    custom_dataset_path: Optional[str] = None
    
    # Benchmark types to run
    run_accuracy_tests: bool = True
    run_performance_tests: bool = True
    run_mutation_tests: bool = True
    run_web_backend_tests: bool = True
    run_comparative_analysis: bool = True
    
    # Performance test settings
    warmup_runs: int = 3
    benchmark_runs: int = 10
    profile_memory: bool = True
    profile_gpu: bool = True
    
    # Accuracy test settings
    tm_score_threshold: float = 0.65
    rmsd_threshold: float = 5.0
    
    # Mutation test settings
    mutation_scan_proteins: int = 10
    mutations_per_protein: int = 100
    
    # Web backend test settings
    concurrent_users: int = 10
    requests_per_user: int = 50
    backend_url: str = "http://localhost:8000"
    
    # Comparative analysis
    compare_with_alphafold: bool = True
    alphafold_results_path: Optional[str] = None


class BenchmarkRunner:
    """Main benchmark runner orchestrating all tests."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.casp_validator = CASPValidator() if config.run_accuracy_tests else None
        self.profiler = PerformanceProfiler() if config.run_performance_tests else None
        self.mutation_benchmark = MutationBenchmark() if config.run_mutation_tests else None
        self.web_tester = WebBackendTester(config.backend_url) if config.run_web_backend_tests else None
        self.comparative_analyzer = ComparativeAnalyzer() if config.run_comparative_analysis else None
        
        # Results storage
        self.results = {
            'config': asdict(config),
            'system_info': self._get_system_info(),
            'accuracy_results': {},
            'performance_results': {},
            'mutation_results': {},
            'web_backend_results': {},
            'comparative_results': {},
            'summary': {}
        }
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for benchmark runs."""
        log_file = self.output_dir / "benchmark.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': torch.__version__,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            
            # Get GPU info
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    info['gpus'] = [
                        {
                            'name': gpu.name,
                            'memory_mb': gpu.memoryTotal,
                            'driver_version': gpu.driver
                        }
                        for gpu in gpus
                    ]
                except:
                    info['gpus'] = []
            else:
                info['gpus'] = []
        
        return info
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all configured benchmarks."""
        logger.info("Starting comprehensive OdinFold benchmark suite")
        start_time = time.time()
        
        try:
            # Run accuracy tests
            if self.config.run_accuracy_tests:
                logger.info("Running accuracy benchmarks...")
                self.results['accuracy_results'] = self._run_accuracy_benchmarks()
            
            # Run performance tests
            if self.config.run_performance_tests:
                logger.info("Running performance benchmarks...")
                self.results['performance_results'] = self._run_performance_benchmarks()
            
            # Run mutation tests
            if self.config.run_mutation_tests:
                logger.info("Running mutation benchmarks...")
                self.results['mutation_results'] = self._run_mutation_benchmarks()
            
            # Run web backend tests
            if self.config.run_web_backend_tests:
                logger.info("Running web backend tests...")
                self.results['web_backend_results'] = self._run_web_backend_tests()
            
            # Run comparative analysis
            if self.config.run_comparative_analysis:
                logger.info("Running comparative analysis...")
                self.results['comparative_results'] = self._run_comparative_analysis()
            
            # Generate summary
            self.results['summary'] = self._generate_summary()
            
            total_time = time.time() - start_time
            self.results['total_benchmark_time'] = total_time
            
            logger.info(f"Benchmark suite completed in {total_time:.2f} seconds")
            
            # Save results
            self._save_results()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise
    
    def _run_accuracy_benchmarks(self) -> Dict[str, Any]:
        """Run accuracy validation benchmarks."""
        results = {}
        
        if self.config.casp_dataset_path:
            logger.info("Validating on CASP dataset...")
            casp_results = self.casp_validator.validate_dataset(
                self.config.casp_dataset_path,
                model_path=self.config.model_path,
                device=self.config.device
            )
            results['casp'] = casp_results
        
        if self.config.custom_dataset_path:
            logger.info("Validating on custom dataset...")
            custom_results = self.casp_validator.validate_dataset(
                self.config.custom_dataset_path,
                model_path=self.config.model_path,
                device=self.config.device
            )
            results['custom'] = custom_results
        
        return results
    
    def _run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        results = {}
        
        # Test different sequence lengths
        sequence_lengths = [64, 128, 256, 512, 1024]
        if self.config.max_sequence_length < 1024:
            sequence_lengths = [l for l in sequence_lengths if l <= self.config.max_sequence_length]
        
        for seq_len in sequence_lengths:
            logger.info(f"Benchmarking sequence length {seq_len}...")
            
            # Generate mock sequence
            sequence = 'A' * seq_len
            
            # Profile performance
            profile_result = self.profiler.profile_inference(
                sequence=sequence,
                model_path=self.config.model_path,
                device=self.config.device,
                warmup_runs=self.config.warmup_runs,
                benchmark_runs=self.config.benchmark_runs,
                profile_memory=self.config.profile_memory,
                profile_gpu=self.config.profile_gpu
            )
            
            results[f'seq_len_{seq_len}'] = profile_result
        
        # Batch size scaling
        if self.config.batch_size > 1:
            logger.info("Benchmarking batch processing...")
            batch_results = self.profiler.profile_batch_inference(
                sequences=['A' * 256] * self.config.batch_size,
                model_path=self.config.model_path,
                device=self.config.device
            )
            results['batch_processing'] = batch_results
        
        return results
    
    def _run_mutation_benchmarks(self) -> Dict[str, Any]:
        """Run mutation scanning benchmarks."""
        results = {}
        
        # Test mutation scanning performance
        logger.info("Benchmarking mutation scanning...")
        mutation_results = self.mutation_benchmark.benchmark_mutation_scanning(
            num_proteins=self.config.mutation_scan_proteins,
            mutations_per_protein=self.config.mutations_per_protein,
            model_path=self.config.model_path,
            device=self.config.device
        )
        results['mutation_scanning'] = mutation_results
        
        # Test ΔΔG prediction accuracy
        if hasattr(self.mutation_benchmark, 'validate_ddg_predictions'):
            logger.info("Validating ΔΔG prediction accuracy...")
            ddg_results = self.mutation_benchmark.validate_ddg_predictions()
            results['ddg_validation'] = ddg_results
        
        return results
    
    def _run_web_backend_tests(self) -> Dict[str, Any]:
        """Run web backend load tests."""
        results = {}
        
        # Test basic functionality
        logger.info("Testing web backend functionality...")
        functionality_results = self.web_tester.test_functionality()
        results['functionality'] = functionality_results
        
        # Load testing
        logger.info("Running web backend load tests...")
        load_results = self.web_tester.run_load_test(
            concurrent_users=self.config.concurrent_users,
            requests_per_user=self.config.requests_per_user
        )
        results['load_test'] = load_results
        
        return results
    
    def _run_comparative_analysis(self) -> Dict[str, Any]:
        """Run comparative analysis with other methods."""
        results = {}
        
        if self.config.compare_with_alphafold and self.config.alphafold_results_path:
            logger.info("Comparing with AlphaFold results...")
            comparison = self.comparative_analyzer.compare_with_alphafold(
                odinfold_results=self.results.get('accuracy_results', {}),
                alphafold_results_path=self.config.alphafold_results_path
            )
            results['alphafold_comparison'] = comparison
        
        return results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary."""
        summary = {
            'timestamp': time.time(),
            'status': 'completed',
            'tests_run': []
        }
        
        # Accuracy summary
        if self.results.get('accuracy_results'):
            accuracy_data = self.results['accuracy_results']
            if 'casp' in accuracy_data:
                casp_data = accuracy_data['casp']
                summary['mean_tm_score'] = np.mean([r['tm_score'] for r in casp_data.get('results', [])])
                summary['mean_rmsd'] = np.mean([r['rmsd'] for r in casp_data.get('results', [])])
            summary['tests_run'].append('accuracy')
        
        # Performance summary
        if self.results.get('performance_results'):
            perf_data = self.results['performance_results']
            # Find fastest inference time
            inference_times = []
            memory_usage = []
            
            for key, result in perf_data.items():
                if isinstance(result, dict) and 'inference_time_ms' in result:
                    inference_times.append(result['inference_time_ms'])
                if isinstance(result, dict) and 'peak_memory_mb' in result:
                    memory_usage.append(result['peak_memory_mb'])
            
            if inference_times:
                summary['fastest_inference_ms'] = min(inference_times)
                summary['mean_inference_ms'] = np.mean(inference_times)
            
            if memory_usage:
                summary['peak_memory_mb'] = max(memory_usage)
                summary['mean_memory_mb'] = np.mean(memory_usage)
            
            summary['tests_run'].append('performance')
        
        # Mutation summary
        if self.results.get('mutation_results'):
            mut_data = self.results['mutation_results']
            if 'mutation_scanning' in mut_data:
                scan_data = mut_data['mutation_scanning']
                summary['mutations_per_second'] = scan_data.get('mutations_per_second', 0)
            summary['tests_run'].append('mutation')
        
        # Web backend summary
        if self.results.get('web_backend_results'):
            web_data = self.results['web_backend_results']
            if 'load_test' in web_data:
                load_data = web_data['load_test']
                summary['requests_per_second'] = load_data.get('requests_per_second', 0)
                summary['mean_response_time_ms'] = load_data.get('mean_response_time_ms', 0)
            summary['tests_run'].append('web_backend')
        
        # Overall assessment
        summary['overall_status'] = self._assess_overall_performance(summary)
        
        return summary
    
    def _assess_overall_performance(self, summary: Dict[str, Any]) -> str:
        """Assess overall performance based on thresholds."""
        issues = []
        
        # Check TM-score threshold
        if 'mean_tm_score' in summary:
            if summary['mean_tm_score'] < self.config.tm_score_threshold:
                issues.append(f"TM-score below threshold: {summary['mean_tm_score']:.3f} < {self.config.tm_score_threshold}")
        
        # Check RMSD threshold
        if 'mean_rmsd' in summary:
            if summary['mean_rmsd'] > self.config.rmsd_threshold:
                issues.append(f"RMSD above threshold: {summary['mean_rmsd']:.3f} > {self.config.rmsd_threshold}")
        
        # Check performance
        if 'mean_inference_ms' in summary:
            if summary['mean_inference_ms'] > 10000:  # 10 seconds
                issues.append(f"Slow inference: {summary['mean_inference_ms']:.0f}ms")
        
        if issues:
            return f"ISSUES_FOUND: {'; '.join(issues)}"
        else:
            return "PASSED"
    
    def _save_results(self):
        """Save benchmark results to files."""
        # Save JSON results
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary CSV
        if self.results.get('summary'):
            summary_df = pd.DataFrame([self.results['summary']])
            summary_file = self.output_dir / "benchmark_summary.csv"
            summary_df.to_csv(summary_file, index=False)
        
        # Save detailed performance data
        if self.results.get('performance_results'):
            perf_data = []
            for test_name, result in self.results['performance_results'].items():
                if isinstance(result, dict):
                    row = {'test_name': test_name}
                    row.update(result)
                    perf_data.append(row)
            
            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                perf_file = self.output_dir / "performance_details.csv"
                perf_df.to_csv(perf_file, index=False)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def generate_report(self) -> str:
        """Generate a human-readable benchmark report."""
        report = []
        report.append("# OdinFold Benchmark Report")
        report.append(f"Generated at: {time.ctime()}")
        report.append("")
        
        # System info
        report.append("## System Information")
        sys_info = self.results['system_info']
        report.append(f"- CPU cores: {sys_info['cpu_count']}")
        report.append(f"- Memory: {sys_info['memory_gb']:.1f} GB")
        report.append(f"- PyTorch: {sys_info['torch_version']}")
        report.append(f"- CUDA available: {sys_info['cuda_available']}")
        
        if sys_info.get('gpus'):
            report.append("- GPUs:")
            for gpu in sys_info['gpus']:
                report.append(f"  - {gpu['name']} ({gpu['memory_mb']} MB)")
        report.append("")
        
        # Summary
        summary = self.results.get('summary', {})
        report.append("## Summary")
        report.append(f"- Overall status: {summary.get('overall_status', 'Unknown')}")
        report.append(f"- Tests run: {', '.join(summary.get('tests_run', []))}")
        
        if 'mean_tm_score' in summary:
            report.append(f"- Mean TM-score: {summary['mean_tm_score']:.3f}")
        if 'mean_rmsd' in summary:
            report.append(f"- Mean RMSD: {summary['mean_rmsd']:.3f} Å")
        if 'fastest_inference_ms' in summary:
            report.append(f"- Fastest inference: {summary['fastest_inference_ms']:.0f} ms")
        if 'mutations_per_second' in summary:
            report.append(f"- Mutation scanning: {summary['mutations_per_second']:.1f} mutations/sec")
        
        report.append("")
        
        # Detailed results would go here...
        
        return "\n".join(report)
