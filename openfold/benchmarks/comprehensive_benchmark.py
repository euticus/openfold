#!/usr/bin/env python3
"""
Comprehensive benchmark suite for OpenFold++ optimizations.

This script benchmarks all major optimizations including:
- CUDA triangle kernels vs PyTorch
- Memory layout optimizations
- Real-time mutation system
- MD refinement pipeline
- Overall model performance
"""

import torch
import torch.nn as nn
import time
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import statistics
import gc

# Import OpenFold++ components
from openfold.model.cuda_triangle_ops import (
    CudaTriangleAttention,
    CudaTriangleMultiplication,
    CUDA_KERNELS_AVAILABLE
)
from openfold.utils.gpu_memory_optimization import (
    MemoryLayoutOptimizer,
    MemoryEfficientAttention,
    MemoryLayoutConfig
)
from openfold.services.optimized_mutation_server import (
    OptimizedDeltaPredictor,
    OptimizedStructureSession
)
from openfold.model.cuda_kernels_interface import kernel_manager


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str
    implementation: str
    avg_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    memory_mb: float
    peak_memory_mb: float
    throughput_ops_per_sec: float
    accuracy_score: float = 1.0
    error_message: str = ""


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    num_iterations: int = 10
    warmup_iterations: int = 3
    enable_cuda_benchmarks: bool = True
    enable_memory_benchmarks: bool = True
    enable_mutation_benchmarks: bool = True
    save_results: bool = True
    output_dir: str = "benchmark_results"
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4]
        if self.sequence_lengths is None:
            self.sequence_lengths = [64, 128, 256]


class ComprehensiveBenchmark:
    """Comprehensive benchmark suite for OpenFold++ optimizations."""
    
    def __init__(self, config: BenchmarkConfig = None):
        """
        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results: List[BenchmarkResult] = []
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        print(f"Benchmark initialized on device: {self.device}")
        print(f"CUDA kernels available: {CUDA_KERNELS_AVAILABLE}")
    
    def benchmark_triangle_operations(self) -> List[BenchmarkResult]:
        """Benchmark triangle attention and multiplication operations."""
        print("\n=== Benchmarking Triangle Operations ===")
        
        results = []
        
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                print(f"\nTesting batch_size={batch_size}, seq_len={seq_len}")
                
                # Triangle Attention Benchmark
                results.extend(self._benchmark_triangle_attention(batch_size, seq_len))
                
                # Triangle Multiplication Benchmark
                results.extend(self._benchmark_triangle_multiplication(batch_size, seq_len))
        
        return results
    
    def _benchmark_triangle_attention(self, batch_size: int, seq_len: int) -> List[BenchmarkResult]:
        """Benchmark triangle attention implementations."""
        results = []
        
        # Test parameters
        num_heads = 8
        head_dim = 64
        c_in = num_heads * head_dim
        c_hidden = 32
        
        # Create test data
        x = torch.randn(batch_size, seq_len, seq_len, c_in, device=self.device)
        mask = torch.ones(batch_size, seq_len, seq_len, device=self.device)
        
        # Benchmark PyTorch implementation (fallback)
        pytorch_attn = CudaTriangleAttention(c_in, c_hidden, num_heads).to(self.device)
        
        # Force PyTorch implementation
        original_available = CUDA_KERNELS_AVAILABLE
        import openfold.model.cuda_triangle_ops as cuda_ops
        cuda_ops.CUDA_KERNELS_AVAILABLE = False
        
        pytorch_result = self._run_benchmark(
            f"TriangleAttention_PyTorch_B{batch_size}_S{seq_len}",
            "PyTorch",
            lambda: pytorch_attn(x, mask),
            batch_size * seq_len * seq_len
        )
        results.append(pytorch_result)
        
        # Restore CUDA availability and benchmark CUDA implementation
        cuda_ops.CUDA_KERNELS_AVAILABLE = original_available
        
        if CUDA_KERNELS_AVAILABLE and self.device.type == "cuda":
            cuda_attn = CudaTriangleAttention(c_in, c_hidden, num_heads).to(self.device)
            
            cuda_result = self._run_benchmark(
                f"TriangleAttention_CUDA_B{batch_size}_S{seq_len}",
                "CUDA",
                lambda: cuda_attn(x, mask),
                batch_size * seq_len * seq_len
            )
            results.append(cuda_result)
            
            # Calculate speedup
            if pytorch_result.avg_time_ms > 0:
                speedup = pytorch_result.avg_time_ms / cuda_result.avg_time_ms
                print(f"  Triangle Attention CUDA speedup: {speedup:.2f}x")
        
        return results
    
    def _benchmark_triangle_multiplication(self, batch_size: int, seq_len: int) -> List[BenchmarkResult]:
        """Benchmark triangle multiplication implementations."""
        results = []
        
        # Test parameters
        c_in = 256
        c_hidden = 128
        
        # Create test data
        x = torch.randn(batch_size, seq_len, seq_len, c_in, device=self.device)
        mask = torch.ones(batch_size, seq_len, seq_len, device=self.device)
        
        # Benchmark PyTorch implementation
        pytorch_mult = CudaTriangleMultiplication(c_in, c_hidden).to(self.device)
        
        # Force PyTorch implementation
        original_available = CUDA_KERNELS_AVAILABLE
        import openfold.model.cuda_triangle_ops as cuda_ops
        cuda_ops.CUDA_KERNELS_AVAILABLE = False
        
        pytorch_result = self._run_benchmark(
            f"TriangleMultiply_PyTorch_B{batch_size}_S{seq_len}",
            "PyTorch",
            lambda: pytorch_mult(x, mask),
            batch_size * seq_len * seq_len
        )
        results.append(pytorch_result)
        
        # Restore and benchmark CUDA implementation
        cuda_ops.CUDA_KERNELS_AVAILABLE = original_available
        
        if CUDA_KERNELS_AVAILABLE and self.device.type == "cuda":
            cuda_mult = CudaTriangleMultiplication(c_in, c_hidden).to(self.device)
            
            cuda_result = self._run_benchmark(
                f"TriangleMultiply_CUDA_B{batch_size}_S{seq_len}",
                "CUDA",
                lambda: cuda_mult(x, mask),
                batch_size * seq_len * seq_len
            )
            results.append(cuda_result)
            
            # Calculate speedup
            if pytorch_result.avg_time_ms > 0:
                speedup = pytorch_result.avg_time_ms / cuda_result.avg_time_ms
                print(f"  Triangle Multiplication CUDA speedup: {speedup:.2f}x")
        
        return results
    
    def benchmark_memory_optimization(self) -> List[BenchmarkResult]:
        """Benchmark memory layout optimizations."""
        print("\n=== Benchmarking Memory Optimizations ===")
        
        results = []
        
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                print(f"\nTesting memory optimization batch_size={batch_size}, seq_len={seq_len}")
                
                embed_dim = 512
                num_heads = 8
                
                # Create test data
                x = torch.randn(batch_size, seq_len, embed_dim, device=self.device)
                
                # Standard attention
                std_attn = nn.MultiheadAttention(embed_dim, num_heads).to(self.device)
                
                std_result = self._run_benchmark(
                    f"Attention_Standard_B{batch_size}_S{seq_len}",
                    "Standard",
                    lambda: std_attn(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))[0],
                    batch_size * seq_len
                )
                results.append(std_result)
                
                # Memory-efficient attention
                mem_attn = MemoryEfficientAttention(embed_dim, num_heads).to(self.device)
                
                mem_result = self._run_benchmark(
                    f"Attention_MemoryOptimized_B{batch_size}_S{seq_len}",
                    "MemoryOptimized",
                    lambda: mem_attn(x, x, x),
                    batch_size * seq_len
                )
                results.append(mem_result)
                
                # Calculate memory efficiency
                if std_result.memory_mb > 0:
                    memory_reduction = (std_result.memory_mb - mem_result.memory_mb) / std_result.memory_mb
                    print(f"  Memory reduction: {memory_reduction:.1%}")
        
        return results
    
    def benchmark_mutation_system(self) -> List[BenchmarkResult]:
        """Benchmark real-time mutation system."""
        print("\n=== Benchmarking Mutation System ===")
        
        results = []
        
        # Test different protein sizes
        protein_sizes = [50, 100, 200]
        
        for size in protein_sizes:
            print(f"\nTesting mutation system with protein size: {size}")
            
            # Create test protein
            test_protein = self._create_test_protein(size)
            
            # Create optimized delta predictor
            from openfold.model.delta_predictor import create_delta_predictor
            base_predictor = create_delta_predictor(model_type="simple_gnn", hidden_dim=32, num_layers=2)
            optimized_predictor = OptimizedDeltaPredictor(base_predictor)
            
            # Create optimized session
            from openfold.services.optimized_mutation_server import OptimizedStructureSession
            session = OptimizedStructureSession(
                session_id="benchmark",
                original_structure=test_protein,
                delta_predictor=optimized_predictor
            )
            
            # Benchmark mutation prediction
            from openfold.services.websocket_server import MutationRequest
            
            def run_mutation():
                mutation_request = MutationRequest(
                    position=size // 2,
                    original_aa="A",
                    target_aa="V",
                    session_id="benchmark"
                )
                return session.apply_mutation(mutation_request)
            
            mutation_result = self._run_benchmark(
                f"MutationPrediction_Size{size}",
                "OptimizedGNN",
                run_mutation,
                1  # One mutation per operation
            )
            results.append(mutation_result)
            
            print(f"  Mutation prediction time: {mutation_result.avg_time_ms:.2f} ms")
        
        return results
    
    def _create_test_protein(self, size: int):
        """Create a test protein structure."""
        positions = np.zeros((size, 37, 3))
        
        for i in range(size):
            positions[i, 0] = [i * 3.8, 0, 0]      # N
            positions[i, 1] = [i * 3.8 + 1.5, 0, 0]  # CA
            positions[i, 2] = [i * 3.8 + 3.0, 0, 0]  # C
        
        atom_mask = np.zeros((size, 37))
        atom_mask[:, :3] = 1.0
        
        np.random.seed(42)
        aatype = np.random.randint(0, 20, size)
        residue_index = np.arange(size)
        b_factors = np.ones((size, 37)) * 50.0
        
        from openfold.np import protein
        return protein.Protein(
            atom_positions=positions,
            atom_mask=atom_mask,
            aatype=aatype,
            residue_index=residue_index,
            b_factors=b_factors
        )
    
    def _run_benchmark(self, name: str, implementation: str, 
                      operation, throughput_ops: int) -> BenchmarkResult:
        """Run a single benchmark."""
        times = []
        memory_before = 0
        memory_after = 0
        peak_memory = 0
        error_message = ""
        
        try:
            # Warmup
            for _ in range(self.config.warmup_iterations):
                _ = operation()
            
            # Clear memory and reset stats
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated() / (1024 * 1024)
            
            # Benchmark
            for i in range(self.config.num_iterations):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                result = operation()
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Memory measurement
            if self.device.type == "cuda":
                memory_after = torch.cuda.memory_allocated() / (1024 * 1024)
                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
        except Exception as e:
            error_message = str(e)
            times = [float('inf')] * self.config.num_iterations
        
        # Calculate statistics
        if times and all(t != float('inf') for t in times):
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            min_time = min(times)
            max_time = max(times)
            throughput = throughput_ops / (avg_time / 1000) if avg_time > 0 else 0
        else:
            avg_time = std_time = min_time = max_time = throughput = 0
        
        return BenchmarkResult(
            name=name,
            implementation=implementation,
            avg_time_ms=avg_time,
            std_time_ms=std_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            memory_mb=memory_after - memory_before,
            peak_memory_mb=peak_memory,
            throughput_ops_per_sec=throughput,
            error_message=error_message
        )
    
    def run_all_benchmarks(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all benchmark suites."""
        print("Starting comprehensive OpenFold++ benchmark suite...")
        
        all_results = {}
        
        # Triangle operations
        if self.config.enable_cuda_benchmarks:
            triangle_results = self.benchmark_triangle_operations()
            all_results["triangle_operations"] = triangle_results
            self.results.extend(triangle_results)
        
        # Memory optimizations
        if self.config.enable_memory_benchmarks:
            memory_results = self.benchmark_memory_optimization()
            all_results["memory_optimization"] = memory_results
            self.results.extend(memory_results)
        
        # Mutation system
        if self.config.enable_mutation_benchmarks:
            mutation_results = self.benchmark_mutation_system()
            all_results["mutation_system"] = mutation_results
            self.results.extend(mutation_results)
        
        # Save results
        if self.config.save_results:
            self._save_results(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict[str, List[BenchmarkResult]]):
        """Save benchmark results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_results = {}
        for category, result_list in results.items():
            json_results[category] = [asdict(r) for r in result_list]
        
        json_file = os.path.join(self.config.output_dir, f"benchmark_results_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {json_file}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report."""
        report = []
        report.append("=" * 80)
        report.append("OPENFOLD++ COMPREHENSIVE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Device: {self.device}")
        report.append(f"CUDA Kernels Available: {CUDA_KERNELS_AVAILABLE}")
        report.append(f"Total Benchmarks: {len(self.results)}")
        report.append("")
        
        # Group results by category
        categories = {}
        for result in self.results:
            category = result.name.split('_')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        # Generate category reports
        for category, results in categories.items():
            report.append(f"=== {category.upper()} BENCHMARKS ===")
            report.append("")
            
            # Find best and worst performers
            valid_results = [r for r in results if r.avg_time_ms > 0 and not r.error_message]
            
            if valid_results:
                fastest = min(valid_results, key=lambda x: x.avg_time_ms)
                slowest = max(valid_results, key=lambda x: x.avg_time_ms)
                
                report.append(f"Fastest: {fastest.name} ({fastest.implementation})")
                report.append(f"  Time: {fastest.avg_time_ms:.2f} ± {fastest.std_time_ms:.2f} ms")
                report.append(f"  Throughput: {fastest.throughput_ops_per_sec:.1f} ops/sec")
                report.append("")
                
                report.append(f"Slowest: {slowest.name} ({slowest.implementation})")
                report.append(f"  Time: {slowest.avg_time_ms:.2f} ± {slowest.std_time_ms:.2f} ms")
                report.append(f"  Throughput: {slowest.throughput_ops_per_sec:.1f} ops/sec")
                report.append("")
                
                # Calculate speedups
                pytorch_results = [r for r in valid_results if r.implementation == "PyTorch"]
                cuda_results = [r for r in valid_results if r.implementation == "CUDA"]
                
                if pytorch_results and cuda_results:
                    avg_pytorch_time = statistics.mean([r.avg_time_ms for r in pytorch_results])
                    avg_cuda_time = statistics.mean([r.avg_time_ms for r in cuda_results])
                    speedup = avg_pytorch_time / avg_cuda_time if avg_cuda_time > 0 else 0
                    
                    report.append(f"Average CUDA Speedup: {speedup:.2f}x")
                    report.append("")
            
            # Detailed results table
            report.append("Detailed Results:")
            report.append("-" * 80)
            report.append(f"{'Name':<40} {'Impl':<15} {'Time (ms)':<12} {'Memory (MB)':<12}")
            report.append("-" * 80)
            
            for result in results:
                if result.error_message:
                    time_str = "ERROR"
                    memory_str = "N/A"
                else:
                    time_str = f"{result.avg_time_ms:.2f}"
                    memory_str = f"{result.memory_mb:.1f}"
                
                report.append(f"{result.name:<40} {result.implementation:<15} {time_str:<12} {memory_str:<12}")
            
            report.append("")
        
        # Summary statistics
        report.append("=== SUMMARY STATISTICS ===")
        report.append("")
        
        successful_benchmarks = [r for r in self.results if not r.error_message and r.avg_time_ms > 0]
        failed_benchmarks = [r for r in self.results if r.error_message or r.avg_time_ms == 0]
        
        report.append(f"Successful benchmarks: {len(successful_benchmarks)}")
        report.append(f"Failed benchmarks: {len(failed_benchmarks)}")
        
        if successful_benchmarks:
            avg_time = statistics.mean([r.avg_time_ms for r in successful_benchmarks])
            total_memory = sum([r.memory_mb for r in successful_benchmarks])
            
            report.append(f"Average execution time: {avg_time:.2f} ms")
            report.append(f"Total memory usage: {total_memory:.1f} MB")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main benchmark function."""
    print("OpenFold++ Comprehensive Benchmark Suite")
    print("=" * 50)
    
    # Configure benchmark
    config = BenchmarkConfig(
        batch_sizes=[1, 2],
        sequence_lengths=[64, 128],
        num_iterations=5,  # Reduced for faster testing
        warmup_iterations=2
    )
    
    # Run benchmarks
    benchmark = ComprehensiveBenchmark(config)
    results = benchmark.run_all_benchmarks()
    
    # Generate and display report
    report = benchmark.generate_report()
    print("\n" + report)
    
    # Save report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(config.output_dir, f"benchmark_report_{timestamp}.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nFull report saved to: {report_file}")
    
    return results


if __name__ == "__main__":
    main()
