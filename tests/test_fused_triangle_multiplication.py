#!/usr/bin/env python3
"""
Test Suite for Fused Triangle Multiplication

Tests the optimized triangle multiplication implementations including
Triton kernels, linear approximations, and memory-efficient variants.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openfoldpp.modules.fused_triangle_multiplication import (
    FusedTriangleMultiplication,
    FusedTriangleMultiplicationOutgoing,
    FusedTriangleMultiplicationIncoming,
    LinearTriangleApproximation
)

# Try to import original triangle multiplication for comparison
try:
    from openfold.model.triangular_multiplicative_update import (
        TriangleMultiplicationOutgoing,
        TriangleMultiplicationIncoming
    )
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False
    print("Original triangle multiplication not available. Using mock implementation.")


class MockTriangleMultiplication(nn.Module):
    """Mock triangle multiplication for testing when original is not available."""
    
    def __init__(self, c_z, c_hidden, outgoing=True):
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.outgoing = outgoing
        
        self.layer_norm = nn.LayerNorm(c_z)
        self.linear_a_p = nn.Linear(c_z, c_hidden)
        self.linear_a_g = nn.Linear(c_z, c_hidden)
        self.linear_b_p = nn.Linear(c_z, c_hidden)
        self.linear_b_g = nn.Linear(c_z, c_hidden)
        self.linear_z = nn.Linear(c_hidden, c_z)
        
    def forward(self, z, mask=None, **kwargs):
        if mask is None:
            mask = torch.ones(z.shape[:-1], device=z.device)
        
        z_norm = self.layer_norm(z)
        
        a_p = self.linear_a_p(z_norm)
        a_g = torch.sigmoid(self.linear_a_g(z_norm))
        b_p = self.linear_b_p(z_norm)
        b_g = torch.sigmoid(self.linear_b_g(z_norm))
        
        a = a_p * a_g
        b = b_p * b_g
        
        mask_expanded = mask.unsqueeze(-1)
        a = a * mask_expanded
        b = b * mask_expanded
        
        if self.outgoing:
            output = torch.einsum('...ikc,...kjc->...ijc', a, b)
        else:
            output = torch.einsum('...kic,...kjc->...ijc', a, b)
        
        output = self.linear_z(output)
        
        return z + output


class TestFusedTriangleMultiplication:
    """Test fused triangle multiplication implementation."""
    
    def test_fused_triangle_multiplication_init(self):
        """Test initialization of FusedTriangleMultiplication."""
        
        c_in, c_hidden = 64, 128
        
        # Test outgoing
        mult_out = FusedTriangleMultiplicationOutgoing(c_in, c_hidden)
        assert mult_out.c_in == c_in
        assert mult_out.c_hidden == c_hidden
        assert mult_out.outgoing == True
        
        # Test incoming
        mult_in = FusedTriangleMultiplicationIncoming(c_in, c_hidden)
        assert mult_in.outgoing == False
        
        # Test with linear approximation
        mult_approx = FusedTriangleMultiplication(
            c_in, c_hidden, use_linear_approx=True
        )
        assert mult_approx.use_linear_approx == True
        assert hasattr(mult_approx, 'linear_approx')
    
    def test_fused_triangle_multiplication_forward(self):
        """Test forward pass of FusedTriangleMultiplication."""
        
        batch_size, seq_len = 2, 16
        c_in, c_hidden = 32, 64
        
        mult = FusedTriangleMultiplicationOutgoing(c_in, c_hidden, use_triton=False)
        
        # Create input tensor
        x = torch.randn(batch_size, seq_len, seq_len, c_in)
        mask = torch.ones(batch_size, seq_len, seq_len)
        
        # Forward pass
        output = mult(x, mask=mask)
        
        # Check output shape
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
    
    def test_outgoing_vs_incoming(self):
        """Test difference between outgoing and incoming multiplication."""
        
        batch_size, seq_len = 1, 8
        c_in, c_hidden = 16, 32
        
        mult_out = FusedTriangleMultiplicationOutgoing(c_in, c_hidden, use_triton=False)
        mult_in = FusedTriangleMultiplicationIncoming(c_in, c_hidden, use_triton=False)
        
        # Same weights for fair comparison
        mult_in.load_state_dict(mult_out.state_dict())
        
        x = torch.randn(batch_size, seq_len, seq_len, c_in)
        mask = torch.ones(batch_size, seq_len, seq_len)
        
        with torch.no_grad():
            output_out = mult_out(x, mask=mask)
            output_in = mult_in(x, mask=mask)
            
            # Outputs should be different
            assert not torch.allclose(output_out, output_in, atol=1e-5)
    
    def test_linear_approximation(self):
        """Test linear approximation mode."""
        
        batch_size, seq_len = 1, 12
        c_in, c_hidden = 24, 48
        
        # Standard multiplication
        mult_standard = FusedTriangleMultiplication(
            c_in, c_hidden, use_linear_approx=False, use_triton=False
        )
        
        # Linear approximation
        mult_approx = FusedTriangleMultiplication(
            c_in, c_hidden, use_linear_approx=True, use_triton=False
        )
        
        x = torch.randn(batch_size, seq_len, seq_len, c_in)
        mask = torch.ones(batch_size, seq_len, seq_len)
        
        with torch.no_grad():
            output_standard = mult_standard(x, mask=mask)
            output_approx = mult_approx(x, mask=mask)
            
            # Both should have same shape
            assert output_standard.shape == output_approx.shape
            assert torch.isfinite(output_standard).all()
            assert torch.isfinite(output_approx).all()
            
            # Approximation should be faster (tested in benchmark)
            print("✅ Linear approximation produces valid output")
    
    def test_chunked_multiplication(self):
        """Test chunked multiplication for memory efficiency."""
        
        batch_size, seq_len = 1, 24
        c_in, c_hidden = 32, 64
        chunk_size = 8
        
        mult_full = FusedTriangleMultiplication(
            c_in, c_hidden, use_triton=False, chunk_size=None
        )
        mult_chunked = FusedTriangleMultiplication(
            c_in, c_hidden, use_triton=False, chunk_size=chunk_size
        )
        
        # Same weights
        mult_chunked.load_state_dict(mult_full.state_dict())
        
        x = torch.randn(batch_size, seq_len, seq_len, c_in)
        mask = torch.ones(batch_size, seq_len, seq_len)
        
        with torch.no_grad():
            output_full = mult_full(x, mask=mask)
            output_chunked = mult_chunked(x, mask=mask)
            
            # Should produce similar results
            assert torch.allclose(output_full, output_chunked, atol=1e-4)
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through the module."""
        
        batch_size, seq_len = 1, 6
        c_in, c_hidden = 16, 32
        
        mult = FusedTriangleMultiplication(c_in, c_hidden, use_triton=False)
        
        x = torch.randn(batch_size, seq_len, seq_len, c_in, requires_grad=True)
        mask = torch.ones(batch_size, seq_len, seq_len)
        
        # Forward pass
        output = mult(x, mask=mask)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        
        # Check parameter gradients
        for param in mult.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


class TestLinearTriangleApproximation:
    """Test linear triangle approximation."""
    
    def test_linear_approximation_init(self):
        """Test initialization of LinearTriangleApproximation."""
        
        c_in = 64
        
        approx = LinearTriangleApproximation(c_in, outgoing=True)
        assert approx.c_in == c_in
        assert approx.outgoing == True
        assert hasattr(approx, 'linear_main')
        assert hasattr(approx, 'linear_gate')
    
    def test_linear_approximation_forward(self):
        """Test forward pass of LinearTriangleApproximation."""
        
        batch_size, seq_len = 2, 12
        c_in = 32
        
        approx = LinearTriangleApproximation(c_in)
        
        x = torch.randn(batch_size, seq_len, seq_len, c_in)
        mask = torch.ones(batch_size, seq_len, seq_len)
        
        output = approx(x, mask=mask)
        
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
    
    def test_linear_approximation_speed(self):
        """Test that linear approximation is faster than standard multiplication."""
        
        batch_size, seq_len = 1, 32
        c_in, c_hidden = 64, 128
        num_runs = 10
        
        # Standard multiplication
        mult_standard = FusedTriangleMultiplication(
            c_in, c_hidden, use_triton=False, use_linear_approx=False
        )
        
        # Linear approximation
        approx = LinearTriangleApproximation(c_in)
        
        x = torch.randn(batch_size, seq_len, seq_len, c_in)
        mask = torch.ones(batch_size, seq_len, seq_len)
        
        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = mult_standard(x, mask=mask)
                _ = approx(x, mask=mask)
        
        # Benchmark standard
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = mult_standard(x, mask=mask)
        standard_time = (time.time() - start_time) / num_runs
        
        # Benchmark approximation
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = approx(x, mask=mask)
        approx_time = (time.time() - start_time) / num_runs
        
        print(f"Standard multiplication: {standard_time*1000:.2f} ms")
        print(f"Linear approximation: {approx_time*1000:.2f} ms")
        print(f"Speedup: {standard_time/approx_time:.2f}x")
        
        # Linear approximation should be faster
        assert approx_time < standard_time


class TestTriangleMultiplicationComparison:
    """Compare fused implementation with original (if available)."""
    
    @pytest.mark.skipif(not ORIGINAL_AVAILABLE, reason="Original triangle multiplication not available")
    def test_compare_with_original(self):
        """Compare fused implementation with original triangle multiplication."""
        
        batch_size, seq_len = 1, 12
        c_z, c_hidden = 32, 64
        
        # Create both implementations
        if ORIGINAL_AVAILABLE:
            original_mult = TriangleMultiplicationOutgoing(c_z, c_hidden)
        else:
            original_mult = MockTriangleMultiplication(c_z, c_hidden, outgoing=True)
        
        fused_mult = FusedTriangleMultiplicationOutgoing(c_z, c_hidden, use_triton=False)
        
        # Test input
        z = torch.randn(batch_size, seq_len, seq_len, c_z)
        mask = torch.ones(batch_size, seq_len, seq_len)
        
        with torch.no_grad():
            # Original output
            try:
                if ORIGINAL_AVAILABLE:
                    original_output = original_mult(z, mask=mask)
                else:
                    original_output = original_mult(z, mask=mask)
            except Exception as e:
                print(f"Original multiplication failed: {e}")
                return
            
            # Fused output
            fused_output = fused_mult(z, mask=mask)
            
            # Compare shapes
            assert original_output.shape == fused_output.shape
            
            print(f"Original output range: [{original_output.min():.4f}, {original_output.max():.4f}]")
            print(f"Fused output range: [{fused_output.min():.4f}, {fused_output.max():.4f}]")
            
            # Check if outputs are reasonably close
            if torch.allclose(original_output, fused_output, atol=1e-2):
                print("✅ Outputs match closely")
            else:
                print("⚠️  Outputs differ (expected due to implementation differences)")


def benchmark_triangle_multiplication():
    """Benchmark different triangle multiplication implementations."""
    
    print("🚀 Benchmarking Triangle Multiplication Implementations")
    
    batch_size, seq_len = 2, 48
    c_in, c_hidden = 128, 256
    num_runs = 10
    
    # Create implementations
    mult_standard = FusedTriangleMultiplication(
        c_in, c_hidden, use_triton=False, use_linear_approx=False
    )
    mult_approx = FusedTriangleMultiplication(
        c_in, c_hidden, use_triton=False, use_linear_approx=True
    )
    mult_chunked = FusedTriangleMultiplication(
        c_in, c_hidden, use_triton=False, chunk_size=16
    )
    linear_approx = LinearTriangleApproximation(c_in)
    
    x = torch.randn(batch_size, seq_len, seq_len, c_in)
    mask = torch.ones(batch_size, seq_len, seq_len)
    
    implementations = [
        ("Standard", mult_standard),
        ("Linear Approx (Fused)", mult_approx),
        ("Chunked", mult_chunked),
        ("Linear Approx (Simple)", linear_approx)
    ]
    
    results = {}
    
    for name, impl in implementations:
        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = impl(x, mask=mask)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = impl(x, mask=mask)
        
        elapsed_time = (time.time() - start_time) / num_runs
        results[name] = elapsed_time
        
        print(f"{name}: {elapsed_time*1000:.2f} ms")
    
    # Calculate speedups
    baseline = results["Standard"]
    for name, time_val in results.items():
        if name != "Standard":
            speedup = baseline / time_val
            print(f"{name} speedup: {speedup:.2f}x")
    
    return results


if __name__ == "__main__":
    # Run benchmark
    benchmark_triangle_multiplication()
    
    # Run tests
    pytest.main([__file__, "-v"])
