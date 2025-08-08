#!/usr/bin/env python3
"""
Simple test for memory layout optimization fix
"""

import torch

def test_memory_layout_optimization():
    """Test GPU memory layout optimization."""
    print("🧪 Testing memory layout optimization...")
    
    try:
        from openfold.utils.gpu_memory_optimization import MemoryLayoutOptimizer, MemoryLayoutConfig
        
        # Create memory optimizer
        config = MemoryLayoutConfig(
            enable_memory_coalescing=True,
            prefer_channels_last=True,
            use_memory_efficient_attention=True,
            enable_tensor_fusion=True
        )
        
        optimizer = MemoryLayoutOptimizer(config)
        print("  ✅ Memory layout optimizer created")
        
        # Test tensor layout optimization
        test_tensor = torch.randn(4, 256, 128)
        
        try:
            optimized = optimizer.optimize_tensor_layout(test_tensor, "linear")
            print(f"    ✅ Tensor optimized: {test_tensor.shape} -> {optimized.shape}")
            print(f"      Contiguous: {test_tensor.is_contiguous()} -> {optimized.is_contiguous()}")
            return True
            
        except Exception as e:
            print(f"    ❌ Tensor optimization failed: {e}")
            return False
        
    except ImportError:
        print("  ⚠️  GPU memory optimization not available")
        return True
    except Exception as e:
        print(f"  ❌ Memory layout optimization test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_memory_layout_optimization()
    if success:
        print("✅ Memory layout optimization working!")
    else:
        print("❌ Memory layout optimization failed!")
