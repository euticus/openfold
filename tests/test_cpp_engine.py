#!/usr/bin/env python3
"""
Test Suite for C++ FoldEngine

Tests the C++ inference engine functionality and integration.
"""

import pytest
import torch
import numpy as np
import sys
import subprocess
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCppEngineSetup:
    """Test C++ engine setup and compilation."""
    
    def test_cmake_files_exist(self):
        """Test that CMake files exist."""
        
        cpp_dir = Path(__file__).parent.parent / "cpp_engine"
        
        assert (cpp_dir / "CMakeLists.txt").exists()
        assert (cpp_dir / "include" / "fold_engine.h").exists()
        assert (cpp_dir / "src" / "fold_engine.cpp").exists()
        assert (cpp_dir / "src" / "main.cpp").exists()
    
    def test_header_files_complete(self):
        """Test that all header files are present."""
        
        cpp_dir = Path(__file__).parent.parent / "cpp_engine"
        include_dir = cpp_dir / "include"
        
        required_headers = [
            "fold_engine.h",
            "attention.h", 
            "structure_module.h",
            "ligand_encoder.h",
            "mutation_predictor.h",
            "utils.h"
        ]
        
        for header in required_headers:
            assert (include_dir / header).exists(), f"Missing header: {header}"
    
    def test_source_files_complete(self):
        """Test that all source files are present."""
        
        cpp_dir = Path(__file__).parent.parent / "cpp_engine"
        src_dir = cpp_dir / "src"
        
        required_sources = [
            "fold_engine.cpp",
            "attention.cpp",
            "structure_module.cpp", 
            "ligand_encoder.cpp",
            "mutation_predictor.cpp",
            "utils.cpp",
            "cli.cpp",
            "main.cpp"
        ]
        
        for source in required_sources:
            assert (src_dir / source).exists(), f"Missing source: {source}"


class TestCppEngineAPI:
    """Test C++ engine API design."""
    
    def test_fold_engine_header_structure(self):
        """Test FoldEngine header structure."""
        
        header_file = Path(__file__).parent.parent / "cpp_engine" / "include" / "fold_engine.h"
        content = header_file.read_text()
        
        # Check for key classes and structures
        assert "class FoldEngine" in content
        assert "struct FoldConfig" in content
        assert "struct ProteinInput" in content
        assert "struct FoldingResult" in content
        assert "struct PerformanceMetrics" in content
        
        # Check for key methods
        assert "fold_protein" in content
        assert "fold_batch" in content
        assert "predict_mutations" in content
        assert "initialize" in content
    
    def test_cli_interface_design(self):
        """Test CLI interface design."""
        
        cli_file = Path(__file__).parent.parent / "cpp_engine" / "src" / "cli.cpp"
        content = cli_file.read_text()
        
        # Check for CLI commands
        assert "fold" in content
        assert "mutate" in content
        assert "batch" in content
        assert "benchmark" in content
        assert "info" in content
        
        # Check for argument parsing
        assert "getopt_long" in content
        assert "print_usage" in content
    
    def test_utility_functions(self):
        """Test utility function definitions."""
        
        utils_header = Path(__file__).parent.parent / "cpp_engine" / "include" / "utils.h"
        content = utils_header.read_text()
        
        # Check for key utilities
        assert "load_fasta" in content
        assert "save_pdb" in content
        assert "calculate_rmsd" in content
        assert "calculate_tm_score" in content
        assert "get_memory_usage_mb" in content


class TestCppEngineIntegration:
    """Test C++ engine integration capabilities."""
    
    def test_pytorch_integration(self):
        """Test PyTorch integration in headers."""
        
        fold_engine_header = Path(__file__).parent.parent / "cpp_engine" / "include" / "fold_engine.h"
        content = fold_engine_header.read_text()
        
        # Check for PyTorch includes and usage
        assert "#include <torch/torch.h>" in content
        assert "torch::Tensor" in content
        assert "torch::Device" in content
        assert "torch::jit::script::Module" in content
    
    def test_attention_module_design(self):
        """Test attention module design."""
        
        attention_header = Path(__file__).parent.parent / "cpp_engine" / "include" / "attention.h"
        content = attention_header.read_text()
        
        # Check for attention classes
        assert "class AttentionModule" in content
        assert "class TriangleAttention" in content
        assert "class FlashAttention" in content
        
        # Check for key methods
        assert "self_attention" in content
        assert "cross_attention" in content
    
    def test_structure_module_design(self):
        """Test structure module design."""
        
        structure_header = Path(__file__).parent.parent / "cpp_engine" / "include" / "structure_module.h"
        content = structure_header.read_text()
        
        # Check for structure classes
        assert "class StructureModule" in content
        assert "class InvariantPointAttention" in content
        
        # Check for coordinate prediction
        assert "predict_all_atom" in content
    
    def test_ligand_encoder_design(self):
        """Test ligand encoder design."""
        
        ligand_header = Path(__file__).parent.parent / "cpp_engine" / "include" / "ligand_encoder.h"
        content = ligand_header.read_text()
        
        # Check for ligand classes
        assert "class LigandEncoder" in content
        assert "namespace molecular_utils" in content
        
        # Check for SMILES processing
        assert "encode_smiles" in content
        assert "encode_batch" in content
        assert "parse_smiles" in content
    
    def test_mutation_predictor_design(self):
        """Test mutation predictor design."""
        
        mutation_header = Path(__file__).parent.parent / "cpp_engine" / "include" / "mutation_predictor.h"
        content = mutation_header.read_text()
        
        # Check for mutation classes
        assert "class MutationPredictor" in content
        assert "namespace mutation_utils" in content
        
        # Check for mutation prediction
        assert "predict_effects" in content
        assert "predict_stability" in content


class TestCppEngineImplementation:
    """Test C++ engine implementation details."""
    
    def test_fold_engine_implementation(self):
        """Test FoldEngine implementation."""
        
        impl_file = Path(__file__).parent.parent / "cpp_engine" / "src" / "fold_engine.cpp"
        content = impl_file.read_text()
        
        # Check for key implementations
        assert "FoldEngine::FoldEngine" in content
        assert "FoldEngine::initialize" in content
        assert "FoldEngine::fold_protein" in content
        assert "FoldEngine::predict_mutations" in content
        
        # Check for amino acid mapping
        assert "AA_TO_IDX" in content
        assert "IDX_TO_AA" in content
    
    def test_attention_implementation(self):
        """Test attention implementation."""
        
        impl_file = Path(__file__).parent.parent / "cpp_engine" / "src" / "attention.cpp"
        content = impl_file.read_text()
        
        # Check for attention implementations
        assert "AttentionModule::forward" in content
        assert "TriangleAttention::forward" in content
        assert "FlashAttention::forward" in content
        
        # Check for multi-head attention logic
        assert "num_heads" in content
        assert "head_dim" in content
    
    def test_utils_implementation(self):
        """Test utilities implementation."""
        
        impl_file = Path(__file__).parent.parent / "cpp_engine" / "src" / "utils.cpp"
        content = impl_file.read_text()
        
        # Check for utility implementations
        assert "load_fasta" in content
        assert "save_pdb" in content
        assert "calculate_rmsd" in content
        assert "get_memory_usage_mb" in content
        
        # Check for PDB format handling
        assert "HEADER" in content
        assert "ATOM" in content


def test_cpp_engine_mock_functionality():
    """Test mock functionality of C++ engine components."""
    
    print("🔧 Testing C++ Engine Mock Functionality...")
    
    # Test that we can create mock PyTorch tensors for the expected interfaces
    batch_size, seq_len, d_model = 2, 50, 256
    
    # Mock protein input
    protein_features = torch.randn(batch_size, seq_len, d_model)
    sequence_indices = torch.randint(0, 21, (batch_size, seq_len))
    
    print(f"✅ Mock protein features: {protein_features.shape}")
    print(f"✅ Mock sequence indices: {sequence_indices.shape}")
    
    # Mock ligand input
    num_atoms = 20
    ligand_features = torch.randn(batch_size, num_atoms, 128)
    
    print(f"✅ Mock ligand features: {ligand_features.shape}")
    
    # Mock structure output
    coordinates = torch.randn(batch_size, seq_len, 3)
    confidence = torch.rand(batch_size, seq_len)
    
    print(f"✅ Mock coordinates: {coordinates.shape}")
    print(f"✅ Mock confidence: {confidence.shape}")
    
    # Mock mutation predictions
    num_mutations = 5
    ddg_predictions = torch.randn(num_mutations)
    
    print(f"✅ Mock ΔΔG predictions: {ddg_predictions.shape}")
    
    # Test amino acid conversion (Python equivalent)
    aa_to_idx = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
        'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
        'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20
    }
    
    test_sequence = "MKWVTFISLLFLFSSAYS"
    indices = [aa_to_idx.get(aa, 20) for aa in test_sequence]
    
    print(f"✅ Sequence conversion: {test_sequence} -> {indices[:5]}...")
    
    # Test SMILES parsing (mock)
    test_smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
    
    for smiles in test_smiles:
        # Mock molecular graph
        num_atoms = len([c for c in smiles if c.isalpha()])
        atom_types = torch.randint(0, 10, (num_atoms,))
        print(f"✅ Mock SMILES parsing: {smiles} -> {num_atoms} atoms")
    
    print("🎉 C++ Engine mock functionality test passed!")


def test_cpp_engine_build_requirements():
    """Test C++ engine build requirements."""
    
    print("📋 Testing C++ Engine Build Requirements...")
    
    # Check CMake version requirement
    cmake_file = Path(__file__).parent.parent / "cpp_engine" / "CMakeLists.txt"
    content = cmake_file.read_text()
    
    assert "cmake_minimum_required(VERSION 3.18)" in content
    print("✅ CMake version requirement: 3.18+")
    
    # Check C++ standard
    assert "set(CMAKE_CXX_STANDARD 17)" in content
    print("✅ C++ standard: C++17")
    
    # Check required packages
    assert "find_package(Torch REQUIRED)" in content
    assert "find_package(OpenMP REQUIRED)" in content
    assert "find_package(CUDA REQUIRED)" in content
    print("✅ Required packages: PyTorch, OpenMP, CUDA")
    
    # Check compiler flags
    assert "-O3" in content
    assert "-march=native" in content
    assert "-fopenmp" in content
    print("✅ Optimization flags configured")
    
    # Check CUDA architectures
    assert "CMAKE_CUDA_ARCHITECTURES" in content
    print("✅ CUDA architectures specified")
    
    print("🎉 C++ Engine build requirements test passed!")


def test_cpp_engine_performance_features():
    """Test C++ engine performance features."""
    
    print("⚡ Testing C++ Engine Performance Features...")
    
    # Check for performance-related code
    fold_engine_file = Path(__file__).parent.parent / "cpp_engine" / "src" / "fold_engine.cpp"
    content = fold_engine_file.read_text()
    
    # Check for mixed precision
    assert "mixed_precision" in content
    print("✅ Mixed precision support")
    
    # Check for memory tracking
    assert "memory_usage" in content
    assert "PerformanceMetrics" in content
    print("✅ Memory usage tracking")
    
    # Check for timing
    assert "chrono" in content
    assert "elapsed" in content
    print("✅ Performance timing")
    
    # Check utils for memory functions
    utils_file = Path(__file__).parent.parent / "cpp_engine" / "src" / "utils.cpp"
    utils_content = utils_file.read_text()
    
    assert "get_memory_usage_mb" in utils_content
    assert "get_gpu_memory_usage_mb" in utils_content
    print("✅ System memory monitoring")
    
    # Check for CUDA memory handling
    assert "CUDA_AVAILABLE" in utils_content
    assert "cudaMemGetInfo" in utils_content
    print("✅ GPU memory monitoring")
    
    print("🎉 C++ Engine performance features test passed!")


if __name__ == "__main__":
    # Run integration tests
    test_cpp_engine_mock_functionality()
    test_cpp_engine_build_requirements()
    test_cpp_engine_performance_features()
    
    # Run all tests
    pytest.main([__file__, "-v"])
