#!/usr/bin/env python3
"""
Setup script for OpenFold++ CUDA kernels.
"""

import os
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup, find_packages

# Check if CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Cannot build CUDA kernels.")

# Get CUDA version
cuda_version = torch.version.cuda
print(f"Building CUDA kernels for CUDA {cuda_version}")

# Define source files
cuda_sources = [
    'src/pybind_interface.cpp',  # Updated to use pybind11 interface
    'src/triangle_attention.cu',
    'src/triangle_multiply.cu'
]

# Include directories
include_dirs = [
    'include',
    torch.utils.cpp_extension.include_paths()[0]
]

# CUDA compilation flags
nvcc_flags = [
    '-O3',
    '--use_fast_math',
    '-Xptxas=-v',
    '-gencode=arch=compute_70,code=sm_70',  # V100
    '-gencode=arch=compute_75,code=sm_75',  # RTX 20xx
    '-gencode=arch=compute_80,code=sm_80',  # A100
    '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx
    '-gencode=arch=compute_89,code=sm_89',  # RTX 40xx
    '-gencode=arch=compute_90,code=sm_90',  # H100
]

# C++ compilation flags
cxx_flags = [
    '-O3',
    '-std=c++14',
    '-DWITH_CUDA'
]

# Define the extension
ext_modules = [
    CUDAExtension(
        name='openfold_cuda_kernels',
        sources=cuda_sources,
        include_dirs=include_dirs,
        extra_compile_args={
            'cxx': cxx_flags,
            'nvcc': nvcc_flags
        },
        libraries=['cublas', 'curand'],
        language='c++'
    )
]

# Setup configuration
setup(
    name='openfold-cuda-kernels',
    version='1.0.0',
    description='CUDA kernels for OpenFold++ triangle operations',
    author='OpenFold++ Team',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    },
    zip_safe=False,
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.12.0',
        'numpy>=1.19.0',
        'pybind11>=2.6.0'
    ]
)
