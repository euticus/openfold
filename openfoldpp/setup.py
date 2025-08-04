#!/usr/bin/env python3
"""
OpenFold++ Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path) as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path) as f:
        long_description = f.read()

setup(
    name="openfoldpp",
    version="1.0.0",
    description="High-Performance Protein Folding Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="OpenFold++ Team",
    author_email="",
    url="https://github.com/euticus/openfold",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "flake8"],
        "cuda": ["flash-attn", "bitsandbytes"],
        "md": ["openmm", "mdtraj"],
        "lm": ["transformers", "tokenizers"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    entry_points={
        "console_scripts": [
            "openfoldpp=openfoldpp.cli:main",
        ],
    },
)
