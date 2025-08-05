#!/usr/bin/env python3
"""
OdinFold Setup Script
The engine that powers FoldForever
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="odinfold",
    version="1.0.0",
    author="OdinFold Team",
    author_email="team@odinfold.ai",
    description="OdinFold - The engine that powers FoldForever",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/euticus/openfold",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "pytest>=7.4.0",
            "pytest-benchmark>=4.0.0",
        ],
        "gpu": [
            "torch>=2.0.1",
        ],
        "relaxation": [
            "openmm>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "odinfold=odinfold.cli:main",
            "odinfold-server=odinfold.api.server:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
