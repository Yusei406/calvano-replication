#!/usr/bin/env python3
"""Setup script for Calvano et al. (2020) Replication package."""

from setuptools import setup, find_packages

def read_requirements():
    """Read requirements.txt for dependencies."""
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="calvano-replication",
    version="1.0.0",
    author="Yusei406",
    description="Python replication of Calvano et al. (2020) Q-learning algorithmic pricing",
    url="https://github.com/Yusei406/calvano-replication",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "calvano-replicate=src.main:main",
        ],
    },
)
