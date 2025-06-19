"""
Benchmark package for comparing Python and Fortran implementations.

This package provides tools to validate the Python Q-learning implementation
against reference Fortran results.
"""

from .compare_fortran import (
    load_fortran_results,
    compare_implementations,
    validate_implementation,
    BenchmarkResults,
    TOLERANCE_EPSILON
)

__all__ = [
    'load_fortran_results',
    'compare_implementations', 
    'validate_implementation',
    'BenchmarkResults',
    'TOLERANCE_EPSILON'
] 