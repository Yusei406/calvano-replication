"""
Data type policy for consistent numerical precision across the codebase.
Centralizes dtype management to ensure compatibility with Fortran double precision.
"""

import numpy as np

# Primary data type for all floating point computations
# Matches Fortran REAL(KIND=8) / double precision
DTYPE = np.float64

# Integer types
INT_DTYPE = np.int64
INT32_DTYPE = np.int32

# Epsilon for numerical comparisons
EPS = np.finfo(DTYPE).eps
SMALL_NUMBER = 1e-14

def ensure_dtype(array: np.ndarray, dtype=None) -> np.ndarray:
    """
    Ensure array has the correct dtype.
    
    Args:
        array: Input array
        dtype: Target dtype (defaults to DTYPE for floats, INT_DTYPE for ints)
        
    Returns:
        Array with correct dtype
    """
    if dtype is None:
        if np.issubdtype(array.dtype, np.floating):
            dtype = DTYPE
        elif np.issubdtype(array.dtype, np.integer):
            dtype = INT_DTYPE
        else:
            # Keep original dtype for other types
            return array
    
    return array.astype(dtype)


def zeros(shape, dtype=None) -> np.ndarray:
    """Create zeros array with standard dtype."""
    if dtype is None:
        dtype = DTYPE
    return np.zeros(shape, dtype=dtype)


def ones(shape, dtype=None) -> np.ndarray:
    """Create ones array with standard dtype."""
    if dtype is None:
        dtype = DTYPE
    return np.ones(shape, dtype=dtype)


def full(shape, fill_value, dtype=None) -> np.ndarray:
    """Create filled array with standard dtype."""
    if dtype is None:
        dtype = DTYPE
    return np.full(shape, fill_value, dtype=dtype)


def empty(shape, dtype=None) -> np.ndarray:
    """Create empty array with standard dtype."""
    if dtype is None:
        dtype = DTYPE
    return np.empty(shape, dtype=dtype)


def array(object, dtype=None) -> np.ndarray:
    """Create array with standard dtype."""
    if dtype is None:
        dtype = DTYPE
    return np.array(object, dtype=dtype)


def is_close(a, b, rtol=1e-12, atol=1e-14) -> bool:
    """
    Check if two values are close with appropriate tolerance for DTYPE.
    
    Args:
        a, b: Values to compare
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if values are close
    """
    return np.allclose(a, b, rtol=rtol, atol=atol)


def are_equal_reals(a: float, b: float, tol: float = 1e-12) -> bool:
    """
    Check equality of two real numbers with tolerance.
    Matches the original generic_routines.are_equal_reals function.
    
    Args:
        a, b: Numbers to compare
        tol: Tolerance for comparison
        
    Returns:
        True if numbers are equal within tolerance
    """
    return abs(a - b) <= tol


def validate_array_dtype(array: np.ndarray, expected_dtype=None, name: str = "array") -> None:
    """
    Validate that array has expected dtype.
    
    Args:
        array: Array to validate
        expected_dtype: Expected dtype (defaults to DTYPE)
        name: Name for error messages
        
    Raises:
        ValueError: If dtype doesn't match
    """
    if expected_dtype is None:
        expected_dtype = DTYPE
    
    if array.dtype != expected_dtype:
        raise ValueError(f"{name} has dtype {array.dtype}, expected {expected_dtype}")


def safe_division(numerator, denominator, default=0.0):
    """
    Safe division that handles division by zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value to return if denominator is zero
        
    Returns:
        Result of division or default value
    """
    if abs(denominator) < SMALL_NUMBER:
        return default
    return numerator / denominator


# Convenience constants
ZERO = DTYPE(0.0)
ONE = DTYPE(1.0)
HALF = DTYPE(0.5)
INF = DTYPE(np.inf)
NINF = DTYPE(-np.inf)
NAN = DTYPE(np.nan) 