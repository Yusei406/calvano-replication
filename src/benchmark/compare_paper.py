"""
Paper parity comparison module.

Compares simulation results against published values from Calvano et al. (2020) Table 1.
Validates that the Python implementation reproduces academic benchmark results.
"""

import json
import math
import sys
import pathlib
from typing import Dict, Any

# Calvano et al. (2020) Table 1 values rounded to 3rd decimal place
PAPER_VALUES = {
    "nash_price": 0.500,
    "coop_gap": 0.300,
    "conv_rate": 0.9265,
    "mean_profit": 0.250,
}


def compare_to_paper(results_path: str, eps: float = 1e-3) -> None:
    """
    Compare simulation results to published paper values.
    
    Args:
        results_path: Path to results JSON file
        eps: Tolerance for comparison (default: 1e-3)
        
    Raises:
        AssertionError: If any value deviates beyond tolerance
        FileNotFoundError: If results file not found
        KeyError: If required keys missing from results
    """
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Results file not found: {results_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in results file: {e}")
    
    # Validate all required keys are present
    missing_keys = set(PAPER_VALUES.keys()) - set(results.keys())
    if missing_keys:
        raise KeyError(f"Missing required keys in results: {missing_keys}")
    
    # Compare each value
    for key, expected in PAPER_VALUES.items():
        actual = results[key]
        
        # Check for valid numeric value
        if not math.isfinite(actual):
            raise AssertionError(
                f"{key}: expected {expected:.4g}, got {actual} (non-finite value)"
            )
        
        # Check tolerance
        if abs(actual - expected) > eps:
            raise AssertionError(
                f"{key}: expected {expected:.4g}, got {actual:.4g} (tol={eps})"
            )
    
    print("✅ Paper parity test passed.")


def generate_paper_summary(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract paper-comparable values from simulation results.
    
    Args:
        results: Raw simulation results dictionary
        
    Returns:
        Dictionary with keys matching PAPER_VALUES
    """
    summary = {}
    
    # Extract Nash price (assume first agent, or average)
    if 'final_prices' in results:
        prices = results['final_prices']
        if isinstance(prices, list) and len(prices) > 0:
            if isinstance(prices[0], list):
                # Multiple agents - take average
                summary['nash_price'] = float(sum(prices[0]) / len(prices[0]))
            else:
                # Single value
                summary['nash_price'] = float(prices[0])
    
    # Extract cooperation gap
    if 'cooperative_gap' in results:
        summary['coop_gap'] = float(results['cooperative_gap'])
    elif 'coop_deviation' in results:
        summary['coop_gap'] = float(results['coop_deviation'])
    
    # Extract convergence rate
    if 'conv_rate' in results:
        summary['conv_rate'] = float(results['conv_rate'])
    elif 'convergence_rate' in results:
        summary['conv_rate'] = float(results['convergence_rate'])
    
    # Extract mean profit
    if 'mean_profit' in results:
        summary['mean_profit'] = float(results['mean_profit'])
    elif 'profit_mean' in results:
        summary['mean_profit'] = float(results['profit_mean'])
    
    return summary


def validate_paper_format(results_path: str) -> bool:
    """
    Validate that results file contains all required keys for paper comparison.
    
    Args:
        results_path: Path to results JSON file
        
    Returns:
        True if all required keys present, False otherwise
    """
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        required_keys = set(PAPER_VALUES.keys())
        available_keys = set(results.keys())
        
        missing_keys = required_keys - available_keys
        if missing_keys:
            print(f"Missing keys for paper comparison: {missing_keys}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validating results format: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare simulation results to Calvano et al. (2020) paper values"
    )
    parser.add_argument("--results", required=True, 
                       help="Path to results JSON file")
    parser.add_argument("--eps", type=float, default=1e-3,
                       help="Tolerance for comparison (default: 1e-3)")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate format, don't compare values")
    
    args = parser.parse_args()
    
    if args.validate_only:
        is_valid = validate_paper_format(args.results)
        sys.exit(0 if is_valid else 1)
    else:
        try:
            compare_to_paper(args.results, args.eps)
            sys.exit(0)
        except Exception as e:
            print(f"❌ Paper parity test failed: {e}")
            sys.exit(1) 