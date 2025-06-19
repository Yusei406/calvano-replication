"""
Fortran benchmark comparison module.

Compares Python Q-learning results with reference Fortran implementation
using identical seeds and parameters. Validates implementation accuracy
with strict tolerance checking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import os
import json
import warnings

# Handle imports for both package and standalone usage
try:
    from ..dtype_policy import DTYPE, array, zeros
    from ..params import SimParams
    from ..q_learning import QLearningSimulation
    from ..analysis.profit_gain import calculate_profits
except ImportError:
    try:
        from dtype_policy import DTYPE, array, zeros
        from params import SimParams
        from q_learning import QLearningSimulation
        from analysis.profit_gain import calculate_profits
    except ImportError:
        # Fallback for testing
        import numpy as np
        DTYPE = np.float64
        array = lambda x: np.array(x, dtype=DTYPE)
        zeros = lambda n: np.zeros(n, dtype=DTYPE)
        
        class SimParams:
            def __init__(self, config=None):
                self.n_agents = 2
                self.n_actions = 11
        
        class QLearningSimulation:
            def __init__(self, params): pass
            def run_simulation(self, seed=None): 
                return {
                    'price_history': np.random.rand(1000, 2),
                    'Q_matrices': [np.random.rand(10, 11), np.random.rand(10, 11)],
                    'converged': True
                }
        
        def calculate_profits(prices, params):
            return np.random.rand(len(prices), 2)


# Strict tolerance for comparison - if exceeded, raises AssertionError
TOLERANCE_EPSILON = 1e-12


@dataclass
class BenchmarkResults:
    """Container for benchmark comparison results."""
    passed: bool
    price_rmse: float
    price_max_error: float
    profit_rmse: float
    profit_max_error: float
    n_timesteps: int
    n_agents: int
    fortran_file: str
    python_seed: int
    errors: List[str]
    warnings: List[str]


def load_fortran_results(fortran_csv_path: str) -> Dict[str, np.ndarray]:
    """
    Load Fortran simulation results from CSV file.
    
    Expected CSV format:
    - timestep,agent_1_price,agent_2_price,...,agent_1_profit,agent_2_profit,...
    - OR separate price and profit files
    
    Args:
        fortran_csv_path: Path to Fortran output CSV file
        
    Returns:
        Dictionary containing 'prices', 'profits', and metadata
    """
    if not os.path.exists(fortran_csv_path):
        raise FileNotFoundError(f"Fortran results file not found: {fortran_csv_path}")
    
    try:
        # Try to load the main results file
        df = pd.read_csv(fortran_csv_path)
        
        # Parse column names to identify structure
        columns = df.columns.tolist()
        
        # Look for common patterns
        price_cols = [col for col in columns if 'price' in col.lower()]
        profit_cols = [col for col in columns if 'profit' in col.lower()]
        
        if not price_cols:
            # Try generic patterns like agent_1, agent_2, etc.
            agent_cols = [col for col in columns if 'agent' in col.lower() and 'profit' not in col.lower()]
            if agent_cols:
                price_cols = agent_cols
        
        if len(price_cols) == 0:
            # Assume first few columns after timestep are prices
            non_time_cols = [col for col in columns if 'time' not in col.lower()]
            n_agents = min(2, len(non_time_cols) // 2)  # Assume equal number of price and profit columns
            price_cols = non_time_cols[:n_agents]
            profit_cols = non_time_cols[n_agents:2*n_agents]
        
        # Extract price data
        if price_cols:
            prices = df[price_cols].values
        else:
            raise ValueError("Could not identify price columns in Fortran CSV")
        
        # Extract profit data (calculate if not present)
        if profit_cols and len(profit_cols) == len(price_cols):
            profits = df[profit_cols].values
        else:
            # Calculate profits from prices using same method as Python
            warnings.warn("Profit columns not found, will calculate from prices")
            profits = None
        
        # Extract timestep if available
        timestep_col = None
        for col in columns:
            if any(keyword in col.lower() for keyword in ['time', 'step', 'period', 't']):
                timestep_col = col
                break
        
        timesteps = df[timestep_col].values if timestep_col else np.arange(len(df))
        
        return {
            'prices': array(prices),
            'profits': array(profits) if profits is not None else None,
            'timesteps': array(timesteps),
            'n_agents': prices.shape[1],
            'n_timesteps': len(prices),
            'columns': columns,
            'raw_data': df
        }
        
    except Exception as e:
        raise ValueError(f"Failed to parse Fortran CSV file {fortran_csv_path}: {e}")


def run_python_simulation(params: SimParams, seed: int) -> Dict[str, Any]:
    """
    Run Python simulation with specified parameters and seed.
    
    Args:
        params: Simulation parameters
        seed: Random seed to use
        
    Returns:
        Dictionary with simulation results
    """
    sim = QLearningSimulation(params)
    results = sim.run_simulation(seed=seed)
    
    # Ensure we have price history
    if 'price_history' not in results or results['price_history'].size == 0:
        raise ValueError("Python simulation failed to produce price history")
    
    return results


def calculate_rmse(array1: np.ndarray, array2: np.ndarray) -> float:
    """Calculate Root Mean Square Error between two arrays."""
    if array1.shape != array2.shape:
        raise ValueError(f"Shape mismatch: {array1.shape} vs {array2.shape}")
    
    mse = np.mean((array1 - array2) ** 2)
    return float(np.sqrt(mse))


def calculate_max_error(array1: np.ndarray, array2: np.ndarray) -> float:
    """Calculate maximum absolute error between two arrays."""
    if array1.shape != array2.shape:
        raise ValueError(f"Shape mismatch: {array1.shape} vs {array2.shape}")
    
    max_err = np.max(np.abs(array1 - array2))
    return float(max_err)


def compare_implementations(
    fortran_csv_path: str,
    python_results: Dict[str, Any],
    params: SimParams,
    tolerance: float = TOLERANCE_EPSILON
) -> BenchmarkResults:
    """
    Compare Python and Fortran implementations with strict tolerance checking.
    
    Args:
        fortran_csv_path: Path to Fortran results CSV
        python_results: Python simulation results
        params: Simulation parameters
        tolerance: Error tolerance (default: 1e-12)
        
    Returns:
        BenchmarkResults with detailed comparison
    """
    errors = []
    warnings_list = []
    
    try:
        # Load Fortran results
        fortran_data = load_fortran_results(fortran_csv_path)
        fortran_prices = fortran_data['prices']
        fortran_profits = fortran_data['profits']
        
        # Extract Python results
        python_prices = python_results.get('price_history', np.array([]))
        if python_prices.size == 0:
            errors.append("Python results contain no price history")
            return BenchmarkResults(
                passed=False,
                price_rmse=np.inf,
                price_max_error=np.inf,
                profit_rmse=np.inf,
                profit_max_error=np.inf,
                n_timesteps=0,
                n_agents=0,
                fortran_file=fortran_csv_path,
                python_seed=python_results.get('seed', -1),
                errors=errors,
                warnings=warnings_list
            )
        
        # Check dimensions compatibility
        if python_prices.shape[1] != fortran_prices.shape[1]:
            errors.append(f"Agent count mismatch: Python {python_prices.shape[1]} vs Fortran {fortran_prices.shape[1]}")
        
        # Align timesteps (use shorter sequence)
        min_timesteps = min(len(python_prices), len(fortran_prices))
        if len(python_prices) != len(fortran_prices):
            warnings_list.append(f"Timestep count mismatch: Python {len(python_prices)} vs Fortran {len(fortran_prices)}, using first {min_timesteps}")
        
        python_prices_aligned = python_prices[:min_timesteps]
        fortran_prices_aligned = fortran_prices[:min_timesteps]
        
        # Compare prices
        price_rmse = calculate_rmse(python_prices_aligned, fortran_prices_aligned)
        price_max_error = calculate_max_error(python_prices_aligned, fortran_prices_aligned)
        
        # Calculate/compare profits
        profit_rmse = np.nan
        profit_max_error = np.nan
        
        if fortran_profits is not None:
            # Use Fortran profits if available
            fortran_profits_aligned = fortran_profits[:min_timesteps]
            
            # Calculate Python profits
            python_profits = calculate_profits(python_prices_aligned, params)
            
            profit_rmse = calculate_rmse(python_profits, fortran_profits_aligned)
            profit_max_error = calculate_max_error(python_profits, fortran_profits_aligned)
        else:
            # Calculate profits for both using same method
            try:
                python_profits = calculate_profits(python_prices_aligned, params)
                fortran_profits_calc = calculate_profits(fortran_prices_aligned, params)
                
                profit_rmse = calculate_rmse(python_profits, fortran_profits_calc)
                profit_max_error = calculate_max_error(python_profits, fortran_profits_calc)
            except Exception as e:
                warnings_list.append(f"Could not calculate profits for comparison: {e}")
        
        # Check against tolerance
        passed = True
        if price_max_error > tolerance:
            errors.append(f"Price max error {price_max_error:.2e} exceeds tolerance {tolerance:.2e}")
            passed = False
        
        if not np.isnan(profit_max_error) and profit_max_error > tolerance:
            errors.append(f"Profit max error {profit_max_error:.2e} exceeds tolerance {tolerance:.2e}")
            passed = False
        
        return BenchmarkResults(
            passed=passed,
            price_rmse=price_rmse,
            price_max_error=price_max_error,
            profit_rmse=profit_rmse if not np.isnan(profit_rmse) else 0.0,
            profit_max_error=profit_max_error if not np.isnan(profit_max_error) else 0.0,
            n_timesteps=min_timesteps,
            n_agents=python_prices.shape[1],
            fortran_file=fortran_csv_path,
            python_seed=python_results.get('seed', -1),
            errors=errors,
            warnings=warnings_list
        )
        
    except Exception as e:
        errors.append(f"Comparison failed: {e}")
        return BenchmarkResults(
            passed=False,
            price_rmse=np.inf,
            price_max_error=np.inf,
            profit_rmse=np.inf,
            profit_max_error=np.inf,
            n_timesteps=0,
            n_agents=0,
            fortran_file=fortran_csv_path,
            python_seed=python_results.get('seed', -1),
            errors=errors,
            warnings=warnings_list
        )


def validate_implementation(
    fortran_csv_path: str,
    params: SimParams,
    seed: int,
    tolerance: float = TOLERANCE_EPSILON
) -> BenchmarkResults:
    """
    Complete validation of Python implementation against Fortran reference.
    
    Args:
        fortran_csv_path: Path to Fortran results CSV
        params: Simulation parameters (must match Fortran run)
        seed: Random seed (must match Fortran run)
        tolerance: Error tolerance (default: 1e-12)
        
    Returns:
        BenchmarkResults with validation outcome
        
    Raises:
        AssertionError: If validation fails with errors exceeding tolerance
    """
    # Run Python simulation with same parameters and seed
    try:
        python_results = run_python_simulation(params, seed)
        python_results['seed'] = seed  # Store seed for reference
    except Exception as e:
        result = BenchmarkResults(
            passed=False,
            price_rmse=np.inf,
            price_max_error=np.inf,
            profit_rmse=np.inf,
            profit_max_error=np.inf,
            n_timesteps=0,
            n_agents=0,
            fortran_file=fortran_csv_path,
            python_seed=seed,
            errors=[f"Python simulation failed: {e}"],
            warnings=[]
        )
        
        # Raise AssertionError for failed validation
        raise AssertionError(f"Validation failed: Python simulation error: {e}")
    
    # Compare implementations
    result = compare_implementations(fortran_csv_path, python_results, params, tolerance)
    
    # Raise AssertionError if validation failed
    if not result.passed:
        error_msg = f"Validation failed with tolerance ε={tolerance:.2e}:\n"
        error_msg += f"  Price RMSE: {result.price_rmse:.2e}\n"
        error_msg += f"  Price Max Error: {result.price_max_error:.2e}\n"
        if not np.isnan(result.profit_rmse):
            error_msg += f"  Profit RMSE: {result.profit_rmse:.2e}\n"
            error_msg += f"  Profit Max Error: {result.profit_max_error:.2e}\n"
        error_msg += f"  Errors: {'; '.join(result.errors)}"
        
        raise AssertionError(error_msg)
    
    return result


def create_benchmark_report(
    results: List[BenchmarkResults],
    output_path: str = "benchmark_report.json"
) -> None:
    """
    Create detailed benchmark report from validation results.
    
    Args:
        results: List of BenchmarkResults
        output_path: Path to save JSON report
    """
    report_data = {
        'summary': {
            'total_tests': len(results),
            'passed': sum(1 for r in results if r.passed),
            'failed': sum(1 for r in results if not r.passed),
            'tolerance_epsilon': TOLERANCE_EPSILON
        },
        'tests': []
    }
    
    for i, result in enumerate(results):
        test_data = {
            'test_id': i + 1,
            'passed': result.passed,
            'fortran_file': result.fortran_file,
            'seed': result.python_seed,
            'metrics': {
                'price_rmse': float(result.price_rmse),
                'price_max_error': float(result.price_max_error),
                'profit_rmse': float(result.profit_rmse) if not np.isnan(result.profit_rmse) else None,
                'profit_max_error': float(result.profit_max_error) if not np.isnan(result.profit_max_error) else None
            },
            'dimensions': {
                'n_timesteps': result.n_timesteps,
                'n_agents': result.n_agents
            },
            'errors': result.errors,
            'warnings': result.warnings
        }
        report_data['tests'].append(test_data)
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"Benchmark report saved to {output_path}")


def run_benchmark_suite(
    fortran_files: List[str],
    params_list: List[SimParams],
    seeds: List[int],
    output_dir: str = "benchmark_results"
) -> List[BenchmarkResults]:
    """
    Run comprehensive benchmark suite against multiple Fortran files.
    
    Args:
        fortran_files: List of Fortran result CSV files
        params_list: List of parameter configurations
        seeds: List of seeds to test
        output_dir: Output directory for results
        
    Returns:
        List of BenchmarkResults
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for i, (fortran_file, params, seed) in enumerate(zip(fortran_files, params_list, seeds)):
        print(f"Running benchmark {i+1}/{len(fortran_files)}: {os.path.basename(fortran_file)}")
        
        try:
            result = validate_implementation(fortran_file, params, seed)
            all_results.append(result)
            print(f"  ✓ PASSED - Max error: {result.price_max_error:.2e}")
            
        except AssertionError as e:
            # Create failed result for reporting
            result = BenchmarkResults(
                passed=False,
                price_rmse=np.inf,
                price_max_error=np.inf,
                profit_rmse=np.inf,
                profit_max_error=np.inf,
                n_timesteps=0,
                n_agents=0,
                fortran_file=fortran_file,
                python_seed=seed,
                errors=[str(e)],
                warnings=[]
            )
            all_results.append(result)
            print(f"  ✗ FAILED - {str(e)}")
        
        except Exception as e:
            print(f"  ✗ ERROR - {str(e)}")
            continue
    
    # Generate report
    report_path = os.path.join(output_dir, "benchmark_report.json")
    create_benchmark_report(all_results, report_path)
    
    # Print summary
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    print(f"\nBenchmark Summary: {passed}/{total} tests passed")
    
    if passed < total:
        print("⚠️  Some benchmarks failed - check report for details")
    else:
        print("✓ All benchmarks passed!")
    
    return all_results 