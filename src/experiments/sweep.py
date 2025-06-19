"""
Parameter Sweep and Batch Experiments for Calvano Q-learning.

Implements parallel parameter grid search:
- Reads JSON/CSV grid configurations
- Uses joblib for parallel execution
- Calls 'calvano run --mode simulate' for each parameter set
- Organizes results in runs/grid_<timestamp>/ structure
- Supports multi-dimensional parameter sweeps
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import time
from datetime import datetime
import itertools

# Parallel processing
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    warnings.warn("joblib not available, running in sequential mode")


def load_parameter_grid(grid_path: str) -> List[Dict[str, Any]]:
    """
    Load parameter grid from JSON or CSV file.
    
    Args:
        grid_path: Path to grid configuration file
        
    Returns:
        List of parameter dictionaries
    """
    grid_path = Path(grid_path)
    
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")
    
    if grid_path.suffix.lower() == '.json':
        return load_json_grid(grid_path)
    elif grid_path.suffix.lower() == '.csv':
        return load_csv_grid(grid_path)
    else:
        raise ValueError(f"Unsupported grid file format: {grid_path.suffix}")


def load_json_grid(grid_path: Path) -> List[Dict[str, Any]]:
    """
    Load parameter grid from JSON file.
    
    JSON format:
    {
        "learning_rate": [0.1, 0.15, 0.2],
        "discount_factor": [0.9, 0.95],
        "exploration_rate": [0.01, 0.05]
    }
    
    Args:
        grid_path: Path to JSON grid file
        
    Returns:
        List of parameter combinations
    """
    with open(grid_path, 'r') as f:
        grid_config = json.load(f)
    
    # Generate all combinations
    param_names = list(grid_config.keys())
    param_values = list(grid_config.values())
    
    combinations = []
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        combinations.append(param_dict)
    
    return combinations


def load_csv_grid(grid_path: Path) -> List[Dict[str, Any]]:
    """
    Load parameter grid from CSV file.
    
    CSV format: Each row represents one parameter combination.
    learning_rate,discount_factor,exploration_rate
    0.1,0.9,0.01
    0.15,0.95,0.05
    
    Args:
        grid_path: Path to CSV grid file
        
    Returns:
        List of parameter combinations
    """
    df = pd.read_csv(grid_path)
    return df.to_dict('records')


def create_base_config(base_config_path: str, param_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create configuration by overriding base config with parameters.
    
    Args:
        base_config_path: Path to base configuration JSON
        param_overrides: Parameters to override
        
    Returns:
        Combined configuration dictionary
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    # Override parameters
    config = base_config.copy()
    config.update(param_overrides)
    
    return config


def run_single_simulation(config: Dict[str, Any], output_dir: str, run_id: str) -> Dict[str, Any]:
    """
    Run a single simulation with given configuration.
    
    Args:
        config: Configuration dictionary
        output_dir: Base output directory
        run_id: Unique identifier for this run
        
    Returns:
        Result dictionary with metadata and paths
    """
    # Create run-specific directory
    run_dir = Path(output_dir) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Prepare calvano command
    calvano_script = Path(__file__).parent.parent.parent / "bin" / "calvano.py"
    
    cmd = [
        sys.executable, str(calvano_script),
        "run", "--config", str(config_path),
        "--mode", "simulate",
        "--output", str(run_dir)
    ]
    
    # Run simulation
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        
        # Check if successful
        if result.returncode == 0:
            status = "success"
            error_msg = None
        else:
            status = "failed"
            error_msg = result.stderr
            
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        status = "timeout"
        error_msg = "Simulation exceeded timeout"
    except Exception as e:
        elapsed_time = time.time() - start_time
        status = "error"
        error_msg = str(e)
    
    # Collect results
    result_dict = {
        'run_id': run_id,
        'config': config,
        'status': status,
        'elapsed_time': elapsed_time,
        'output_dir': str(run_dir),
        'error_msg': error_msg
    }
    
    # Try to load simulation results if successful
    if status == "success":
        summary_path = run_dir / "logs" / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    simulation_results = json.load(f)
                result_dict['simulation_results'] = simulation_results
            except Exception as e:
                warnings.warn(f"Could not load simulation results for {run_id}: {e}")
    
    return result_dict


def run_parameter_sweep(grid_path: str, base_config_path: str, output_dir: str, 
                       n_jobs: int = 1, timeout: int = 300) -> Dict[str, Any]:
    """
    Run parameter sweep with parallel execution.
    
    Args:
        grid_path: Path to parameter grid file
        base_config_path: Path to base configuration
        output_dir: Output directory for all runs
        n_jobs: Number of parallel jobs
        timeout: Timeout per simulation in seconds
        
    Returns:
        Dictionary with sweep results and metadata
    """
    print(f"Loading parameter grid from: {grid_path}")
    param_combinations = load_parameter_grid(grid_path)
    
    print(f"Found {len(param_combinations)} parameter combinations")
    print(f"Running with {n_jobs} parallel jobs")
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(output_dir) / f"grid_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    # Save grid configuration for reference
    grid_metadata = {
        'timestamp': timestamp,
        'grid_path': str(grid_path),
        'base_config_path': str(base_config_path),
        'n_combinations': len(param_combinations),
        'n_jobs': n_jobs,
        'timeout': timeout
    }
    
    with open(sweep_dir / "grid_metadata.json", 'w') as f:
        json.dump(grid_metadata, f, indent=2)
    
    # Prepare simulation configurations
    simulation_configs = []
    for i, params in enumerate(param_combinations):
        config = create_base_config(base_config_path, params)
        run_id = f"{i:04d}"
        simulation_configs.append((config, str(sweep_dir), run_id))
    
    # Run simulations
    start_time = time.time()
    
    if JOBLIB_AVAILABLE and n_jobs != 1:
        print("Running simulations in parallel...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_single_simulation)(config, output_dir, run_id)
            for config, output_dir, run_id in simulation_configs
        )
    else:
        print("Running simulations sequentially...")
        results = []
        for config, output_dir, run_id in simulation_configs:
            result = run_single_simulation(config, output_dir, run_id)
            results.append(result)
            print(f"  Completed {run_id}: {result['status']}")
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful_runs = [r for r in results if r['status'] == 'success']
    failed_runs = [r for r in results if r['status'] != 'success']
    
    sweep_summary = {
        'metadata': grid_metadata,
        'total_runs': len(results),
        'successful_runs': len(successful_runs),
        'failed_runs': len(failed_runs),
        'total_time': total_time,
        'avg_time_per_run': total_time / len(results) if results else 0,
        'output_directory': str(sweep_dir),
        'results': results
    }
    
    # Save sweep summary
    with open(sweep_dir / "sweep_summary.json", 'w') as f:
        json.dump(sweep_summary, f, indent=2)
    
    print(f"\n✅ Parameter sweep completed:")
    print(f"  Total runs: {len(results)}")
    print(f"  Successful: {len(successful_runs)}")
    print(f"  Failed: {len(failed_runs)}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Results saved to: {sweep_dir}")
    
    return sweep_summary


def create_example_grid_files(output_dir: str) -> None:
    """
    Create example grid configuration files.
    
    Args:
        output_dir: Directory to save example files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # JSON grid example
    json_grid = {
        "learning_rate": [0.1, 0.15, 0.2],
        "discount_factor": [0.9, 0.95],
        "exploration_rate": [0.01, 0.05],
        "n_runs": [10]  # Keep small for testing
    }
    
    with open(output_path / "example_grid.json", 'w') as f:
        json.dump(json_grid, f, indent=2)
    
    # CSV grid example
    csv_data = [
        {"learning_rate": 0.1, "discount_factor": 0.9, "exploration_rate": 0.01, "n_runs": 5},
        {"learning_rate": 0.15, "discount_factor": 0.95, "exploration_rate": 0.05, "n_runs": 5},
        {"learning_rate": 0.2, "discount_factor": 0.9, "exploration_rate": 0.01, "n_runs": 5},
    ]
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path / "example_grid.csv", index=False)
    
    # Mini grid for CI testing
    mini_grid = {
        "learning_rate": [0.15],
        "discount_factor": [0.95],
        "n_runs": [2]
    }
    
    with open(output_path / "mini_grid.json", 'w') as f:
        json.dump(mini_grid, f, indent=2)
    
    print(f"Example grid files created in: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run parameter sweep experiments")
    parser.add_argument("--grid", required=True, help="Path to parameter grid file")
    parser.add_argument("--config", required=True, help="Path to base configuration file")
    parser.add_argument("--output", default="runs", help="Output directory")
    parser.add_argument("--njobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per simulation (seconds)")
    parser.add_argument("--create-examples", action="store_true", 
                       help="Create example grid files and exit")
    
    args = parser.parse_args()
    
    if args.create_examples:
        create_example_grid_files("grids")
        exit(0)
    
    # Run parameter sweep
    result = run_parameter_sweep(
        grid_path=args.grid,
        base_config_path=args.config,
        output_dir=args.output,
        n_jobs=args.njobs,
        timeout=args.timeout
    )
    
    # Exit with appropriate code
    if result['failed_runs'] == 0:
        print("✅ All simulations completed successfully")
        exit(0)
    else:
        print(f"⚠ {result['failed_runs']} simulations failed")
        exit(1) 