"""
Main execution module for Calvano Q-learning simulation.

Supports multiple execution modes:
- simulate: Run simulation only and save logs
- analyse: Load existing logs and perform Phase 2 analysis only  
- full: Run simulation + analysis and generate figures/tables
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Core simulation imports
from params import SimParams
from q_learning import run_simulation

# Simplified Phase 2 analysis imports (only essential ones)
try:
    from analysis.convergence_results import aggregate_runs, to_dataframe
except ImportError:
    print("Warning: Analysis modules not fully available, running in simulation-only mode")
    aggregate_runs = None
    to_dataframe = None

# Try to import plotting modules
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: Matplotlib not available, figures will be skipped")
    PLOTTING_AVAILABLE = False


def setup_output_directory(base_dir: str = "runs") -> str:
    """
    Create timestamped output directory.
    
    Args:
        base_dir: Base directory name
        
    Returns:
        Path to created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    
    return str(output_dir)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config JSON file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_run_logs(run_results: List[Dict], output_dir: str) -> None:
    """
    Save simulation run logs to files.
    
    Args:
        run_results: List of run result dictionaries
        output_dir: Output directory path
    """
    logs_dir = Path(output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    
    # Save individual run logs
    for i, result in enumerate(run_results):
        log_file = logs_dir / f"run_{i:04d}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                serializable_result[key] = value.tolist()
            elif isinstance(value, (np.bool_, np.integer, np.floating)):
                serializable_result[key] = value.item()  # Convert numpy scalars to Python types
            else:
                serializable_result[key] = value
        
        with open(log_file, 'w') as f:
            json.dump(serializable_result, f, indent=2)
    
    # Save aggregated summary
    summary_file = logs_dir / "summary.json"
    summary = {
        'n_runs': len(run_results),
        'config_used': 'config.json',
        'timestamp': datetime.now().isoformat()
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved {len(run_results)} run logs to {logs_dir}")


def load_run_logs(log_dir: str) -> List[Dict]:
    """
    Load simulation run logs from directory.
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        List of run result dictionaries
    """
    logs_path = Path(log_dir)
    if not logs_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    
    # Look for log files
    run_files = sorted(logs_path.glob("run_*.json"))
    if not run_files:
        # Try loading from logs subdirectory
        logs_subdir = logs_path / "logs"
        if logs_subdir.exists():
            run_files = sorted(logs_subdir.glob("run_*.json"))
    
    if not run_files:
        raise FileNotFoundError(f"No run log files found in {log_dir}")
    
    run_results = []
    for log_file in run_files:
        with open(log_file, 'r') as f:
            result = json.load(f)
            
            # Convert lists back to numpy arrays
            for key, value in result.items():
                if isinstance(value, list) and key in ['final_prices', 'final_profits', 'price_history', 'profit_history']:
                    result[key] = np.array(value)
            
            run_results.append(result)
    
    print(f"✓ Loaded {len(run_results)} run logs from {log_dir}")
    return run_results


def run_simulation_mode(config: Dict, output_dir: str, n_runs: int = None) -> List[Dict]:
    """
    Run simulation mode: execute Q-learning simulations and save logs.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory path
        n_runs: Number of runs (overrides config if provided)
        
    Returns:
        List of run result dictionaries
    """
    print("=== SIMULATION MODE ===")
    
    # Create simulation parameters
    params = SimParams(config)
    
    # Override number of runs if specified
    if n_runs is not None:
        print(f"Overriding n_runs: {config.get('n_runs', 'default')} → {n_runs}")
        config['n_runs'] = n_runs
    
    n_runs = config.get('n_runs', 10)
    print(f"Running {n_runs} simulations...")
    
    # Run simulations
    run_results = []
    for i in range(n_runs):
        print(f"Run {i+1}/{n_runs}", end="\r")
        
        try:
            result = run_simulation(params)
            run_results.append(result)
        except Exception as e:
            print(f"\n❌ Run {i+1} failed: {e}")
            continue
    
    print(f"\n✓ Completed {len(run_results)}/{n_runs} simulations")
    
    # Save logs
    save_run_logs(run_results, output_dir)
    
    return run_results


def run_analysis_mode(run_results: List[Dict], params: SimParams, output_dir: str) -> Dict:
    """
    Run analysis mode: perform simplified Phase 2 analysis on simulation results.
    
    Args:
        run_results: List of run result dictionaries
        params: Simulation parameters
        output_dir: Output directory path
        
    Returns:
        Dictionary containing all analysis results
    """
    print("=== ANALYSIS MODE ===")
    
    if not run_results:
        print("❌ No simulation results to analyze")
        return {}
    
    print(f"Analyzing {len(run_results)} simulation runs...")
    
    # Basic convergence analysis
    print("→ Basic convergence analysis...")
    
    if aggregate_runs is not None:
        try:
            convergence_stats = aggregate_runs(run_results)
        except Exception as e:
            print(f"   Warning: Convergence analysis failed: {e}")
            convergence_stats = None
    else:
        # Simple fallback analysis
        convergence_flags = [r.get('overall_converged', False) for r in run_results]
        conv_rate = sum(convergence_flags) / len(convergence_flags) if convergence_flags else 0.0
        convergence_stats = {'conv_rate': conv_rate, 'n_runs': len(run_results)}
    
    # Basic statistics
    final_prices = [r.get('final_prices', []) for r in run_results if 'final_prices' in r]
    final_profits = [r.get('final_profits', []) for r in run_results if 'final_profits' in r]
    
    price_stats = {}
    if final_prices:
        final_prices = np.array(final_prices)
        price_stats = {
            'mean_prices': np.mean(final_prices, axis=0).tolist(),
            'std_prices': np.std(final_prices, axis=0).tolist()
        }
    
    profit_stats = {}
    if final_profits:
        final_profits = np.array(final_profits)
        profit_stats = {
            'mean_profits': np.mean(final_profits, axis=0).tolist(),
            'std_profits': np.std(final_profits, axis=0).tolist()
        }
    
    # Aggregate all results
    analysis_results = {
        'convergence_stats': convergence_stats,
        'price_stats': price_stats,
        'profit_stats': profit_stats,
        'total_runs': len(run_results),
        'successful_runs': len([r for r in run_results if r.get('overall_converged', False)])
    }
    
    print("✓ Analysis completed")
    return analysis_results


def generate_outputs(analysis_results: Dict, params: SimParams, output_dir: str) -> None:
    """
    Generate simplified figures and tables from analysis results.
    
    Args:
        analysis_results: Dictionary containing analysis results
        params: Simulation parameters
        output_dir: Output directory path
    """
    print("=== GENERATING OUTPUTS ===")
    
    figures_dir = Path(output_dir) / "figures"
    tables_dir = Path(output_dir) / "tables"
    
    # Ensure directories exist
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate basic figure
    if PLOTTING_AVAILABLE and 'price_stats' in analysis_results:
        print("→ Generating basic convergence figure...")
        
        try:
            plt.figure(figsize=(8, 6))
            
            price_stats = analysis_results['price_stats']
            if 'mean_prices' in price_stats:
                mean_prices = price_stats['mean_prices']
                agents = range(len(mean_prices))
                
                plt.bar(agents, mean_prices)
                plt.xlabel('Agent')
                plt.ylabel('Final Price')
                plt.title('Mean Final Prices by Agent')
                plt.grid(True, alpha=0.3)
                
                # Add error bars if available
                if 'std_prices' in price_stats:
                    plt.errorbar(agents, mean_prices, yerr=price_stats['std_prices'], 
                               fmt='none', color='red', capsize=5)
                
                plt.tight_layout()
                plt.savefig(figures_dir / "convergence_analysis.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"✓ Basic figure saved to {figures_dir}")
            
        except Exception as e:
            print(f"❌ Figure generation failed: {e}")
    
    # Generate basic table
    print("→ Generating basic summary table...")
    
    try:
        # Create summary CSV
        summary_data = []
        
        # Add convergence stats
        conv_stats = analysis_results.get('convergence_stats', {})
        if isinstance(conv_stats, dict):
            summary_data.append(['Metric', 'Value'])
            summary_data.append(['Total Runs', analysis_results.get('total_runs', 0)])
            summary_data.append(['Successful Runs', analysis_results.get('successful_runs', 0)])
            summary_data.append(['Convergence Rate', f"{conv_stats.get('conv_rate', 0):.3f}"])
        
        # Add price stats
        price_stats = analysis_results.get('price_stats', {})
        if 'mean_prices' in price_stats:
            for i, price in enumerate(price_stats['mean_prices']):
                summary_data.append([f'Mean Price Agent {i+1}', f"{price:.4f}"])
        
        # Save to CSV
        import csv
        with open(tables_dir / "summary_results.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(summary_data)
        
        print(f"✓ Summary table saved to {tables_dir}")
        
    except Exception as e:
        print(f"❌ Table generation failed: {e}")


def run_deep_analysis_mode(run_results: List[Dict], params: SimParams, output_dir: str) -> Dict:
    """
    Run deep analysis mode with advanced Phase 2 features.
    
    Args:
        run_results: List of simulation results
        params: Simulation parameters
        output_dir: Output directory path
        
    Returns:
        Dictionary containing analysis results
    """
    print("=== DEEP ANALYSIS MODE ===")
    
    # Import advanced analysis modules
    try:
        from analysis.qgap_to_maximum import summary_qgap
        from analysis.learning_trajectory import run_monte_carlo_trajectories
        from analysis.detailed_analysis import create_detailed_statistics
        
        # Q-gap analysis
        print("→ Running Q-gap analysis...")
        Q_matrices = []
        for result in run_results:
            if 'Q_matrices' in result:
                Q_matrices.extend(result['Q_matrices'])
        
        if Q_matrices:
            qgap_summary = summary_qgap(Q_matrices)
            print(f"✓ Q-gap analysis completed for {len(Q_matrices)} Q-matrices")
        else:
            qgap_summary = None
            print("⚠ No Q-matrices found for Q-gap analysis")
        
        # Learning trajectory analysis
        print("→ Running learning trajectory analysis...")
        seeds = [result.get('seed', i) for i, result in enumerate(run_results)]
        if seeds:
            trajectory_results = run_monte_carlo_trajectories(params, seeds[:5], output_dir)  # Limit to 5 for performance
            print(f"✓ Trajectory analysis completed for {len(seeds[:5])} seeds")
        else:
            trajectory_results = None
            print("⚠ No seeds found for trajectory analysis")
        
        # Detailed agent-specific analysis
        print("→ Running detailed agent analysis...")
        if run_results:
            # Use first successful run for detailed analysis
            first_result = run_results[0]
            detailed_stats = create_detailed_statistics(first_result, params, output_dir)
            print(f"✓ Detailed analysis completed")
        else:
            detailed_stats = None
            print("⚠ No results available for detailed analysis")
        
        # Combine standard and deep analysis results
        standard_results = run_analysis_mode(run_results, params, output_dir)
        
        deep_results = {
            **standard_results,
            'qgap_analysis': qgap_summary.to_dict() if qgap_summary is not None and hasattr(qgap_summary, 'to_dict') else None,
            'trajectory_analysis': trajectory_results,
            'detailed_stats': detailed_stats.to_dict() if detailed_stats is not None and hasattr(detailed_stats, 'to_dict') else None,
            'analysis_mode': 'deep'
        }
        
        print("✓ Deep analysis completed")
        return deep_results
        
    except ImportError as e:
        print(f"⚠ Deep analysis modules not available: {e}")
        print("Falling back to standard analysis...")
        return run_analysis_mode(run_results, params, output_dir)
    except Exception as e:
        print(f"❌ Deep analysis failed: {e}")
        print("Falling back to standard analysis...")
        return run_analysis_mode(run_results, params, output_dir)


def run_benchmark_mode(args, config: Dict, params: SimParams, output_dir: str) -> int:
    """
    Run benchmark mode to compare against Fortran implementation.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
        params: Simulation parameters
        output_dir: Output directory path
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("=== BENCHMARK MODE ===")
    
    try:
        from benchmark.compare_fortran import validate_implementation, TOLERANCE_EPSILON
        
        # Validate implementation against Fortran results
        print(f"→ Comparing against Fortran results: {args.fortran}")
        print(f"→ Using seed: {args.seed}")
        print(f"→ Tolerance: {args.tolerance:.2e}")
        
        result = validate_implementation(
            fortran_csv_path=args.fortran,
            params=params,
            seed=args.seed,
            tolerance=args.tolerance
        )
        
        # Save benchmark report
        import json
        report_path = Path(output_dir) / "benchmark_report.json"
        report_data = {
            'passed': result.passed,
            'price_rmse': result.price_rmse,
            'price_max_error': result.price_max_error,
            'profit_rmse': result.profit_rmse,
            'profit_max_error': result.profit_max_error,
            'n_timesteps': result.n_timesteps,
            'n_agents': result.n_agents,
            'fortran_file': result.fortran_file,
            'python_seed': result.python_seed,
            'tolerance_used': args.tolerance,
            'errors': result.errors,
            'warnings': result.warnings
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"✓ Benchmark report saved to {report_path}")
        
        if result.passed:
            print("✅ BENCHMARK PASSED")
            print(f"   Price RMSE: {result.price_rmse:.2e}")
            print(f"   Price Max Error: {result.price_max_error:.2e}")
            if not np.isnan(result.profit_rmse):
                print(f"   Profit RMSE: {result.profit_rmse:.2e}")
                print(f"   Profit Max Error: {result.profit_max_error:.2e}")
            return 0
        else:
            print("❌ BENCHMARK FAILED")
            print(f"   Errors: {'; '.join(result.errors)}")
            return 1
            
    except ImportError as e:
        print(f"❌ Benchmark modules not available: {e}")
        return 1
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return 1


def main():
    """Main execution function with mode support."""
    parser = argparse.ArgumentParser(description="Calvano Q-learning simulation and analysis")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    parser.add_argument("--mode", type=str, choices=["simulate", "analyse", "full", "benchmark"], default="simulate",
                      help="Execution mode: simulate, analyse, full, or benchmark")
    parser.add_argument("--logdir", type=str, help="Directory containing logs for analyse mode")
    parser.add_argument("--output", type=str, help="Output directory (default: auto-generated)")
    parser.add_argument("--n-runs", type=int, help="Number of simulation runs (overrides config)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (validate only)")
    
    # Deep analysis mode arguments
    parser.add_argument("--analysis-mode", type=str, choices=["standard", "deep"], default="standard",
                      help="Analysis mode: standard or deep (with advanced features)")
    
    # Benchmark mode arguments
    parser.add_argument("--fortran", type=str, help="Path to Fortran output CSV file for benchmarking")
    parser.add_argument("--python", type=str, help="Path to Python simulation results for benchmarking")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for benchmark comparison")
    parser.add_argument("--tolerance", type=float, default=1e-12, help="Error tolerance for benchmark validation")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        params = SimParams(config)
        print(f"✓ Loaded config: {args.config}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return 1
    
    # Setup output directory
    if args.output:
        output_dir = args.output
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    else:
        output_dir = setup_output_directory()
    
    print(f"✓ Output directory: {output_dir}")
    
    # Dry run mode
    if args.dry_run:
        print("✓ Dry run completed successfully")
        return 0
    
    # Execute based on mode
    try:
        if args.mode == "simulate":
            # Simulate only
            run_results = run_simulation_mode(config, output_dir, args.n_runs)
            print(f"✓ Simulation completed. Results saved to {output_dir}")
            
        elif args.mode == "analyse":
            # Analyse only
            if not args.logdir:
                print("❌ --logdir required for analyse mode")
                return 1
            
            run_results = load_run_logs(args.logdir)
            
            if args.analysis_mode == "deep":
                # Run deep analysis with advanced features
                analysis_results = run_deep_analysis_mode(run_results, params, output_dir)
            else:
                # Standard analysis
                analysis_results = run_analysis_mode(run_results, params, output_dir)
            
            generate_outputs(analysis_results, params, output_dir)
            print(f"✓ Analysis completed. Results saved to {output_dir}")
            
        elif args.mode == "full":
            # Simulate + Analyse
            run_results = run_simulation_mode(config, output_dir, args.n_runs)
            analysis_results = run_analysis_mode(run_results, params, output_dir)
            generate_outputs(analysis_results, params, output_dir)
            print(f"✓ Full pipeline completed. Results saved to {output_dir}")
            print(f"✓ Saved figures to {output_dir}/figures/")
            print(f"✓ Saved tables to {output_dir}/tables/")
            
        elif args.mode == "benchmark":
            # Benchmark mode
            if not args.fortran:
                print("❌ --fortran required for benchmark mode")
                return 1
            
            return run_benchmark_mode(args, config, params, output_dir)
        
        return 0
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 