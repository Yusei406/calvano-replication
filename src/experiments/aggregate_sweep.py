"""
Aggregate Parameter Sweep Results for Calvano Q-learning.

Processes parameter sweep outputs:
- Aggregates convergence rates and other metrics
- Generates Table 2 compatible grid_summary.csv/.tex
- Creates visualization of parameter effects
- Provides statistical analysis of parameter sensitivity
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings


def load_sweep_results(sweep_dir: str) -> Dict[str, Any]:
    """
    Load parameter sweep results from directory.
    
    Args:
        sweep_dir: Directory containing sweep results
        
    Returns:
        Dictionary with sweep data and metadata
    """
    sweep_path = Path(sweep_dir)
    
    # Load sweep summary
    summary_path = sweep_path / "sweep_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Sweep summary not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        sweep_summary = json.load(f)
    
    return {
        'sweep_summary': sweep_summary,
        'sweep_dir': str(sweep_path)
    }


def extract_simulation_results(sweep_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract simulation results into a structured DataFrame.
    
    Args:
        sweep_data: Sweep data dictionary
        
    Returns:
        DataFrame with parameter combinations and results
    """
    results = sweep_data['sweep_summary']['results']
    
    rows = []
    for result in results:
        if result['status'] == 'success' and 'simulation_results' in result:
            # Extract parameters
            params = result['config'].copy()
            
            # Extract simulation metrics
            sim_results = result['simulation_results']
            
            # Combine into single row
            row = {
                'run_id': result['run_id'],
                'status': result['status'],
                'elapsed_time': result['elapsed_time'],
                **params,  # Parameter values
                **sim_results  # Simulation results
            }
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def aggregate_by_parameters(df: pd.DataFrame, param_columns: List[str]) -> pd.DataFrame:
    """
    Aggregate results by parameter combinations.
    
    Args:
        df: DataFrame with individual run results
        param_columns: List of parameter column names
        
    Returns:
        DataFrame with aggregated statistics
    """
    # Only use successful runs for aggregation
    success_df = df[df['status'] == 'success'].copy()
    
    if success_df.empty:
        warnings.warn("No successful runs found for aggregation")
        return pd.DataFrame()
    
    # Define metrics to aggregate
    metric_columns = [
        'convergence_rate', 'avg_convergence_time', 'nash_price', 
        'mean_profit', 'cooperative_gap'
    ]
    
    # Filter to available metrics
    available_metrics = [col for col in metric_columns if col in success_df.columns]
    
    # Group by parameters and aggregate
    grouped = success_df.groupby(param_columns)
    
    aggregated_stats = {}
    
    for metric in available_metrics:
        if success_df[metric].dtype in ['float64', 'int64']:
            # Numeric metrics - compute mean, std, min, max
            metric_stats = grouped[metric].agg(['count', 'mean', 'std', 'min', 'max'])
            metric_stats.columns = [f'{metric}_{stat}' for stat in metric_stats.columns]
            
            for col in metric_stats.columns:
                aggregated_stats[col] = metric_stats[col]
    
    # Combine all statistics
    agg_df = pd.DataFrame(aggregated_stats).reset_index()
    
    return agg_df


def generate_grid_summary_table(df: pd.DataFrame, param_columns: List[str], output_dir: str) -> Tuple[str, str]:
    """
    Generate Table 2 compatible grid summary in CSV and LaTeX formats.
    
    Args:
        df: Aggregated results DataFrame
        param_columns: List of parameter column names
        output_dir: Output directory
        
    Returns:
        Tuple of (CSV path, LaTeX path)
    """
    # Create a formatted table for publication
    display_df = df.copy()
    
    # Format numeric columns to 3 decimal places
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in param_columns:  # Don't format parameter values
            display_df[col] = display_df[col].round(3)
    
    # Create formatted strings for mean ± std
    if 'convergence_rate_mean' in display_df.columns and 'convergence_rate_std' in display_df.columns:
        display_df['Convergence Rate'] = display_df.apply(
            lambda row: f"{row['convergence_rate_mean']:.3f} ± {row['convergence_rate_std']:.3f}",
            axis=1
        )
    
    # Select columns for final table
    final_columns = param_columns[:]
    
    # Add formatted metrics if available
    if 'Convergence Rate' in display_df.columns:
        final_columns.append('Convergence Rate')
    
    # Filter to available columns
    final_columns = [col for col in final_columns if col in display_df.columns]
    final_df = display_df[final_columns]
    
    # Save CSV
    os.makedirs(Path(output_dir) / "tables", exist_ok=True)
    csv_path = Path(output_dir) / "tables" / "grid_summary.csv"
    final_df.to_csv(csv_path, index=False)
    
    # Generate LaTeX
    latex_content = generate_grid_latex_table(final_df)
    tex_path = Path(output_dir) / "tables" / "grid_summary.tex"
    
    with open(tex_path, 'w') as f:
        f.write(latex_content)
    
    print(f"✓ Grid summary table generated:")
    print(f"  CSV: {csv_path}")
    print(f"  LaTeX: {tex_path}")
    
    return str(csv_path), str(tex_path)


def generate_grid_latex_table(df: pd.DataFrame) -> str:
    """
    Generate LaTeX table for grid summary.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        LaTeX table string
    """
    # Start LaTeX table
    n_cols = len(df.columns)
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Parameter Sweep Results Summary}",
        "\\label{tab:grid_summary}",
        f"\\begin{{tabular}}{{'l' * n_cols}}",
        "\\toprule"
    ]
    
    # Add header
    header = " & ".join(df.columns) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")
    
    # Add data rows
    for _, row in df.iterrows():
        row_str = " & ".join(str(val) for val in row) + " \\\\"
        latex_lines.append(row_str)
    
    # Close table
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_lines)


def aggregate_parameter_sweep(sweep_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Complete aggregation of parameter sweep results.
    
    Args:
        sweep_dir: Directory containing sweep results
        output_dir: Output directory for aggregated results
        
    Returns:
        Dictionary with aggregation results and paths
    """
    print(f"Aggregating parameter sweep from: {sweep_dir}")
    
    # Load sweep data
    sweep_data = load_sweep_results(sweep_dir)
    
    # Extract results to DataFrame
    results_df = extract_simulation_results(sweep_data)
    
    if results_df.empty:
        raise ValueError("No results found in sweep directory")
    
    # Identify parameter columns
    config_keys = set()
    for result in sweep_data['sweep_summary']['results']:
        config_keys.update(result['config'].keys())
    
    param_columns = [col for col in config_keys if col in results_df.columns]
    
    print(f"Found {len(results_df)} results with parameters: {param_columns}")
    
    # Aggregate by parameters
    agg_df = aggregate_by_parameters(results_df, param_columns)
    
    if agg_df.empty:
        warnings.warn("No successful runs to aggregate")
        return {'error': 'No successful runs found'}
    
    # Generate outputs
    outputs = {}
    
    # Grid summary table
    csv_path, tex_path = generate_grid_summary_table(agg_df, param_columns, output_dir)
    outputs['grid_summary_csv'] = csv_path
    outputs['grid_summary_tex'] = tex_path
    
    # Save aggregated data
    agg_data_path = Path(output_dir) / "tables" / "aggregated_results.csv"
    agg_df.to_csv(agg_data_path, index=False)
    outputs['aggregated_data'] = str(agg_data_path)
    
    print(f"✅ Parameter sweep aggregation completed:")
    print(f"  Aggregated {len(agg_df)} parameter combinations")
    print(f"  Results saved to: {output_dir}")
    
    return outputs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate parameter sweep results")
    parser.add_argument("--sweep-dir", required=True, help="Directory containing sweep results")
    parser.add_argument("--output", required=True, help="Output directory for aggregated results")
    
    args = parser.parse_args()
    
    try:
        result = aggregate_parameter_sweep(args.sweep_dir, args.output)
        
        if 'error' in result:
            print(f"❌ Aggregation failed: {result['error']}")
            exit(1)
        else:
            print("✅ Aggregation completed successfully")
            exit(0)
            
    except Exception as e:
        print(f"❌ Aggregation error: {e}")
        exit(1)
