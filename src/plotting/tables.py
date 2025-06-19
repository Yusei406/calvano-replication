"""
Table generation module.

Implements functions to generate CSV tables for the paper's results.
Based on the paper's Table 1-3 specifications.

References:
    - Table 1: Summary statistics and convergence results
    - Table 2: Profit analysis and welfare comparisons
    - Table 3: Robustness checks and parameter sensitivity
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from ..analysis.convergence_results import ConvergenceStats, to_dataframe
from ..analysis.impulse_response import ImpulseResponseResult


def generate_table1(stats_dict: Dict[str, ConvergenceStats], 
                   save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate Table 1: Summary statistics and convergence results.
    
    Args:
        stats_dict: Dictionary mapping experiment names to ConvergenceStats
        save_path: Path to save CSV file
        
    Returns:
        DataFrame with Table 1 format
    """
    # Create individual DataFrames for each experiment
    dfs = []
    for exp_name, stats in stats_dict.items():
        df = to_dataframe(stats, exp_name)
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    # Combine all experiments
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Reorder columns for paper format
    base_columns = ['Experiment', 'N_Runs', 'Convergence_Rate', 'Nash_Gap', 'Coop_Gap']
    
    # Find all agent-specific columns
    agent_price_cols = sorted([col for col in combined_df.columns if col.startswith('Mean_Price_Agent_')])
    agent_profit_cols = sorted([col for col in combined_df.columns if col.startswith('Mean_Profit_Agent_')])
    
    # Additional columns
    extra_columns = [col for col in combined_df.columns 
                    if col not in base_columns + agent_price_cols + agent_profit_cols]
    
    # Final column order
    final_columns = base_columns + agent_price_cols + agent_profit_cols + extra_columns
    final_columns = [col for col in final_columns if col in combined_df.columns]
    
    table1_df = combined_df[final_columns].copy()
    
    # Add summary row
    if len(table1_df) > 1:
        summary_row = {}
        summary_row['Experiment'] = 'AVERAGE'
        
        for col in table1_df.columns:
            if col == 'Experiment':
                continue
            elif col == 'N_Runs':
                summary_row[col] = int(table1_df[col].sum())
            elif col in ['Convergence_Rate', 'Nash_Gap', 'Coop_Gap', 'Price_Volatility']:
                # Convert string percentages back to float for averaging
                values = []
                for val in table1_df[col]:
                    try:
                        values.append(float(val))
                    except:
                        values.append(0.0)
                summary_row[col] = f"{np.mean(values):.4f}"
            elif col.startswith(('Mean_Price_', 'Mean_Profit_', 'Std_')):
                # Average numerical columns
                values = []
                for val in table1_df[col]:
                    try:
                        values.append(float(val))
                    except:
                        values.append(0.0)
                summary_row[col] = f"{np.mean(values):.4f}"
            else:
                summary_row[col] = ""
        
        summary_df = pd.DataFrame([summary_row])
        table1_df = pd.concat([table1_df, summary_df], ignore_index=True)
    
    if save_path:
        table1_df.to_csv(save_path, index=False)
        print(f"Table 1 saved: {save_path}")
    
    return table1_df


def generate_table2(profit_analysis_dict: Dict[str, Dict[str, Any]], 
                   save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate Table 2: Profit analysis and welfare comparisons.
    
    Args:
        profit_analysis_dict: Dictionary with profit analysis results
        save_path: Path to save CSV file
        
    Returns:
        DataFrame with Table 2 format
    """
    rows = []
    
    for exp_name, analysis in profit_analysis_dict.items():
        row = {
            'Experiment': exp_name,
            'Mean_Total_Welfare': f"{analysis.get('total_welfare', 0):.4f}",
            'Nash_Welfare': f"{analysis.get('nash_welfare', 0):.4f}",
            'Coop_Welfare': f"{analysis.get('coop_welfare', 0):.4f}",
            'Efficiency_Ratio': f"{analysis.get('efficiency_ratio', 0):.4f}",
            'Welfare_Gain_vs_Nash_Pct': f"{analysis.get('welfare_gain_vs_nash_percent', 0):.2f}%",
            'Welfare_Loss_vs_Coop_Pct': f"{-analysis.get('welfare_gain_vs_coop_percent', 0):.2f}%"
        }
        
        # Add agent-specific profit gains
        nash_gains = analysis.get('nash_gains_percent', [])
        coop_gains = analysis.get('coop_gains_percent', [])
        random_gains = analysis.get('random_gains_percent', [])
        
        for i, (nash_gain, coop_gain, random_gain) in enumerate(zip(nash_gains, coop_gains, random_gains)):
            row[f'Agent_{i+1}_Nash_Gain_Pct'] = f"{nash_gain:.2f}%"
            row[f'Agent_{i+1}_Coop_Gain_Pct'] = f"{coop_gain:.2f}%"
            row[f'Agent_{i+1}_Random_Gain_Pct'] = f"{random_gain:.2f}%"
        
        rows.append(row)
    
    table2_df = pd.DataFrame(rows)
    
    if len(table2_df) > 1:
        # Add average row
        summary_row = {'Experiment': 'AVERAGE'}
        
        for col in table2_df.columns:
            if col == 'Experiment':
                continue
            
            # Extract numerical values
            values = []
            for val in table2_df[col]:
                try:
                    # Remove percentage signs and convert
                    clean_val = str(val).replace('%', '')
                    values.append(float(clean_val))
                except:
                    values.append(0.0)
            
            if col.endswith('_Pct'):
                summary_row[col] = f"{np.mean(values):.2f}%"
            else:
                summary_row[col] = f"{np.mean(values):.4f}"
        
        summary_df = pd.DataFrame([summary_row])
        table2_df = pd.concat([table2_df, summary_df], ignore_index=True)
    
    if save_path:
        table2_df.to_csv(save_path, index=False)
        print(f"Table 2 saved: {save_path}")
    
    return table2_df


def generate_table3(robustness_results: Dict[str, Dict[str, Any]], 
                   save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate Table 3: Robustness checks and parameter sensitivity.
    
    Args:
        robustness_results: Dictionary with robustness test results
        save_path: Path to save CSV file
        
    Returns:
        DataFrame with Table 3 format
    """
    rows = []
    
    for test_name, results in robustness_results.items():
        row = {
            'Test': test_name,
            'Parameter_Varied': results.get('parameter', 'N/A'),
            'Baseline_Value': f"{results.get('baseline_value', 0):.4f}",
            'Test_Value': f"{results.get('test_value', 0):.4f}",
            'Baseline_Conv_Rate': f"{results.get('baseline_convergence_rate', 0):.3f}",
            'Test_Conv_Rate': f"{results.get('test_convergence_rate', 0):.3f}",
            'Conv_Rate_Change': f"{results.get('convergence_rate_change', 0):.3f}",
            'Baseline_Nash_Gap': f"{results.get('baseline_nash_gap', 0):.4f}",
            'Test_Nash_Gap': f"{results.get('test_nash_gap', 0):.4f}",
            'Nash_Gap_Change': f"{results.get('nash_gap_change', 0):.4f}",
            'Statistical_Significance': results.get('significant', 'N/A')
        }
        
        rows.append(row)
    
    table3_df = pd.DataFrame(rows)
    
    if save_path:
        table3_df.to_csv(save_path, index=False)
        print(f"Table 3 saved: {save_path}")
    
    return table3_df


def generate_summary_statistics_table(all_results: Dict[str, Any], 
                                    save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate comprehensive summary statistics table.
    
    Args:
        all_results: Dictionary with all analysis results
        save_path: Path to save CSV file
        
    Returns:
        Summary statistics DataFrame
    """
    summary_data = []
    
    # Basic simulation statistics
    summary_data.append({
        'Statistic': 'Total Simulation Runs',
        'Value': str(all_results.get('total_runs', 0)),
        'Description': 'Total number of Q-learning simulations executed'
    })
    
    summary_data.append({
        'Statistic': 'Overall Convergence Rate',
        'Value': f"{all_results.get('overall_convergence_rate', 0):.3f}",
        'Description': 'Fraction of runs that achieved convergence'
    })
    
    summary_data.append({
        'Statistic': 'Mean Nash Distance',
        'Value': f"{all_results.get('mean_nash_distance', 0):.4f}",
        'Description': 'Average distance from Nash equilibrium'
    })
    
    summary_data.append({
        'Statistic': 'Mean Cooperative Distance',
        'Value': f"{all_results.get('mean_coop_distance', 0):.4f}",
        'Description': 'Average distance from cooperative equilibrium'
    })
    
    # Efficiency measures
    if 'efficiency_measures' in all_results:
        eff = all_results['efficiency_measures']
        summary_data.append({
            'Statistic': 'Average Efficiency Ratio',
            'Value': f"{eff.get('mean_efficiency_ratio', 0):.4f}",
            'Description': 'Mean efficiency relative to first-best outcome'
        })
        
        summary_data.append({
            'Statistic': 'Welfare Gain vs Nash (%)',
            'Value': f"{eff.get('welfare_gain_vs_nash_percent', 0):.2f}%",
            'Description': 'Welfare improvement over Nash equilibrium'
        })
    
    # Learning dynamics
    if 'learning_dynamics' in all_results:
        ld = all_results['learning_dynamics']
        summary_data.append({
            'Statistic': 'Mean Convergence Time',
            'Value': f"{ld.get('mean_convergence_time', 0):.1f}",
            'Description': 'Average time steps to convergence'
        })
        
        summary_data.append({
            'Statistic': 'Price Volatility Decline',
            'Value': f"{ld.get('volatility_decline_rate', 0):.4f}",
            'Description': 'Rate of volatility reduction during learning'
        })
    
    # Robustness indicators
    if 'robustness_summary' in all_results:
        rob = all_results['robustness_summary']
        summary_data.append({
            'Statistic': 'Parameter Sensitivity',
            'Value': f"{rob.get('sensitivity_score', 0):.3f}",
            'Description': 'Overall sensitivity to parameter changes'
        })
        
        summary_data.append({
            'Statistic': 'Stable Equilibria Fraction',
            'Value': f"{rob.get('stable_equilibria_fraction', 0):.3f}",
            'Description': 'Fraction of runs reaching stable equilibria'
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    if save_path:
        summary_df.to_csv(save_path, index=False)
        print(f"Summary statistics saved: {save_path}")
    
    return summary_df


def create_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """
    Convert DataFrame to LaTeX table format.
    
    Args:
        df: DataFrame to convert
        caption: Table caption
        label: Table label for referencing
        
    Returns:
        LaTeX table string
    """
    latex_str = "\\begin{table}[htbp]\n"
    latex_str += "\\centering\n"
    latex_str += f"\\caption{{{caption}}}\n"
    latex_str += f"\\label{{{label}}}\n"
    
    # Create tabular environment
    n_cols = len(df.columns)
    col_spec = "l" + "c" * (n_cols - 1)
    latex_str += f"\\begin{{tabular}}{{{col_spec}}}\n"
    latex_str += "\\toprule\n"
    
    # Header
    header = " & ".join(df.columns) + " \\\\\n"
    latex_str += header
    latex_str += "\\midrule\n"
    
    # Data rows
    for _, row in df.iterrows():
        row_str = " & ".join(str(val) for val in row.values) + " \\\\\n"
        latex_str += row_str
    
    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\end{table}\n"
    
    return latex_str


def export_all_tables(results_dict: Dict[str, Any], output_dir: str = "tables/") -> None:
    """
    Export all tables to specified directory.
    
    Args:
        results_dict: Dictionary with all analysis results
        output_dir: Directory to save tables
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Table 1: Convergence Results
    if 'convergence_stats' in results_dict:
        table1 = generate_table1(results_dict['convergence_stats'], 
                                str(output_path / "table1_convergence.csv"))
    
    # Table 2: Profit Analysis
    if 'profit_analysis' in results_dict:
        table2 = generate_table2(results_dict['profit_analysis'],
                                str(output_path / "table2_profits.csv"))
    
    # Table 3: Robustness
    if 'robustness_results' in results_dict:
        table3 = generate_table3(results_dict['robustness_results'],
                                str(output_path / "table3_robustness.csv"))
    
    # Summary Statistics
    summary_table = generate_summary_statistics_table(results_dict,
                                                     str(output_path / "summary_statistics.csv"))
    
    print(f"All tables exported to {output_dir}")


def compare_with_baseline(new_results: Dict[str, Any], baseline_results: Dict[str, Any],
                         save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create comparison table between new results and baseline.
    
    Args:
        new_results: New analysis results
        baseline_results: Baseline results for comparison
        save_path: Path to save comparison table
        
    Returns:
        Comparison DataFrame
    """
    comparison_data = []
    
    metrics = [
        ('Convergence Rate', 'convergence_rate'),
        ('Nash Distance', 'nash_distance'),
        ('Cooperative Distance', 'coop_distance'),
        ('Total Welfare', 'total_welfare'),
        ('Efficiency Ratio', 'efficiency_ratio')
    ]
    
    for metric_name, metric_key in metrics:
        baseline_val = baseline_results.get(metric_key, 0)
        new_val = new_results.get(metric_key, 0)
        
        if baseline_val != 0:
            pct_change = ((new_val - baseline_val) / baseline_val) * 100
        else:
            pct_change = 0 if new_val == 0 else float('inf')
        
        comparison_data.append({
            'Metric': metric_name,
            'Baseline': f"{baseline_val:.4f}",
            'New': f"{new_val:.4f}",
            'Absolute_Change': f"{new_val - baseline_val:.4f}",
            'Percent_Change': f"{pct_change:.2f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if save_path:
        comparison_df.to_csv(save_path, index=False)
        print(f"Comparison table saved: {save_path}")
    
    return comparison_df


def create_impulse_response_table(
    results: List[ImpulseResponseResult],
    title: str = "Impulse Response Analysis Results"
) -> pd.DataFrame:
    """
    Create a table summarizing impulse response analysis results.
    
    Args:
        results: List of ImpulseResponseResult objects
        title: Table title
        
    Returns:
        DataFrame containing summary statistics
    """
    if not results:
        return pd.DataFrame()
    
    # Extract key metrics
    data = []
    for i, result in enumerate(results):
        row = {
            'Shock_Time': result.shock_time,
            'Shock_Magnitude': result.shock_magnitude,
            'Affected_Agent': result.affected_agent,
            'Max_Deviation': result.max_deviation,
            'Convergence_Time': result.convergence_time if result.convergence_time is not None else np.nan,
            'Recovery_Rate': result.recovery_rate if result.recovery_rate is not None else np.nan
        }
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add summary statistics
    summary = {
        'Mean_Max_Deviation': df['Max_Deviation'].mean(),
        'Mean_Convergence_Time': df['Convergence_Time'].mean(),
        'Mean_Recovery_Rate': df['Recovery_Rate'].mean(),
        'Convergence_Rate': df['Convergence_Time'].notna().mean()
    }
    
    # Add summary row
    df.loc['Summary'] = summary
    
    return df


def create_shock_statistics_table(
    stats: Dict[str, float],
    title: str = "Shock Response Statistics"
) -> pd.DataFrame:
    """
    Create a table of aggregate shock statistics.
    
    Args:
        stats: Dictionary of shock statistics
        title: Table title
        
    Returns:
        DataFrame containing statistics
    """
    if not stats:
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame([stats])
    
    # Rename columns for display
    column_map = {
        'mean_convergence_time': 'Mean Convergence Time',
        'mean_max_deviation': 'Mean Max Deviation',
        'mean_recovery_rate': 'Mean Recovery Rate',
        'convergence_rate': 'Convergence Rate'
    }
    
    df = df.rename(columns=column_map)
    
    return df


def create_optimal_shock_table(
    result: ImpulseResponseResult,
    title: str = "Optimal Shock Analysis Results"
) -> pd.DataFrame:
    """
    Create a table summarizing optimal shock analysis results.
    
    Args:
        result: ImpulseResponseResult from optimal shock
        title: Table title
        
    Returns:
        DataFrame containing results
    """
    if not result.price_response.size:
        return pd.DataFrame()
    
    # Extract key metrics
    data = {
        'Shock_Time': result.shock_time,
        'Shock_Magnitude': result.shock_magnitude,
        'Affected_Agent': result.affected_agent,
        'Max_Deviation': result.max_deviation,
        'Convergence_Time': result.convergence_time if result.convergence_time is not None else np.nan,
        'Recovery_Rate': result.recovery_rate if result.recovery_rate is not None else np.nan
    }
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    return df


def format_impulse_response_table(
    df: pd.DataFrame,
    precision: int = 4
) -> pd.DataFrame:
    """
    Format impulse response table for display.
    
    Args:
        df: Input DataFrame
        precision: Number of decimal places
        
    Returns:
        Formatted DataFrame
    """
    if df.empty:
        return df
    
    # Format numeric columns
    numeric_cols = ['Shock_Magnitude', 'Max_Deviation', 'Convergence_Time', 'Recovery_Rate']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f'{x:.{precision}f}' if pd.notnull(x) else 'N/A')
    
    # Format integer columns
    int_cols = ['Shock_Time', 'Affected_Agent']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f'{int(x)}' if pd.notnull(x) else 'N/A')
    
    return df


def save_impulse_response_tables(
    results: List[ImpulseResponseResult],
    stats: Dict[str, float],
    optimal_result: Optional[ImpulseResponseResult] = None,
    output_dir: str = "results",
    prefix: str = "impulse_response"
) -> None:
    """
    Save all impulse response tables to CSV files.
    
    Args:
        results: List of ImpulseResponseResult objects
        stats: Dictionary of shock statistics
        optimal_result: Optional ImpulseResponseResult from optimal shock
        output_dir: Directory to save tables
        prefix: Prefix for output filenames
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results table
    results_df = create_impulse_response_table(results)
    if not results_df.empty:
        results_df = format_impulse_response_table(results_df)
        results_df.to_csv(os.path.join(output_dir, f"{prefix}_results.csv"), index=False)
    
    # Save statistics table
    stats_df = create_shock_statistics_table(stats)
    if not stats_df.empty:
        stats_df = format_impulse_response_table(stats_df)
        stats_df.to_csv(os.path.join(output_dir, f"{prefix}_statistics.csv"), index=False)
    
    # Save optimal shock table if provided
    if optimal_result is not None:
        optimal_df = create_optimal_shock_table(optimal_result)
        if not optimal_df.empty:
            optimal_df = format_impulse_response_table(optimal_df)
            optimal_df.to_csv(os.path.join(output_dir, f"{prefix}_optimal.csv"), index=False) 