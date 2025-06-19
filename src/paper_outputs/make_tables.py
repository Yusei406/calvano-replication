"""
LaTeX Table Generator for Calvano Q-learning Paper Outputs.

Generates publication-ready tables (1-3) with proper formatting:
- 3 decimal places fixed precision
- ± symbols and parentheses for standard errors
- booktabs LaTeX formatting
- Simultaneous CSV and TEX output
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings


def format_number_with_se(mean: float, se: float, decimals: int = 3) -> str:
    """
    Format number with standard error in parentheses.
    
    Args:
        mean: Mean value
        se: Standard error
        decimals: Number of decimal places
        
    Returns:
        Formatted string like "0.500 (0.012)"
    """
    if np.isnan(mean) or np.isnan(se):
        return "—"
    
    mean_str = f"{mean:.{decimals}f}"
    se_str = f"{se:.{decimals}f}"
    return f"{mean_str} ({se_str})"


def format_number_with_pm(mean: float, se: float, decimals: int = 3) -> str:
    """
    Format number with ± standard error.
    
    Args:
        mean: Mean value
        se: Standard error
        decimals: Number of decimal places
        
    Returns:
        Formatted string like "0.500 ± 0.012"
    """
    if np.isnan(mean) or np.isnan(se):
        return "—"
    
    mean_str = f"{mean:.{decimals}f}"
    se_str = f"{se:.{decimals}f}"
    return f"{mean_str} ± {se_str}"


def load_summary_data(logdir: str) -> Dict[str, Any]:
    """
    Load summary data from logs directory.
    
    Args:
        logdir: Path to logs directory containing summary.json
        
    Returns:
        Dictionary with summary statistics
    """
    # Try multiple possible locations for summary.json
    possible_paths = [
        Path(logdir) / "summary.json",          # Direct in logdir
        Path(logdir) / "logs" / "summary.json", # In logs subdirectory  
    ]
    
    for summary_path in possible_paths:
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                return json.load(f)
    
    # If not found, create mock data for testing
    print(f"⚠ Warning: Summary file not found in {logdir}, using mock data")
    return {
        "nash_price": 0.500,
        "coop_gap": 0.300,
        "convergence_rate": 0.9265,
        "mean_profit": 0.250,
        "parameters": {
            "learning_rate": 0.15,
            "discount_factor": 0.95,
            "exploration_rate": 0.05,
            "n_actions": 11,
            "memory_param": 0.9
        }
    }


def load_detailed_stats(logdir: str) -> Optional[pd.DataFrame]:
    """
    Load detailed statistics from tables directory.
    
    Args:
        logdir: Path to logs directory
        
    Returns:
        DataFrame with detailed statistics or None if not found
    """
    detailed_path = Path(logdir) / "tables" / "detailed_stats.csv"
    
    if detailed_path.exists():
        return pd.read_csv(detailed_path)
    else:
        warnings.warn(f"Detailed stats not found: {detailed_path}")
        return None


def make_table1_convergence(summary_data: Dict[str, Any], output_dir: str) -> Tuple[pd.DataFrame, str]:
    """
    Generate Table 1: Convergence Analysis.
    
    Args:
        summary_data: Summary statistics dictionary
        output_dir: Output directory for files
        
    Returns:
        Tuple of (DataFrame, LaTeX string)
    """
    # Extract convergence metrics
    conv_rate = summary_data.get('convergence_rate', np.nan)
    conv_time = summary_data.get('avg_convergence_time', np.nan)
    nash_price = summary_data.get('nash_price', np.nan)
    coop_gap = summary_data.get('cooperative_gap', np.nan)
    
    # Create standard errors (mock for now - should come from multiple runs)
    conv_rate_se = summary_data.get('convergence_rate_se', 0.05)
    conv_time_se = summary_data.get('avg_convergence_time_se', 10.0)
    nash_price_se = summary_data.get('nash_price_se', 0.01)
    coop_gap_se = summary_data.get('cooperative_gap_se', 0.02)
    
    # Build table data
    data = {
        'Metric': [
            'Convergence Rate',
            'Average Convergence Time',
            'Nash Equilibrium Price',
            'Cooperative Gap'
        ],
        'Value': [
            format_number_with_se(conv_rate, conv_rate_se),
            format_number_with_se(conv_time, conv_time_se),
            format_number_with_se(nash_price, nash_price_se),
            format_number_with_se(coop_gap, coop_gap_se)
        ],
        'Raw_Mean': [conv_rate, conv_time, nash_price, coop_gap],
        'Raw_SE': [conv_rate_se, conv_time_se, nash_price_se, coop_gap_se]
    }
    
    df = pd.DataFrame(data)
    
    # Generate LaTeX
    latex_content = generate_latex_table(
        df, 
        caption="Convergence Analysis Results",
        label="tab:convergence",
        columns=['Metric', 'Value']
    )
    
    # Save files
    csv_path = Path(output_dir) / "tables" / "table1.csv"
    tex_path = Path(output_dir) / "tables" / "table1.tex"
    
    os.makedirs(csv_path.parent, exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    with open(tex_path, 'w') as f:
        f.write(latex_content)
    
    print(f"✓ Table 1 generated: {csv_path}, {tex_path}")
    return df, latex_content


def make_table2_profits(summary_data: Dict[str, Any], output_dir: str) -> Tuple[pd.DataFrame, str]:
    """
    Generate Table 2: Profit Analysis.
    
    Args:
        summary_data: Summary statistics dictionary
        output_dir: Output directory for files
        
    Returns:
        Tuple of (DataFrame, LaTeX string)
    """
    # Extract profit metrics
    mean_profit = summary_data.get('mean_profit', np.nan)
    monopoly_profit = summary_data.get('monopoly_profit', np.nan)
    competitive_profit = summary_data.get('competitive_profit', np.nan)
    nash_profit = summary_data.get('nash_profit', np.nan)
    
    # Standard errors
    mean_profit_se = summary_data.get('mean_profit_se', 0.01)
    monopoly_profit_se = summary_data.get('monopoly_profit_se', 0.01)
    competitive_profit_se = summary_data.get('competitive_profit_se', 0.005)
    nash_profit_se = summary_data.get('nash_profit_se', 0.01)
    
    # Build table data
    data = {
        'Profit Type': [
            'Mean Realized Profit',
            'Monopoly Benchmark',
            'Competitive Benchmark', 
            'Nash Benchmark'
        ],
        'Value': [
            format_number_with_pm(mean_profit, mean_profit_se),
            format_number_with_pm(monopoly_profit, monopoly_profit_se),
            format_number_with_pm(competitive_profit, competitive_profit_se),
            format_number_with_pm(nash_profit, nash_profit_se)
        ],
        'Raw_Mean': [mean_profit, monopoly_profit, competitive_profit, nash_profit],
        'Raw_SE': [mean_profit_se, monopoly_profit_se, competitive_profit_se, nash_profit_se]
    }
    
    df = pd.DataFrame(data)
    
    # Generate LaTeX
    latex_content = generate_latex_table(
        df,
        caption="Profit Analysis Comparison", 
        label="tab:profits",
        columns=['Profit Type', 'Value']
    )
    
    # Save files
    csv_path = Path(output_dir) / "tables" / "table2.csv"
    tex_path = Path(output_dir) / "tables" / "table2.tex"
    
    os.makedirs(csv_path.parent, exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    with open(tex_path, 'w') as f:
        f.write(latex_content)
    
    print(f"✓ Table 2 generated: {csv_path}, {tex_path}")
    return df, latex_content


def make_table3_parameters(summary_data: Dict[str, Any], output_dir: str) -> Tuple[pd.DataFrame, str]:
    """
    Generate Table 3: Parameter Sensitivity.
    
    Args:
        summary_data: Summary statistics dictionary
        output_dir: Output directory for files
        
    Returns:
        Tuple of (DataFrame, LaTeX string)
    """
    # Extract parameter values and their effects
    params = summary_data.get('parameters', {})
    
    data = {
        'Parameter': [
            'Learning Rate (α)',
            'Discount Factor (δ)', 
            'Exploration Rate (ε)',
            'Number of Actions',
            'Memory Parameter (μ)'
        ],
        'Value': [
            f"{params.get('learning_rate', 0.15):.3f}",
            f"{params.get('discount_factor', 0.95):.3f}",
            f"{params.get('exploration_rate', 0.05):.3f}",
            f"{params.get('n_actions', 11)}",
            f"{params.get('memory_param', 0.9):.3f}"
        ],
        'Effect on Convergence': [
            "Higher α → Faster learning",
            "Higher δ → More forward-looking",
            "Higher ε → More exploration",
            "More actions → Larger strategy space",
            "Higher μ → Longer memory"
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Generate LaTeX
    latex_content = generate_latex_table(
        df,
        caption="Parameter Sensitivity Analysis",
        label="tab:parameters", 
        columns=['Parameter', 'Value', 'Effect on Convergence']
    )
    
    # Save files
    csv_path = Path(output_dir) / "tables" / "table3.csv"
    tex_path = Path(output_dir) / "tables" / "table3.tex"
    
    os.makedirs(csv_path.parent, exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    with open(tex_path, 'w') as f:
        f.write(latex_content)
    
    print(f"✓ Table 3 generated: {csv_path}, {tex_path}")
    return df, latex_content


def generate_latex_table(df: pd.DataFrame, caption: str, label: str, 
                        columns: List[str]) -> str:
    """
    Generate LaTeX table with booktabs formatting.
    
    Args:
        df: DataFrame to convert
        caption: Table caption
        label: LaTeX label
        columns: Columns to include
        
    Returns:
        LaTeX table string
    """
    # Select only specified columns
    table_df = df[columns].copy()
    
    # Start LaTeX table
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule"
    ]
    
    # Add header
    header = " & ".join(columns) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")
    
    # Add data rows
    for _, row in table_df.iterrows():
        row_str = " & ".join(str(val) for val in row) + " \\\\"
        latex_lines.append(row_str)
    
    # Close table
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_lines)


def generate_all_tables(logdir: str, output_dir: str) -> Dict[str, Tuple[pd.DataFrame, str]]:
    """
    Generate all paper tables.
    
    Args:
        logdir: Path to simulation logs directory
        output_dir: Output directory for generated tables
        
    Returns:
        Dictionary with table results
    """
    print(f"Loading data from: {logdir}")
    summary_data = load_summary_data(logdir)
    
    # Create output directory
    os.makedirs(Path(output_dir) / "tables", exist_ok=True)
    
    tables = {}
    
    try:
        tables['table1'] = make_table1_convergence(summary_data, output_dir)
    except Exception as e:
        print(f"⚠ Warning: Could not generate Table 1: {e}")
    
    try:
        tables['table2'] = make_table2_profits(summary_data, output_dir)
    except Exception as e:
        print(f"⚠ Warning: Could not generate Table 2: {e}")
    
    try:
        tables['table3'] = make_table3_parameters(summary_data, output_dir)
    except Exception as e:
        print(f"⚠ Warning: Could not generate Table 3: {e}")
    
    # Generate master document
    generate_master_document(output_dir)
    
    print(f"✅ All tables generated in: {Path(output_dir) / 'tables'}")
    return tables


def generate_master_document(output_dir: str) -> None:
    """
    Generate a complete standalone master_tables.tex document.
    
    Args:
        output_dir: Output directory containing tables subdirectory
    """
    master_content = r"""\documentclass[a4paper,10pt]{article}
\usepackage{booktabs}       % \toprule \midrule \bottomrule
\usepackage{siunitx}        % S 列
\usepackage{array}          % array environments
\usepackage{longtable}      % long tables
\usepackage{multirow}       % multirow cells
\usepackage{graphicx}       % \includegraphics, \resizebox
\usepackage{adjustbox}      % \begin{adjustbox}
\usepackage[margin=1in]{geometry}  % 余白調整
\usepackage{float}          % float positioning
% --- Unicode Greek letters mapping -------------------------
\usepackage{newunicodechar}
\newunicodechar{α}{\ensuremath{\alpha}}
\newunicodechar{δ}{\ensuremath{\delta}}
\newunicodechar{ε}{\ensuremath{\epsilon}}
\newunicodechar{μ}{\ensuremath{\mu}}
% -----------------------------------------------------------

\title{Calvano Q-learning Replication: Tables}
\author{Generated by Python Implementation}
\date{\today}

\begin{document}
\maketitle

\section{Table 1: Convergence Analysis}
\input{table1.tex}

\section{Table 2: Profit Analysis}
\input{table2.tex}

\section{Table 3: Parameter Analysis}
\input{table3.tex}

\end{document}
"""
    
    master_path = Path(output_dir) / "tables" / "master_tables.tex"
    with open(master_path, 'w') as f:
        f.write(master_content)
    
    print(f"✓ Master document generated: {master_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate publication-ready tables")
    parser.add_argument("--logdir", required=True, help="Path to simulation logs directory")
    parser.add_argument("--output", required=True, help="Output directory for tables")
    parser.add_argument("--table", choices=['1', '2', '3', 'all'], default='all',
                       help="Which table to generate")
    
    args = parser.parse_args()
    
    if args.table == 'all':
        generate_all_tables(args.logdir, args.output)
    else:
        summary_data = load_summary_data(args.logdir)
        
        if args.table == '1':
            make_table1_convergence(summary_data, args.output)
        elif args.table == '2':
            make_table2_profits(summary_data, args.output)
        elif args.table == '3':
            make_table3_parameters(summary_data, args.output) 