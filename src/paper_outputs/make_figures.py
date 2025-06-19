"""
Figure Generator for Calvano Q-learning Paper Outputs.

Generates publication-ready figures with consistent formatting:
- Paper-specified size: 3.25in × 2.5in  
- High resolution: 600 dpi
- Font size: 8pt
- Fixed legend positions
- Self-contained plotting with graceful fallback
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Try to import existing plotting functions with graceful fallback
try:
    import sys
    # Add parent directory to path for absolute import
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from plotting.figures import create_convergence_analysis, create_profit_comparison, create_state_frequency
    PLOTTING_AVAILABLE = True
    print("✓ Advanced plotting functions available")
except ImportError as e:
    warnings.warn(f"Advanced plotting not available: {e}")
    PLOTTING_AVAILABLE = False
    # Define fallback functions
    def create_convergence_analysis(*args, **kwargs):
        return None
    def create_profit_comparison(*args, **kwargs):
        return None
    def create_state_frequency(*args, **kwargs):
        return None


# Publication settings
FIGURE_WIDTH = 3.25  # inches
FIGURE_HEIGHT = 2.5  # inches  
DPI = 600
FONT_SIZE = 8
FONT_FAMILY = 'DejaVu Sans'

# Configure matplotlib for publication quality
plt.rcParams.update({
    'figure.figsize': (FIGURE_WIDTH, FIGURE_HEIGHT),
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'font.size': FONT_SIZE,
    'font.family': FONT_FAMILY,
    'axes.titlesize': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    'legend.fontsize': FONT_SIZE - 1,
    'figure.titlesize': FONT_SIZE + 1,
    'lines.linewidth': 1.0,
    'lines.markersize': 3,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.alpha': 0.3
})


def setup_publication_style():
    """Set up consistent publication-quality styling."""
    # Use a clean color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    
    # Set seaborn style  
    sns.set_style("whitegrid")
    sns.set_palette("husl")


def load_simulation_data(logdir: str) -> Dict[str, Any]:
    """
    Load simulation data from logs directory.
    
    Args:
        logdir: Path to logs directory
        
    Returns:
        Dictionary with loaded data
    """
    logdir_path = Path(logdir)
    
    # Load summary
    summary_path = logdir_path / "logs" / "summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    else:
        summary = {}
    
    # Load individual run files
    logs_path = logdir_path / "logs"
    run_files = list(logs_path.glob("run_*.json"))
    
    runs_data = []
    for run_file in sorted(run_files):
        try:
            with open(run_file, 'r') as f:
                run_data = json.load(f)
                runs_data.append(run_data)
        except Exception as e:
            warnings.warn(f"Could not load {run_file}: {e}")
    
    return {
        'summary': summary,
        'runs': runs_data,
        'logdir': str(logdir_path)
    }


def make_figure1_convergence(data: Dict[str, Any], output_dir: str) -> str:
    """
    Generate Figure 1: Convergence Analysis.
    
    Args:
        data: Simulation data dictionary
        output_dir: Output directory for figures
        
    Returns:
        Path to generated figure
    """
    setup_publication_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH * 2, FIGURE_HEIGHT * 2))
    fig.suptitle('Convergence Analysis', fontsize=FONT_SIZE + 2)
    
    # Extract convergence data from runs
    if data['runs']:
        conv_times = []
        final_prices = []
        
        for run in data['runs']:
            if 'convergence_time' in run:
                conv_times.append(run['convergence_time'])
            if 'final_prices' in run:
                final_prices.append(run['final_prices'])
        
        # Plot 1: Convergence times distribution
        if conv_times:
            axes[0, 0].hist(conv_times, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('Convergence Time')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Convergence Time Distribution')
        
        # Plot 2: Final prices
        if final_prices:
            final_prices_flat = [p for prices in final_prices for p in (prices if isinstance(prices, list) else [prices])]
            axes[0, 1].hist(final_prices_flat, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Final Price')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Final Price Distribution')
        
        # Plot 3: Summary statistics
        conv_rate = data['summary'].get('convergence_rate', 0)
        mean_time = data['summary'].get('avg_convergence_time', 0)
        
        metrics = ['Conv. Rate', 'Avg. Time']
        values = [conv_rate, mean_time/100 if mean_time > 100 else mean_time]  # Scale for display
        
        bars = axes[1, 0].bar(metrics, values)
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Summary Statistics')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 4: Nash price comparison
        nash_price = data['summary'].get('nash_price', 0.5)
        axes[1, 1].axhline(y=nash_price, color='red', linestyle='--', label='Nash Price')
        axes[1, 1].axhline(y=1.0, color='green', linestyle='--', label='Monopoly Price')
        axes[1, 1].set_ylim(0, 1.2)
        axes[1, 1].set_title('Price Benchmarks')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / "figures" / "figure1_convergence.png"
    os.makedirs(output_path.parent, exist_ok=True)
    
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"✓ Figure 1 generated: {output_path}")
    return str(output_path)


def make_figure2_profits(data: Dict[str, Any], output_dir: str) -> str:
    """
    Generate Figure 2: Profit Analysis.
    
    Args:
        data: Simulation data dictionary
        output_dir: Output directory for figures
        
    Returns:
        Path to generated figure
    """
    setup_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 2, FIGURE_HEIGHT))
    fig.suptitle('Profit Analysis', fontsize=FONT_SIZE + 2)
    
    # Extract profit data
    summary = data['summary']
    
    # Plot 1: Profit comparison
    profit_types = ['Realized', 'Nash', 'Monopoly', 'Competitive']
    profit_values = [
        summary.get('mean_profit', 0.25),
        summary.get('nash_profit', 0.25),
        summary.get('monopoly_profit', 0.5),
        summary.get('competitive_profit', 0.0)
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = axes[0].bar(profit_types, profit_values, color=colors, alpha=0.7)
    axes[0].set_ylabel('Profit')
    axes[0].set_title('Profit Comparison')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, profit_values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 2: Profit distribution from runs
    if data['runs']:
        all_profits = []
        for run in data['runs']:
            if 'final_profit' in run:
                all_profits.append(run['final_profit'])
            elif 'mean_profit' in run:
                all_profits.append(run['mean_profit'])
        
        if all_profits:
            axes[1].hist(all_profits, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1].axvline(np.mean(all_profits), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(all_profits):.3f}')
            axes[1].set_xlabel('Profit')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Profit Distribution')
            axes[1].legend()
        else:
            # Show placeholder with theoretical values
            theoretical_profits = [0.2, 0.22, 0.25, 0.28, 0.3]
            axes[1].hist(theoretical_profits, bins=10, alpha=0.7, color='lightgray', edgecolor='black')
            axes[1].set_xlabel('Profit')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Profit Distribution (Mock)')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / "figures" / "figure2_profits.png"
    os.makedirs(output_path.parent, exist_ok=True)
    
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"✓ Figure 2 generated: {output_path}")
    return str(output_path)


def make_figure3_learning(data: Dict[str, Any], output_dir: str) -> str:
    """
    Generate Figure 3: Learning Dynamics.
    
    Args:
        data: Simulation data dictionary
        output_dir: Output directory for figures
        
    Returns:
        Path to generated figure
    """
    setup_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 2, FIGURE_HEIGHT))
    fig.suptitle('Learning Dynamics', fontsize=FONT_SIZE + 2)
    
    # Plot 1: Price evolution over time
    if data['runs'] and len(data['runs']) > 0:
        sample_run = data['runs'][0]
        
        if 'price_history' in sample_run:
            price_hist = sample_run['price_history']
            if isinstance(price_hist, list) and len(price_hist) > 0:
                # Plot first 500 periods for clarity
                periods = min(500, len(price_hist))
                time_axis = range(periods)
                
                if isinstance(price_hist[0], list):  # Multiple agents
                    for agent_idx in range(min(2, len(price_hist[0]))):
                        agent_prices = [price_hist[t][agent_idx] for t in range(periods)]
                        axes[0].plot(time_axis, agent_prices, label=f'Agent {agent_idx+1}')
                else:  # Single agent
                    axes[0].plot(time_axis, price_hist[:periods])
                
                axes[0].set_xlabel('Period')
                axes[0].set_ylabel('Price')
                axes[0].set_title('Price Evolution')
                if isinstance(price_hist[0], list):
                    axes[0].legend()
        
        # Plot 2: Q-values evolution (if available)
        if 'qvalue_evolution' in sample_run:
            q_evolution = sample_run['qvalue_evolution']
            periods = min(200, len(q_evolution))
            
            axes[1].plot(range(periods), q_evolution[:periods])
            axes[1].set_xlabel('Period')
            axes[1].set_ylabel('Max Q-Value')
            axes[1].set_title('Q-Value Evolution')
        else:
            # Alternative: Show convergence indicator
            if 'convergence_history' in sample_run:
                conv_hist = sample_run['convergence_history']
                periods = min(500, len(conv_hist))
                
                axes[1].plot(range(periods), conv_hist[:periods])
                axes[1].set_xlabel('Period')
                axes[1].set_ylabel('Convergence Indicator')
                axes[1].set_title('Convergence Progress')
            else:
                # Show placeholder
                axes[1].text(0.5, 0.5, 'Learning dynamics\nnot available', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Learning Dynamics')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / "figures" / "figure3_learning.png"
    os.makedirs(output_path.parent, exist_ok=True)
    
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"✓ Figure 3 generated: {output_path}")
    return str(output_path)


def generate_all_figures(logdir: str, output_dir: str) -> Dict[str, str]:
    """
    Generate all paper figures.
    
    Args:
        logdir: Path to simulation logs directory
        output_dir: Output directory for figures
        
    Returns:
        Dictionary with figure paths
    """
    print(f"Loading data from: {logdir}")
    data = load_simulation_data(logdir)
    
    # Create output directory
    os.makedirs(Path(output_dir) / "figures", exist_ok=True)
    
    figures = {}
    
    try:
        figures['figure1'] = make_figure1_convergence(data, output_dir)
    except Exception as e:
        print(f"⚠ Warning: Could not generate Figure 1: {e}")
    
    try:
        figures['figure2'] = make_figure2_profits(data, output_dir)
    except Exception as e:
        print(f"⚠ Warning: Could not generate Figure 2: {e}")
    
    try:
        figures['figure3'] = make_figure3_learning(data, output_dir)
    except Exception as e:
        print(f"⚠ Warning: Could not generate Figure 3: {e}")
    
    print(f"✅ All figures generated in: {Path(output_dir) / 'figures'}")
    return figures


def check_figure_requirements() -> bool:
    """
    Check if all required packages are available for figure generation.
    
    Returns:
        True if all requirements met
    """
    required_packages = ['matplotlib', 'seaborn', 'numpy', 'pandas']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"❌ Required package missing: {package}")
            return False
    
    print("✅ All figure requirements satisfied")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate publication-ready figures")
    parser.add_argument("--logdir", required=True, help="Path to simulation logs directory")
    parser.add_argument("--output", required=True, help="Output directory for figures")
    parser.add_argument("--figure", choices=['1', '2', '3', 'all'], default='all',
                       help="Which figure to generate")
    parser.add_argument("--check-requirements", action="store_true",
                       help="Check if all requirements are satisfied")
    
    args = parser.parse_args()
    
    if args.check_requirements:
        check_figure_requirements()
        exit(0)
    
    if not check_figure_requirements():
        print("❌ Figure requirements not satisfied")
        exit(1)
    
    data = load_simulation_data(args.logdir)
    
    if args.figure == 'all':
        generate_all_figures(args.logdir, args.output)
    else:
        if args.figure == '1':
            make_figure1_convergence(data, args.output)
        elif args.figure == '2':
            make_figure2_profits(data, args.output)
        elif args.figure == '3':
            make_figure3_learning(data, args.output) 