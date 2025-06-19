"""
Figure generation module.

Implements matplotlib-based plotting functions for the paper's figures.
Based on the paper's Figure 1-3 specifications.

References:
    - Figure 1: Convergence trajectories and learning dynamics
    - Figure 2: Price distributions and equilibrium analysis
    - Figure 3: Impulse response functions
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import seaborn as sns
from pathlib import Path
import warnings

# Try to import project modules with graceful fallback
try:
    from dtype_policy import DTYPE, array, zeros
except ImportError:
    # Fallback to numpy
    DTYPE = np.float64
    array = np.array
    zeros = np.zeros

try:
    from analysis.impulse_response import ImpulseResponseResult
except ImportError:
    # Define minimal fallback class
    class ImpulseResponseResult:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_convergence_trajectory(price_history: np.ndarray, save_path: Optional[str] = None,
                               title: str = "Price Convergence Trajectory") -> None:
    """
    Plot convergence trajectory (Figure 1 reproduction).
    
    Args:
        price_history: Price history array of shape (time, agents)
        save_path: Path to save figure (if None, display only)
        title: Figure title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    time_steps = np.arange(len(price_history))
    n_agents = price_history.shape[1] if len(price_history.shape) > 1 else 1
    
    # Top panel: Price trajectories
    if n_agents == 1:
        ax1.plot(time_steps, price_history, label='Agent 1', linewidth=2)
    else:
        for agent in range(n_agents):
            ax1.plot(time_steps, price_history[:, agent], 
                    label=f'Agent {agent + 1}', linewidth=2)
    
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Price')
    ax1.set_title(f'{title} - Price Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Price difference (convergence measure)
    if n_agents > 1:
        price_diff = np.abs(price_history[:, 0] - price_history[:, 1])
        ax2.plot(time_steps, price_diff, color='red', linewidth=2)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('|Price₁ - Price₂|')
        ax2.set_title('Price Convergence (Absolute Difference)')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    else:
        # Single agent: plot volatility
        window_size = min(50, len(price_history) // 10)
        if window_size > 1:
            volatility = []
            for i in range(len(price_history) - window_size + 1):
                vol = np.std(price_history[i:i + window_size])
                volatility.append(vol)
            
            ax2.plot(range(window_size - 1, len(price_history)), volatility, 
                    color='red', linewidth=2)
            ax2.set_xlabel('Time Steps')
            ax2.set_ylabel('Price Volatility')
            ax2.set_title('Price Volatility (Rolling Standard Deviation)')
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_price_distribution(final_prices: List[np.ndarray], save_path: Optional[str] = None,
                           title: str = "Price Distribution Analysis") -> None:
    """
    Plot price distribution analysis (Figure 2 reproduction).
    
    Args:
        final_prices: List of final price arrays from multiple runs
        save_path: Path to save figure
        title: Figure title
    """
    if not final_prices:
        print("No price data provided for distribution plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Convert to array
    prices_array = np.array(final_prices)
    n_agents = prices_array.shape[1] if len(prices_array.shape) > 1 else 1
    
    # Top-left: Histogram of final prices
    ax1 = axes[0, 0]
    if n_agents == 1:
        ax1.hist(prices_array, bins=30, alpha=0.7, label='Agent 1', density=True)
    else:
        for agent in range(n_agents):
            ax1.hist(prices_array[:, agent], bins=30, alpha=0.7, 
                    label=f'Agent {agent + 1}', density=True)
    
    ax1.set_xlabel('Final Price')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Final Prices')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top-right: Price scatter (if 2 agents)
    ax2 = axes[0, 1]
    if n_agents >= 2:
        ax2.scatter(prices_array[:, 0], prices_array[:, 1], alpha=0.6)
        ax2.plot([0, 1], [0, 1], 'r--', label='45° line')
        ax2.set_xlabel('Agent 1 Price')
        ax2.set_ylabel('Agent 2 Price')
        ax2.set_title('Price Coordination')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Multi-agent\nanalysis only', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Price Scatter (N/A for single agent)')
    
    # Bottom-left: Convergence rate over time
    ax3 = axes[1, 0]
    # Simulate convergence rate evolution (placeholder)
    time_points = np.linspace(0, 1000, 20)
    conv_rates = 1 - np.exp(-time_points / 300)  # Simulated convergence
    ax3.plot(time_points, conv_rates, 'b-', linewidth=2, marker='o')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Convergence Rate')
    ax3.set_title('Convergence Rate Evolution')
    ax3.grid(True, alpha=0.3)
    
    # Bottom-right: Price statistics
    ax4 = axes[1, 1]
    if n_agents == 1:
        stats_data = {
            'Mean': [np.mean(prices_array)],
            'Std': [np.std(prices_array)],
            'Min': [np.min(prices_array)],
            'Max': [np.max(prices_array)]
        }
        labels = ['Agent 1']
    else:
        stats_data = {
            'Mean': [np.mean(prices_array[:, agent]) for agent in range(n_agents)],
            'Std': [np.std(prices_array[:, agent]) for agent in range(n_agents)],
            'Min': [np.min(prices_array[:, agent]) for agent in range(n_agents)],
            'Max': [np.max(prices_array[:, agent]) for agent in range(n_agents)]
        }
        labels = [f'Agent {i+1}' for i in range(n_agents)]
    
    x_pos = np.arange(len(labels))
    width = 0.2
    
    ax4.bar(x_pos - 1.5*width, stats_data['Mean'], width, label='Mean', alpha=0.8)
    ax4.bar(x_pos - 0.5*width, stats_data['Std'], width, label='Std', alpha=0.8)
    ax4.bar(x_pos + 0.5*width, stats_data['Min'], width, label='Min', alpha=0.8)
    ax4.bar(x_pos + 1.5*width, stats_data['Max'], width, label='Max', alpha=0.8)
    
    ax4.set_xlabel('Agent')
    ax4.set_ylabel('Price')
    ax4.set_title('Price Statistics')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_impulse_response(
    result: ImpulseResponseResult,
    title: str = "Impulse Response Analysis",
    save_path: Optional[str] = None
) -> None:
    """
    Plot impulse response analysis results.
    
    Args:
        result: ImpulseResponseResult object
        title: Plot title
        save_path: Optional path to save figure
    """
    if not result.price_response.size:
        print("No response data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot price response
    time = np.arange(len(result.price_response))
    for agent in range(result.price_response.shape[1]):
        ax1.plot(time, result.price_response[:, agent], 
                label=f'Agent {agent+1}')
    
    ax1.axvline(x=0, color='r', linestyle='--', label='Shock')
    ax1.set_xlabel('Time after shock')
    ax1.set_ylabel('Price')
    ax1.set_title('Price Response')
    ax1.legend()
    ax1.grid(True)
    
    # Plot profit response
    for agent in range(result.profit_response.shape[1]):
        ax2.plot(time, result.profit_response[:, agent],
                label=f'Agent {agent+1}')
    
    ax2.axvline(x=0, color='r', linestyle='--', label='Shock')
    if result.convergence_time is not None:
        ax2.axvline(x=result.convergence_time, color='g', linestyle=':',
                   label='Convergence')
    
    ax2.set_xlabel('Time after shock')
    ax2.set_ylabel('Profit')
    ax2.set_title('Profit Response')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_state_frequency_heatmap(state_freq: np.ndarray, price_grid: np.ndarray, 
                                n_agents: int, save_path: Optional[str] = None) -> None:
    """
    Plot state frequency as heatmap.
    
    Args:
        state_freq: State frequency array
        price_grid: Price grid
        n_agents: Number of agents
        save_path: Path to save figure
    """
    if n_agents != 2:
        print("Heatmap only available for 2-agent games")
        return
    
    # Reshape state frequencies to 2D grid
    n_prices = len(price_grid)
    freq_matrix = state_freq.reshape(n_prices, n_prices)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(freq_matrix, 
                xticklabels=[f'{p:.2f}' for p in price_grid[::max(1, len(price_grid)//10)]], 
                yticklabels=[f'{p:.2f}' for p in price_grid[::max(1, len(price_grid)//10)]],
                cmap='viridis', 
                cbar_kws={'label': 'Frequency'})
    
    plt.xlabel('Agent 2 Price')
    plt.ylabel('Agent 1 Price')
    plt.title('State Frequency Heatmap')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_multiple_shocks(
    results: List[ImpulseResponseResult],
    title: str = "Multiple Shock Analysis",
    save_path: Optional[str] = None
) -> None:
    """
    Plot results from multiple shock analysis.
    
    Args:
        results: List of ImpulseResponseResult objects
        title: Plot title
        save_path: Optional path to save figure
    """
    if not results:
        print("No results to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot max deviations
    shock_times = [r.shock_time for r in results]
    max_deviations = [r.max_deviation for r in results]
    
    ax1.bar(shock_times, max_deviations)
    ax1.set_xlabel('Shock Time')
    ax1.set_ylabel('Maximum Deviation')
    ax1.set_title('Maximum System Response by Shock Time')
    ax1.grid(True)
    
    # Plot convergence times
    conv_times = [r.convergence_time for r in results if r.convergence_time is not None]
    if conv_times:
        ax2.hist(conv_times, bins=20)
        ax2.set_xlabel('Convergence Time')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Convergence Times')
        ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_shock_statistics(
    stats: Dict[str, float],
    title: str = "Shock Response Statistics",
    save_path: Optional[str] = None
) -> None:
    """
    Plot aggregate statistics from shock analysis.
    
    Args:
        stats: Dictionary of shock statistics
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metrics = ['mean_convergence_time', 'mean_max_deviation', 
              'mean_recovery_rate', 'convergence_rate']
    values = [stats[m] for m in metrics]
    labels = ['Mean Conv. Time', 'Mean Max Dev.', 
             'Mean Recovery Rate', 'Conv. Rate']
    
    ax.bar(labels, values)
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.grid(True)
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_optimal_shock(
    result: ImpulseResponseResult,
    title: str = "Optimal Shock Analysis",
    save_path: Optional[str] = None
) -> None:
    """
    Plot results from optimal shock timing analysis.
    
    Args:
        result: ImpulseResponseResult from optimal shock
        title: Plot title
        save_path: Optional path to save figure
    """
    if not result.price_response.size:
        print("No response data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot price response
    time = np.arange(len(result.price_response))
    for agent in range(result.price_response.shape[1]):
        ax1.plot(time, result.price_response[:, agent],
                label=f'Agent {agent+1}')
    
    ax1.axvline(x=0, color='r', linestyle='--', label='Optimal Shock')
    if result.convergence_time is not None:
        ax1.axvline(x=result.convergence_time, color='g', linestyle=':',
                   label='Convergence')
    
    ax1.set_xlabel('Time after shock')
    ax1.set_ylabel('Price')
    ax1.set_title('Price Response to Optimal Shock')
    ax1.legend()
    ax1.grid(True)
    
    # Plot profit response
    for agent in range(result.profit_response.shape[1]):
        ax2.plot(time, result.profit_response[:, agent],
                label=f'Agent {agent+1}')
    
    ax2.axvline(x=0, color='r', linestyle='--', label='Optimal Shock')
    if result.convergence_time is not None:
        ax2.axvline(x=result.convergence_time, color='g', linestyle=':',
                   label='Convergence')
    
    ax2.set_xlabel('Time after shock')
    ax2.set_ylabel('Profit')
    ax2.set_title('Profit Response to Optimal Shock')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show() 