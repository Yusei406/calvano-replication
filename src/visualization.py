import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import pandas as pd

def plot_learning_trajectories(results: Dict[str, Any], save_path: str = None):
    """
    Plot learning trajectories for profit gains, incentive compatibility, and incentive ratios.
    
    Args:
        results: Dictionary containing simulation results
        save_path: Path to save the plot (optional)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot profit gains
    ax1.plot(results['profit_gains'].mean(axis=0), label='Mean')
    ax1.fill_between(
        range(len(results['profit_gains'].mean(axis=0))),
        results['profit_gains'].mean(axis=0) - results['profit_gains'].std(axis=0),
        results['profit_gains'].mean(axis=0) + results['profit_gains'].std(axis=0),
        alpha=0.2
    )
    ax1.set_title('Profit Gains')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Profit Gain')
    ax1.legend()
    
    # Plot incentive compatibility
    ax2.plot(results['incentive_compatibility'].mean(axis=0), label='Mean')
    ax2.fill_between(
        range(len(results['incentive_compatibility'].mean(axis=0))),
        results['incentive_compatibility'].mean(axis=0) - results['incentive_compatibility'].std(axis=0),
        results['incentive_compatibility'].mean(axis=0) + results['incentive_compatibility'].std(axis=0),
        alpha=0.2
    )
    ax2.set_title('Incentive Compatibility')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Incentive Compatibility')
    ax2.legend()
    
    # Plot incentive ratios
    ax3.plot(results['incentive_ratios'].mean(axis=0), label='Mean')
    ax3.fill_between(
        range(len(results['incentive_ratios'].mean(axis=0))),
        results['incentive_ratios'].mean(axis=0) - results['incentive_ratios'].std(axis=0),
        results['incentive_ratios'].mean(axis=0) + results['incentive_ratios'].std(axis=0),
        alpha=0.2
    )
    ax3.set_title('Incentive Ratios')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Incentive Ratio')
    ax3.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_equilibrium_check(results: Dict[str, Any], save_path: str = None):
    """
    Plot equilibrium check results.
    
    Args:
        results: Dictionary containing equilibrium check results
        save_path: Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot best response frequencies
    ax1.bar(range(len(results['best_response_freq'])), results['best_response_freq'])
    ax1.set_title('Best Response Frequencies')
    ax1.set_xlabel('Agent')
    ax1.set_ylabel('Frequency')
    
    # Plot equilibrium frequencies
    ax2.bar(range(len(results['equilibrium_freq'])), results['equilibrium_freq'])
    ax2.set_title('Equilibrium Frequencies')
    ax2.set_xlabel('State Type')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_q_gaps(results: Dict[str, Any], save_path: str = None):
    """
    Plot Q gap results.
    
    Args:
        results: Dictionary containing Q gap results
        save_path: Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Q gaps by state type
    ax1.boxplot([
        results['q_gaps_on_path'],
        results['q_gaps_off_path'],
        results['q_gaps_best_responding'],
        results['q_gaps_not_best_responding']
    ])
    ax1.set_title('Q Gaps by State Type')
    ax1.set_xticklabels(['On Path', 'Off Path', 'Best Responding', 'Not Best Responding'])
    ax1.set_ylabel('Q Gap')
    
    # Plot Q gaps by agent
    ax2.boxplot([results['q_gaps_by_agent'][i] for i in range(len(results['q_gaps_by_agent']))])
    ax2.set_title('Q Gaps by Agent')
    ax2.set_xlabel('Agent')
    ax2.set_ylabel('Q Gap')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_impulse_response(results: Dict[str, Any], save_path: str = None):
    """
    Plot impulse response results.
    
    Args:
        results: Dictionary containing impulse response results
        save_path: Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot price responses
    ax1.plot(results['price_responses'].mean(axis=0), label='Mean')
    ax1.fill_between(
        range(len(results['price_responses'].mean(axis=0))),
        results['price_responses'].mean(axis=0) - results['price_responses'].std(axis=0),
        results['price_responses'].mean(axis=0) + results['price_responses'].std(axis=0),
        alpha=0.2
    )
    ax1.set_title('Price Responses')
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Price')
    ax1.legend()
    
    # Plot profit responses
    ax2.plot(results['profit_responses'].mean(axis=0), label='Mean')
    ax2.fill_between(
        range(len(results['profit_responses'].mean(axis=0))),
        results['profit_responses'].mean(axis=0) - results['profit_responses'].std(axis=0),
        results['profit_responses'].mean(axis=0) + results['profit_responses'].std(axis=0),
        alpha=0.2
    )
    ax2.set_title('Profit Responses')
    ax2.set_xlabel('Period')
    ax2.set_ylabel('Profit')
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def create_summary_table(results: Dict[str, Any], save_path: str = None) -> pd.DataFrame:
    """
    Create a summary table of the simulation results.
    
    Args:
        results: Dictionary containing simulation results
        save_path: Path to save the table (optional)
        
    Returns:
        DataFrame containing the summary statistics
    """
    summary = pd.DataFrame({
        'Metric': [
            'Average Profit Gain',
            'Incentive Compatibility',
            'Incentive Ratio',
            'Best Response Frequency',
            'Equilibrium Frequency',
            'Average Q Gap',
            'Price Response Magnitude',
            'Profit Response Magnitude'
        ],
        'Mean': [
            results['profit_gains'].mean(),
            results['incentive_compatibility'].mean(),
            results['incentive_ratios'].mean(),
            results['best_response_freq'].mean(),
            results['equilibrium_freq'].mean(),
            results['q_gaps'].mean(),
            results['price_responses'].mean(),
            results['profit_responses'].mean()
        ],
        'Std': [
            results['profit_gains'].std(),
            results['incentive_compatibility'].std(),
            results['incentive_ratios'].std(),
            results['best_response_freq'].std(),
            results['equilibrium_freq'].std(),
            results['q_gaps'].std(),
            results['price_responses'].std(),
            results['profit_responses'].std()
        ]
    })
    
    if save_path:
        summary.to_csv(save_path, index=False)
    
    return summary 