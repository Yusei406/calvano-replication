"""
Convergence results analysis module for Phase 2.

Implements statistical analysis of Q-learning convergence.
Based on the paper's convergence analysis methodology.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

# Handle imports for both package and standalone usage
try:
    from ..dtype_policy import DTYPE, array, zeros
except ImportError:
    try:
        from dtype_policy import DTYPE, array, zeros
    except ImportError:
        # Fallback for testing
        import numpy as np
        DTYPE = np.float64
        array = lambda x: np.array(x, dtype=DTYPE)
        zeros = lambda n: np.zeros(n, dtype=DTYPE)


@dataclass
class ConvergenceStats:
    """Container for convergence analysis results."""
    n_runs: int
    conv_rate: float
    mean_price: np.ndarray
    std_price: np.ndarray
    mean_profit: np.ndarray
    std_profit: np.ndarray
    nash_gap: float
    coop_gap: float
    conv_time: Optional[float] = None
    volatility: Optional[float] = None


def aggregate_runs(run_logs: List[Dict]) -> ConvergenceStats:
    """
    Aggregate statistics from multiple simulation runs.
    
    Args:
        run_logs: List of run result dictionaries
        
    Returns:
        ConvergenceStats object with aggregated results
    """
    if not run_logs:
        return ConvergenceStats(
            n_runs=0, conv_rate=0.0, mean_price=array([]), 
            std_price=array([]), mean_profit=array([]), 
            std_profit=array([]), nash_gap=float('inf'), 
            coop_gap=float('inf')
        )
    
    # Extract convergence flags
    convergence_flags = [
        log.get('overall_converged', False) for log in run_logs
    ]
    n_runs = len(run_logs)
    conv_rate = sum(convergence_flags) / n_runs

    # Extract final prices and profits
    final_prices = []
    final_profits = []
    nash_distances = []
    coop_distances = []
    convergence_times = []
    volatilities = []

    for log in run_logs:
        if 'final_prices' in log:
            final_prices.append(log['final_prices'])
        
        if 'final_profits' in log:
            final_profits.append(log['final_profits'])
        
        if 'nash_distance' in log:
            nash_distances.append(log['nash_distance'])
        
        if 'coop_distance' in log:
            coop_distances.append(log['coop_distance'])
        
        if 'convergence_time' in log and log['convergence_time'] is not None:
            convergence_times.append(log['convergence_time'])
        
        if 'final_volatility' in log:
            volatilities.append(log['final_volatility'])

    # Calculate statistics
    if final_prices:
        # Convert to numpy array if list of arrays
        if isinstance(final_prices[0], (list, np.ndarray)):
            final_prices = np.array(final_prices)
            mean_price = array(np.mean(final_prices, axis=0))
            std_price = array(np.std(final_prices, axis=0))
        else:
            # Handle single values
            mean_price = array([np.mean(final_prices)])
            std_price = array([np.std(final_prices)])
    else:
        mean_price = array([])
        std_price = array([])

    if final_profits:
        # Convert to numpy array if list of arrays  
        if isinstance(final_profits[0], (list, np.ndarray)):
            final_profits = np.array(final_profits)
            mean_profit = array(np.mean(final_profits, axis=0))
            std_profit = array(np.std(final_profits, axis=0))
        else:
            # Handle single values
            mean_profit = array([np.mean(final_profits)])
            std_profit = array([np.std(final_profits)])
    else:
        mean_profit = array([])
        std_profit = array([])

    # Average distances
    nash_gap = np.mean(nash_distances) if nash_distances else float('inf')
    coop_gap = np.mean(coop_distances) if coop_distances else float('inf')

    # Average convergence time and volatility
    avg_conv_time = np.mean(convergence_times) if convergence_times else None
    avg_volatility = np.mean(volatilities) if volatilities else None

    return ConvergenceStats(
        n_runs=n_runs,
        conv_rate=conv_rate,
        mean_price=mean_price,
        std_price=std_price,
        mean_profit=mean_profit,
        std_profit=std_profit,
        nash_gap=nash_gap,
        coop_gap=coop_gap,
        conv_time=avg_conv_time,
        volatility=avg_volatility
    )


def to_dataframe(stats: ConvergenceStats, experiment_name: str) -> pd.DataFrame:
    """
    Convert ConvergenceStats to pandas DataFrame for tabular output.
    
    Args:
        stats: ConvergenceStats object
        experiment_name: Name of the experiment
        
    Returns:
        DataFrame with convergence results
    """
    data = {
        'Experiment': experiment_name,
        'N_Runs': stats.n_runs,
        'Convergence_Rate': f"{stats.conv_rate:.3f}",
        'Nash_Gap': f"{stats.nash_gap:.4f}",
        'Coop_Gap': f"{stats.coop_gap:.4f}"
    }
    
    # Add agent-specific price and profit columns
    for i, price in enumerate(stats.mean_price):
        data[f'Mean_Price_Agent_{i+1}'] = f"{price:.4f}"
    
    for i, price_std in enumerate(stats.std_price):
        data[f'Std_Price_Agent_{i+1}'] = f"{price_std:.4f}"
    
    for i, profit in enumerate(stats.mean_profit):
        data[f'Mean_Profit_Agent_{i+1}'] = f"{profit:.4f}"
    
    for i, profit_std in enumerate(stats.std_profit):
        data[f'Std_Profit_Agent_{i+1}'] = f"{profit_std:.4f}"
    
    # Add optional fields
    if stats.conv_time is not None:
        data['Mean_Conv_Time'] = f"{stats.conv_time:.1f}"
    
    if stats.volatility is not None:
        data['Price_Volatility'] = f"{stats.volatility:.4f}"
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    return df


def compare_experiments(stats_dict: Dict[str, ConvergenceStats]) -> pd.DataFrame:
    """
    Compare convergence results across multiple experiments.
    
    Args:
        stats_dict: Dictionary mapping experiment names to ConvergenceStats
        
    Returns:
        DataFrame comparing all experiments
    """
    dfs = []
    
    for exp_name, stats in stats_dict.items():
        df = to_dataframe(stats, exp_name)
        dfs.append(df)
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
    else:
        combined_df = pd.DataFrame()
    
    return combined_df 