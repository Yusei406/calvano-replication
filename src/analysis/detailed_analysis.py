"""
Detailed analysis module for Phase 2.

Provides comprehensive agent-specific statistics including convergence speeds,
state frequencies, and profit distributions. Generates detailed CSV tables
and visualization figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import os
import json

# Handle imports for both package and standalone usage
try:
    from ..dtype_policy import DTYPE, array, zeros
    from ..params import SimParams
    from ..convergence import calculate_nash_equilibrium, calculate_cooperative_equilibrium
    from .convergence_results import ConvergenceStats
    from .profit_gain import calculate_profits, calculate_gains
    from .state_frequency import StateFrequencyAnalysis
    from .learning_trajectory import TrajectoryResults
    from .qgap_to_maximum import compute_qgap_statistics
except ImportError:
    try:
        from dtype_policy import DTYPE, array, zeros
        from params import SimParams
        from convergence import calculate_nash_equilibrium, calculate_cooperative_equilibrium
        from convergence_results import ConvergenceStats
        from profit_gain import calculate_profits, calculate_gains
        from state_frequency import StateFrequencyAnalysis
        from learning_trajectory import TrajectoryResults
        from qgap_to_maximum import compute_qgap_statistics
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
        
        def calculate_nash_equilibrium(params):
            return np.array([1.0, 1.0])
        
        def calculate_cooperative_equilibrium(params):
            return np.array([1.2, 1.2])
        
        class ConvergenceStats:
            def __init__(self):
                self.convergence_rate = 0.95
                self.nash_gap = 0.01
                self.cooperative_gap = 0.05
                self.final_prices = np.array([1.0, 1.0])
        
        class StateFrequencyAnalysis:
            def __init__(self, prices, params):
                self.frequencies = np.random.rand(10, 10)
                self.cycles = []
                self.volatility = 0.1
        
        class TrajectoryResults:
            def __init__(self):
                self.seeds = [1, 2, 3]
                self.convergence_times = [500, 450, 600]
                self.final_prices = [np.array([1.0, 1.0]), np.array([1.1, 1.0]), np.array([0.9, 1.0])]
        
        def calculate_profits(prices, params):
            return np.random.rand(len(prices), 2)
        
        def calculate_gains(profits, profits_nash, profits_coop):
            return {"nash_gain": 0.1, "coop_gain": -0.05}
        
        def compute_qgap_statistics(Q):
            from types import SimpleNamespace
            return SimpleNamespace(mean_gap=0.01, max_gap=0.05)


@dataclass
class DetailedStats:
    """Container for detailed agent-specific statistics."""
    agent_id: int
    convergence_time: Optional[int]
    convergence_rate: Optional[float]
    final_price: float
    price_volatility: float
    nash_deviation: float
    cooperative_deviation: float
    profit_mean: float
    profit_std: float
    profit_nash_gain: float
    profit_coop_gain: float
    state_entropy: float
    dominant_states: List[Tuple[int, float]]
    qgap_mean: float
    qgap_final: float
    cycle_frequency: float


def analyze_agent_convergence(
    agent_prices: np.ndarray,
    agent_Q_history: Optional[List[np.ndarray]] = None,
    nash_price: float = 1.0,
    cooperative_price: float = 1.2
) -> Dict[str, Any]:
    """
    Analyze convergence properties for a single agent.
    
    Args:
        agent_prices: Price history for the agent
        agent_Q_history: Optional Q-matrix history for the agent
        nash_price: Nash equilibrium price for this agent
        cooperative_price: Cooperative equilibrium price for this agent
        
    Returns:
        Dictionary with convergence analysis results
    """
    if len(agent_prices) == 0:
        return {}
    
    # Basic convergence metrics
    final_price = agent_prices[-1]
    nash_deviation = abs(final_price - nash_price)
    coop_deviation = abs(final_price - cooperative_price)
    
    # Price volatility (rolling standard deviation)
    if len(agent_prices) > 10:
        window_size = min(50, len(agent_prices) // 4)
        rolling_std = pd.Series(agent_prices).rolling(window=window_size).std()
        price_volatility = float(np.nanmean(rolling_std))
    else:
        price_volatility = float(np.std(agent_prices))
    
    # Convergence time estimation
    convergence_time = None
    convergence_rate = None
    
    if len(agent_prices) > 20:
        # Define convergence as staying within 5% of final price
        tolerance = 0.05 * abs(final_price) if final_price != 0 else 0.05
        
        # Look for the last time the price was outside the tolerance band
        final_band = np.abs(agent_prices - final_price) <= tolerance
        
        # Find first sustained convergence (at least 10 consecutive periods)
        sustained_periods = 10
        for t in range(len(agent_prices) - sustained_periods):
            if np.all(final_band[t:t+sustained_periods]):
                convergence_time = t
                break
        
        # Estimate convergence rate
        if convergence_time and convergence_time > 10:
            early_prices = agent_prices[:convergence_time]
            if len(early_prices) > 5:
                # Fit exponential convergence model
                times = np.arange(len(early_prices))
                deviations = np.abs(early_prices - final_price)
                
                # Remove zero deviations to avoid log issues
                nonzero_mask = deviations > 1e-10
                if np.sum(nonzero_mask) > 5:
                    log_deviations = np.log(deviations[nonzero_mask])
                    valid_times = times[nonzero_mask]
                    
                    # Linear fit to log(deviation) vs time
                    coeffs = np.polyfit(valid_times, log_deviations, 1)
                    convergence_rate = -coeffs[0]  # Negative slope indicates convergence
    
    # Q-gap analysis if available
    qgap_mean = np.nan
    qgap_final = np.nan
    
    if agent_Q_history and len(agent_Q_history) > 0:
        try:
            # Analyze final Q-matrix
            final_Q = agent_Q_history[-1]
            if final_Q is not None and final_Q.size > 0:
                qgap_stats = compute_qgap_statistics(final_Q)
                qgap_mean = qgap_stats.mean_gap
                qgap_final = qgap_stats.mean_gap  # Same as mean for final matrix
        except Exception as e:
            print(f"Warning: Q-gap analysis failed: {e}")
    
    return {
        'final_price': final_price,
        'nash_deviation': nash_deviation,
        'cooperative_deviation': coop_deviation,
        'price_volatility': price_volatility,
        'convergence_time': convergence_time,
        'convergence_rate': convergence_rate,
        'qgap_mean': qgap_mean,
        'qgap_final': qgap_final
    }


def analyze_agent_frequencies(
    agent_prices: np.ndarray,
    params: SimParams
) -> Dict[str, Any]:
    """
    Analyze state frequencies and cycles for a single agent.
    
    Args:
        agent_prices: Price history for the agent
        params: Simulation parameters
        
    Returns:
        Dictionary with frequency analysis results
    """
    if len(agent_prices) == 0:
        return {}
    
    try:
        # Use StateFrequencyAnalysis for detailed frequency statistics
        freq_analysis = StateFrequencyAnalysis(
            agent_prices.reshape(-1, 1),  # Reshape for single agent
            params
        )
        
        # Extract key metrics
        frequencies = freq_analysis.frequencies
        
        # Calculate state entropy
        flat_freq = frequencies.flatten()
        nonzero_freq = flat_freq[flat_freq > 0]
        if len(nonzero_freq) > 0:
            state_entropy = -np.sum(nonzero_freq * np.log(nonzero_freq))
        else:
            state_entropy = 0.0
        
        # Find dominant states (top 3)
        state_indices = np.unravel_index(np.argsort(frequencies.flatten())[::-1][:3], frequencies.shape)
        dominant_states = []
        for i in range(len(state_indices[0])):
            state_id = state_indices[0][i] * frequencies.shape[1] + state_indices[1][i]
            frequency = frequencies[state_indices[0][i], state_indices[1][i]]
            dominant_states.append((int(state_id), float(frequency)))
        
        # Cycle frequency
        cycle_frequency = len(freq_analysis.cycles) / max(1, len(agent_prices))
        
        return {
            'state_entropy': state_entropy,
            'dominant_states': dominant_states,
            'cycle_frequency': cycle_frequency,
            'volatility': freq_analysis.volatility
        }
        
    except Exception as e:
        print(f"Warning: Frequency analysis failed: {e}")
        return {
            'state_entropy': 0.0,
            'dominant_states': [],
            'cycle_frequency': 0.0,
            'volatility': 0.0
        }


def analyze_agent_profits(
    agent_prices: np.ndarray,
    params: SimParams,
    nash_prices: np.ndarray,
    cooperative_prices: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze profit statistics for a single agent.
    
    Args:
        agent_prices: Price history for the agent
        params: Simulation parameters
        nash_prices: Nash equilibrium prices
        cooperative_prices: Cooperative equilibrium prices
        
    Returns:
        Dictionary with profit analysis results
    """
    if len(agent_prices) == 0:
        return {}
    
    try:
        # Create price matrix for profit calculation (assume other agent uses Nash)
        if agent_prices.ndim == 1:
            # Single agent case - assume symmetric game
            price_matrix = np.column_stack([agent_prices, np.full_like(agent_prices, nash_prices[0])])
        else:
            price_matrix = agent_prices
        
        # Calculate profits
        profits = calculate_profits(price_matrix, params)
        agent_profits = profits[:, 0]  # First agent's profits
        
        # Calculate benchmark profits
        nash_profit_matrix = np.tile(nash_prices, (len(agent_prices), 1))
        nash_profits = calculate_profits(nash_profit_matrix, params)[:, 0]
        
        coop_profit_matrix = np.tile(cooperative_prices, (len(agent_prices), 1))
        coop_profits = calculate_profits(coop_profit_matrix, params)[:, 0]
        
        # Calculate gains
        gains = calculate_gains(
            agent_profits.reshape(-1, 1),
            nash_profits.reshape(-1, 1),
            coop_profits.reshape(-1, 1)
        )
        
        return {
            'profit_mean': float(np.mean(agent_profits)),
            'profit_std': float(np.std(agent_profits)),
            'profit_nash_gain': gains.get('nash_gain', 0.0),
            'profit_coop_gain': gains.get('coop_gain', 0.0)
        }
        
    except Exception as e:
        print(f"Warning: Profit analysis failed: {e}")
        return {
            'profit_mean': 0.0,
            'profit_std': 0.0,
            'profit_nash_gain': 0.0,
            'profit_coop_gain': 0.0
        }


def create_detailed_statistics(
    results: Dict[str, Any],
    params: SimParams,
    output_dir: str = "."
) -> pd.DataFrame:
    """
    Create comprehensive detailed statistics table for all agents.
    
    Args:
        results: Simulation results dictionary
        params: Simulation parameters
        output_dir: Output directory for saving files
        
    Returns:
        DataFrame with detailed statistics
    """
    # Calculate equilibrium benchmarks
    nash_prices = calculate_nash_equilibrium(params)
    cooperative_prices = calculate_cooperative_equilibrium(params)
    
    # Extract data from results
    price_history = results.get('price_history', np.array([]))
    Q_matrices = results.get('Q_matrices', [])
    Q_history = results.get('Q_history', [])
    
    if price_history.size == 0:
        print("Warning: No price history found in results")
        return pd.DataFrame()
    
    n_agents = min(params.n_agents, price_history.shape[1] if price_history.ndim > 1 else 1)
    
    detailed_stats = []
    
    for agent_id in range(n_agents):
        # Extract agent-specific data
        if price_history.ndim > 1:
            agent_prices = price_history[:, agent_id]
        else:
            agent_prices = price_history
        
        agent_Q_history = []
        if Q_history:
            for Q_t in Q_history:
                if agent_id < len(Q_t):
                    agent_Q_history.append(Q_t[agent_id])
        
        # Analyze convergence
        conv_analysis = analyze_agent_convergence(
            agent_prices,
            agent_Q_history,
            nash_prices[agent_id] if len(nash_prices) > agent_id else nash_prices[0],
            cooperative_prices[agent_id] if len(cooperative_prices) > agent_id else cooperative_prices[0]
        )
        
        # Analyze frequencies
        freq_analysis = analyze_agent_frequencies(agent_prices, params)
        
        # Analyze profits
        profit_analysis = analyze_agent_profits(
            agent_prices,
            params,
            nash_prices,
            cooperative_prices
        )
        
        # Combine all analyses
        agent_stats = DetailedStats(
            agent_id=agent_id + 1,
            convergence_time=conv_analysis.get('convergence_time'),
            convergence_rate=conv_analysis.get('convergence_rate'),
            final_price=conv_analysis.get('final_price', 0.0),
            price_volatility=conv_analysis.get('price_volatility', 0.0),
            nash_deviation=conv_analysis.get('nash_deviation', 0.0),
            cooperative_deviation=conv_analysis.get('cooperative_deviation', 0.0),
            profit_mean=profit_analysis.get('profit_mean', 0.0),
            profit_std=profit_analysis.get('profit_std', 0.0),
            profit_nash_gain=profit_analysis.get('profit_nash_gain', 0.0),
            profit_coop_gain=profit_analysis.get('profit_coop_gain', 0.0),
            state_entropy=freq_analysis.get('state_entropy', 0.0),
            dominant_states=freq_analysis.get('dominant_states', []),
            qgap_mean=conv_analysis.get('qgap_mean', np.nan),
            qgap_final=conv_analysis.get('qgap_final', np.nan),
            cycle_frequency=freq_analysis.get('cycle_frequency', 0.0)
        )
        
        detailed_stats.append(agent_stats)
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'Agent_ID': stats.agent_id,
            'Convergence_Time': stats.convergence_time,
            'Convergence_Rate': f"{stats.convergence_rate:.6f}" if stats.convergence_rate is not None else "N/A",
            'Final_Price': f"{stats.final_price:.6f}",
            'Price_Volatility': f"{stats.price_volatility:.6f}",
            'Nash_Deviation': f"{stats.nash_deviation:.6f}",
            'Cooperative_Deviation': f"{stats.cooperative_deviation:.6f}",
            'Profit_Mean': f"{stats.profit_mean:.6f}",
            'Profit_Std': f"{stats.profit_std:.6f}",
            'Profit_Nash_Gain': f"{stats.profit_nash_gain:.6f}",
            'Profit_Coop_Gain': f"{stats.profit_coop_gain:.6f}",
            'State_Entropy': f"{stats.state_entropy:.6f}",
            'Dominant_State_1': f"{stats.dominant_states[0][0]}({stats.dominant_states[0][1]:.3f})" if len(stats.dominant_states) > 0 else "N/A",
            'Dominant_State_2': f"{stats.dominant_states[1][0]}({stats.dominant_states[1][1]:.3f})" if len(stats.dominant_states) > 1 else "N/A",
            'Dominant_State_3': f"{stats.dominant_states[2][0]}({stats.dominant_states[2][1]:.3f})" if len(stats.dominant_states) > 2 else "N/A",
            'QGap_Mean': f"{stats.qgap_mean:.6f}" if not np.isnan(stats.qgap_mean) else "N/A",
            'QGap_Final': f"{stats.qgap_final:.6f}" if not np.isnan(stats.qgap_final) else "N/A",
            'Cycle_Frequency': f"{stats.cycle_frequency:.6f}"
        }
        for stats in detailed_stats
    ])
    
    # Save to CSV
    os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
    csv_path = os.path.join(output_dir, 'tables', 'detailed_stats.csv')
    df.to_csv(csv_path, index=False)
    print(f"Detailed statistics saved to {csv_path}")
    
    # Create visualization
    create_detailed_statistics_plot(detailed_stats, output_dir)
    
    return df


def create_detailed_statistics_plot(detailed_stats: List[DetailedStats], output_dir: str) -> None:
    """
    Create comprehensive visualization of detailed statistics.
    
    Args:
        detailed_stats: List of DetailedStats objects
        output_dir: Output directory for saving figures
    """
    if not detailed_stats:
        return
    
    # Set up the plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Detailed Agent Statistics', fontsize=16)
    
    # Extract data for plotting
    agent_ids = [stats.agent_id for stats in detailed_stats]
    
    # Plot 1: Convergence Times
    conv_times = [stats.convergence_time if stats.convergence_time is not None else 0 for stats in detailed_stats]
    axes[0, 0].bar(agent_ids, conv_times)
    axes[0, 0].set_title('Convergence Times')
    axes[0, 0].set_xlabel('Agent ID')
    axes[0, 0].set_ylabel('Time Steps')
    
    # Plot 2: Price Deviations
    nash_devs = [stats.nash_deviation for stats in detailed_stats]
    coop_devs = [stats.cooperative_deviation for stats in detailed_stats]
    
    x = np.arange(len(agent_ids))
    width = 0.35
    axes[0, 1].bar(x - width/2, nash_devs, width, label='Nash Deviation')
    axes[0, 1].bar(x + width/2, coop_devs, width, label='Cooperative Deviation')
    axes[0, 1].set_title('Price Deviations from Equilibria')
    axes[0, 1].set_xlabel('Agent ID')
    axes[0, 1].set_ylabel('Absolute Deviation')
    axes[0, 1].legend()
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(agent_ids)
    
    # Plot 3: Profit Statistics
    profit_means = [stats.profit_mean for stats in detailed_stats]
    profit_stds = [stats.profit_std for stats in detailed_stats]
    
    axes[0, 2].errorbar(agent_ids, profit_means, yerr=profit_stds, fmt='o-', capsize=5)
    axes[0, 2].set_title('Profit Statistics (Mean ± Std)')
    axes[0, 2].set_xlabel('Agent ID')
    axes[0, 2].set_ylabel('Profit')
    
    # Plot 4: Q-Gap Analysis
    qgap_means = [stats.qgap_mean if not np.isnan(stats.qgap_mean) else 0 for stats in detailed_stats]
    axes[1, 0].bar(agent_ids, qgap_means)
    axes[1, 0].set_title('Mean Q-Value Gaps')
    axes[1, 0].set_xlabel('Agent ID')
    axes[1, 0].set_ylabel('Q-Gap')
    
    # Plot 5: State Entropy and Cycle Frequency
    entropies = [stats.state_entropy for stats in detailed_stats]
    cycle_freqs = [stats.cycle_frequency for stats in detailed_stats]
    
    ax5_twin = axes[1, 1].twinx()
    line1 = axes[1, 1].plot(agent_ids, entropies, 'bo-', label='State Entropy')
    line2 = ax5_twin.plot(agent_ids, cycle_freqs, 'rs-', label='Cycle Frequency')
    
    axes[1, 1].set_xlabel('Agent ID')
    axes[1, 1].set_ylabel('State Entropy', color='b')
    ax5_twin.set_ylabel('Cycle Frequency', color='r')
    axes[1, 1].set_title('State Dynamics')
    
    # Combine legends
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 6: Profit Gains
    nash_gains = [stats.profit_nash_gain for stats in detailed_stats]
    coop_gains = [stats.profit_coop_gain for stats in detailed_stats]
    
    x = np.arange(len(agent_ids))
    axes[1, 2].bar(x - width/2, nash_gains, width, label='vs Nash')
    axes[1, 2].bar(x + width/2, coop_gains, width, label='vs Cooperative')
    axes[1, 2].set_title('Profit Gains')
    axes[1, 2].set_xlabel('Agent ID')
    axes[1, 2].set_ylabel('Profit Gain')
    axes[1, 2].legend()
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(agent_ids)
    axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    fig_path = os.path.join(output_dir, 'figures', 'detailed_stats.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed statistics plot saved to {fig_path}")


def create_agent_comparison_table(
    results_list: List[Dict[str, Any]],
    params_list: List[SimParams],
    labels: List[str],
    output_dir: str = "."
) -> pd.DataFrame:
    """
    Create comparison table across different simulation runs/configurations.
    
    Args:
        results_list: List of simulation results
        params_list: List of parameter configurations
        labels: Labels for each configuration
        output_dir: Output directory for saving files
        
    Returns:
        Comparison DataFrame
    """
    comparison_data = []
    
    for i, (results, params, label) in enumerate(zip(results_list, params_list, labels)):
        df = create_detailed_statistics(results, params, output_dir)
        
        # Aggregate across agents
        if not df.empty:
            summary_row = {
                'Configuration': label,
                'N_Agents': len(df),
                'Avg_Convergence_Time': df['Convergence_Time'].apply(lambda x: float(x) if x != 'N/A' else np.nan).mean(),
                'Avg_Nash_Deviation': df['Nash_Deviation'].astype(float).mean(),
                'Avg_Coop_Deviation': df['Cooperative_Deviation'].astype(float).mean(),
                'Avg_Profit_Mean': df['Profit_Mean'].astype(float).mean(),
                'Avg_State_Entropy': df['State_Entropy'].astype(float).mean(),
                'Avg_Cycle_Frequency': df['Cycle_Frequency'].astype(float).mean()
            }
            comparison_data.append(summary_row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison table
    os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
    comparison_path = os.path.join(output_dir, 'tables', 'agent_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Agent comparison table saved to {comparison_path}")
    
    return comparison_df 