"""
Learning trajectory analysis module for Phase 2.

Implements Monte Carlo analysis of learning trajectories with multiple seeds,
tracking price histories, Q-gap histories, and incentive compatibility measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import json
import os

# Handle imports for both package and standalone usage
try:
    from ..dtype_policy import DTYPE, array, zeros
    from ..params import SimParams
    from ..q_learning import QLearningSimulation
    from ..convergence import calculate_nash_equilibrium
    from .qgap_to_maximum import compute_qgap, analyze_qgap_convergence
except ImportError:
    try:
        from dtype_policy import DTYPE, array, zeros
        from params import SimParams
        from q_learning import QLearningSimulation
        from convergence import calculate_nash_equilibrium
        from qgap_to_maximum import compute_qgap, analyze_qgap_convergence
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
                self.lambda_param = 0.5
                self.a_param = 1.0
                self.demand_model = "logit"
        
        class QLearningSimulation:
            def __init__(self, params): pass
            def run_simulation(self, seed=None): 
                return {
                    'price_history': np.random.rand(1000, 2),
                    'Q_matrices': [np.random.rand(10, 11), np.random.rand(10, 11)],
                    'converged': True
                }
        
        def calculate_nash_equilibrium(params):
            return np.array([1.0, 1.0])
        
        def compute_qgap(Q):
            return np.random.rand(*Q.shape) * 0.1
        
        def analyze_qgap_convergence(Q_hist, threshold=1e-3):
            return {'convergence_time': 500, 'convergence_rate': 0.01}


@dataclass
class TrajectoryResults:
    """Container for learning trajectory analysis results."""
    seeds: List[int]
    price_histories: List[np.ndarray]
    qgap_histories: List[List[np.ndarray]]
    incentive_compatibility: List[np.ndarray]
    convergence_times: List[Optional[int]]
    convergence_rates: List[Optional[float]]
    final_prices: List[np.ndarray]
    nash_deviations: List[float]


def compute_incentive_compatibility(prices: np.ndarray, nash_prices: np.ndarray) -> np.ndarray:
    """
    Compute incentive compatibility indicator IA_t = |π_t - π_nash|.
    
    Args:
        prices: Price history array of shape (T, n_agents)
        nash_prices: Nash equilibrium prices array of shape (n_agents,)
        
    Returns:
        Array of incentive compatibility measures over time
    """
    if prices.ndim != 2:
        raise ValueError(f"prices must be 2D array, got shape {prices.shape}")
    
    if nash_prices.ndim != 1:
        raise ValueError(f"nash_prices must be 1D array, got shape {nash_prices.shape}")
    
    T, n_agents = prices.shape
    
    if len(nash_prices) != n_agents:
        raise ValueError(f"nash_prices length {len(nash_prices)} doesn't match n_agents {n_agents}")
    
    # Compute absolute deviation from Nash for each agent and time
    deviations = np.abs(prices - nash_prices[np.newaxis, :])
    
    # Take average deviation across agents (or max, depending on interpretation)
    incentive_compatibility = np.mean(deviations, axis=1)
    
    return array(incentive_compatibility)


def run_monte_carlo_trajectories(
    params: SimParams, 
    seeds: List[int], 
    save_dir: Optional[str] = None
) -> TrajectoryResults:
    """
    Run Monte Carlo simulation with multiple seeds to analyze learning trajectories.
    
    Args:
        params: Simulation parameters
        seeds: List of random seeds to use
        save_dir: Optional directory to save results
        
    Returns:
        TrajectoryResults object with aggregated results
    """
    if not seeds:
        raise ValueError("seeds list cannot be empty")
    
    # Calculate Nash equilibrium for incentive compatibility
    try:
        nash_prices = calculate_nash_equilibrium(params)
    except Exception as e:
        print(f"Warning: Failed to calculate Nash equilibrium: {e}")
        nash_prices = np.ones(params.n_agents)  # Fallback
    
    # Initialize result containers
    price_histories = []
    qgap_histories = []
    incentive_compatibility = []
    convergence_times = []
    convergence_rates = []
    final_prices = []
    nash_deviations = []
    
    print(f"Running Monte Carlo with {len(seeds)} seeds...")
    
    for i, seed in enumerate(seeds):
        print(f"  Seed {seed} ({i+1}/{len(seeds)})")
        
        try:
            # Run simulation with current seed
            sim = QLearningSimulation(params)
            results = sim.run_simulation(seed=seed)
            
            # Extract price history
            price_hist = results.get('price_history', np.array([]))
            if price_hist.size == 0:
                print(f"    Warning: No price history for seed {seed}")
                continue
            
            price_histories.append(price_hist)
            
            # Extract Q-matrices and compute gaps over time
            Q_matrices_history = results.get('Q_history', [])
            if not Q_matrices_history:
                # Use final Q-matrices if no history available
                Q_matrices_final = results.get('Q_matrices', [])
                if Q_matrices_final:
                    Q_matrices_history = [Q_matrices_final]
            
            # Compute Q-gap history for each agent
            agent_qgap_histories = []
            for agent_idx in range(params.n_agents):
                agent_qgaps = []
                for Q_matrices_t in Q_matrices_history:
                    if agent_idx < len(Q_matrices_t) and Q_matrices_t[agent_idx] is not None:
                        qgap = compute_qgap(Q_matrices_t[agent_idx])
                        agent_qgaps.append(qgap)
                    else:
                        # Fill with zeros if Q-matrix not available
                        agent_qgaps.append(zeros((10, params.n_actions)))  # Default shape
                agent_qgap_histories.append(agent_qgaps)
            
            qgap_histories.append(agent_qgap_histories)
            
            # Compute incentive compatibility
            ia_measures = compute_incentive_compatibility(price_hist, nash_prices)
            incentive_compatibility.append(ia_measures)
            
            # Analyze convergence
            if Q_matrices_history and len(Q_matrices_history) > 0:
                # Use first agent's Q-matrices for convergence analysis
                agent_0_Q_history = [Q_t[0] if len(Q_t) > 0 else None for Q_t in Q_matrices_history]
                conv_analysis = analyze_qgap_convergence(agent_0_Q_history)
                convergence_times.append(conv_analysis.get('convergence_time'))
                convergence_rates.append(conv_analysis.get('convergence_rate'))
            else:
                convergence_times.append(None)
                convergence_rates.append(None)
            
            # Record final prices and Nash deviation
            if len(price_hist) > 0:
                final_price = price_hist[-1]
                final_prices.append(final_price)
                nash_dev = np.mean(np.abs(final_price - nash_prices))
                nash_deviations.append(nash_dev)
            else:
                final_prices.append(nash_prices.copy())
                nash_deviations.append(0.0)
                
        except Exception as e:
            print(f"    Error processing seed {seed}: {e}")
            continue
    
    # Create results object
    results = TrajectoryResults(
        seeds=seeds,
        price_histories=price_histories,
        qgap_histories=qgap_histories,
        incentive_compatibility=incentive_compatibility,
        convergence_times=convergence_times,
        convergence_rates=convergence_rates,
        final_prices=final_prices,
        nash_deviations=nash_deviations
    )
    
    # Save results if requested
    if save_dir:
        save_trajectory_results(results, save_dir)
    
    return results


def save_trajectory_results(results: TrajectoryResults, save_dir: str) -> None:
    """
    Save trajectory results to files.
    
    Args:
        results: TrajectoryResults object to save
        save_dir: Directory to save results in
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save summary statistics
    summary_data = {
        'seeds': results.seeds,
        'convergence_times': results.convergence_times,
        'convergence_rates': results.convergence_rates,
        'nash_deviations': results.nash_deviations
    }
    
    with open(os.path.join(save_dir, 'trajectory_summary.json'), 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif obj is None:
                return None
            return obj
        
        serializable_data = {k: [convert_numpy(item) for item in v] for k, v in summary_data.items()}
        json.dump(serializable_data, f, indent=2)
    
    # Save detailed data for each seed
    for i, seed in enumerate(results.seeds):
        if i < len(results.price_histories):
            seed_dir = os.path.join(save_dir, f'seed_{seed}')
            os.makedirs(seed_dir, exist_ok=True)
            
            # Save price history
            if i < len(results.price_histories):
                np.savetxt(
                    os.path.join(seed_dir, 'price_history.csv'),
                    results.price_histories[i],
                    delimiter=',',
                    header='Agent_1,Agent_2',
                    comments=''
                )
            
            # Save incentive compatibility
            if i < len(results.incentive_compatibility):
                np.savetxt(
                    os.path.join(seed_dir, 'incentive_compatibility.csv'),
                    results.incentive_compatibility[i],
                    delimiter=',',
                    header='IA_measure',
                    comments=''
                )


def load_trajectory_results(save_dir: str) -> Optional[TrajectoryResults]:
    """
    Load trajectory results from files.
    
    Args:
        save_dir: Directory to load results from
        
    Returns:
        TrajectoryResults object or None if loading fails
    """
    try:
        # Load summary
        with open(os.path.join(save_dir, 'trajectory_summary.json'), 'r') as f:
            summary_data = json.load(f)
        
        seeds = summary_data['seeds']
        convergence_times = summary_data['convergence_times']
        convergence_rates = summary_data['convergence_rates']
        nash_deviations = summary_data['nash_deviations']
        
        # Load detailed data
        price_histories = []
        incentive_compatibility = []
        final_prices = []
        
        for seed in seeds:
            seed_dir = os.path.join(save_dir, f'seed_{seed}')
            
            # Load price history
            price_file = os.path.join(seed_dir, 'price_history.csv')
            if os.path.exists(price_file):
                price_hist = np.loadtxt(price_file, delimiter=',', skiprows=1)
                price_histories.append(price_hist)
                final_prices.append(price_hist[-1] if len(price_hist) > 0 else np.array([1.0, 1.0]))
            
            # Load incentive compatibility
            ia_file = os.path.join(seed_dir, 'incentive_compatibility.csv')
            if os.path.exists(ia_file):
                ia_data = np.loadtxt(ia_file, delimiter=',', skiprows=1)
                incentive_compatibility.append(ia_data)
        
        return TrajectoryResults(
            seeds=seeds,
            price_histories=price_histories,
            qgap_histories=[],  # Not saved/loaded in this version
            incentive_compatibility=incentive_compatibility,
            convergence_times=convergence_times,
            convergence_rates=convergence_rates,
            final_prices=final_prices,
            nash_deviations=nash_deviations
        )
        
    except Exception as e:
        print(f"Failed to load trajectory results: {e}")
        return None


def summarize_trajectories(results: TrajectoryResults) -> pd.DataFrame:
    """
    Create summary DataFrame of trajectory analysis results.
    
    Args:
        results: TrajectoryResults object
        
    Returns:
        DataFrame with summary statistics
    """
    if not results.seeds:
        return pd.DataFrame()
    
    summary_rows = []
    
    for i, seed in enumerate(results.seeds):
        row = {'Seed': seed}
        
        # Convergence metrics
        if i < len(results.convergence_times):
            row['Convergence_Time'] = results.convergence_times[i]
        if i < len(results.convergence_rates):
            row['Convergence_Rate'] = f"{results.convergence_rates[i]:.6f}" if results.convergence_rates[i] is not None else "N/A"
        
        # Nash deviation
        if i < len(results.nash_deviations):
            row['Nash_Deviation'] = f"{results.nash_deviations[i]:.6f}"
        
        # Final prices
        if i < len(results.final_prices):
            final_price = results.final_prices[i]
            for j, price in enumerate(final_price):
                row[f'Final_Price_Agent_{j+1}'] = f"{price:.6f}"
        
        # Incentive compatibility statistics
        if i < len(results.incentive_compatibility):
            ia_data = results.incentive_compatibility[i]
            if len(ia_data) > 0:
                row['IA_Mean'] = f"{np.mean(ia_data):.6f}"
                row['IA_Final'] = f"{ia_data[-1]:.6f}"
                row['IA_Max'] = f"{np.max(ia_data):.6f}"
        
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows) 