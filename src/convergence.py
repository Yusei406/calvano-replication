"""
Convergence detection and analysis for Q-learning.
Implements the exact convergence criteria from the paper.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Handle imports for both package and standalone usage
try:
    from .params import SimParams
except ImportError:
    from params import SimParams

def has_converged(price_hist: np.ndarray, window: int = 1000, tol: float = 1e-4) -> bool:
    """
    Check if prices have converged based on paper's criteria.
    
    Args:
        price_hist: Price history array of shape (time_steps, n_agents)
        window: Number of steps to check for convergence
        tol: Tolerance for price changes
        
    Returns:
        True if converged, False otherwise
    """
    if len(price_hist) < window:
        return False
    
    # Check last 'window' steps
    recent_prices = price_hist[-window:]
    
    # Calculate price range in the window
    price_ranges = np.max(recent_prices, axis=0) - np.min(recent_prices, axis=0)
    
    # Converged if all agents' price ranges are within tolerance
    return np.all(price_ranges <= tol)


def has_strategy_converged(strategy_hist: List[np.ndarray], window: int = 1000) -> bool:
    """
    Check if strategies have converged (no changes in window).
    
    Args:
        strategy_hist: List of strategy arrays over time
        window: Number of steps to check for convergence
        
    Returns:
        True if strategies haven't changed in the last 'window' steps
    """
    if len(strategy_hist) < window:
        return False
    
    # Check if strategy is the same for the last 'window' steps
    reference_strategy = strategy_hist[-1]
    for i in range(1, window + 1):
        if not np.array_equal(strategy_hist[-i], reference_strategy):
            return False
    
    return True


def compute_nash_distance(prices: np.ndarray, params: SimParams) -> float:
    """
    Compute distance from Nash equilibrium.
    
    Args:
        prices: Current prices array of shape (n_agents,)
        params: Simulation parameters
        
    Returns:
        Average distance from theoretical Nash prices
    """
    # Theoretical Nash price for logit demand (simplified)
    nash_price = compute_theoretical_nash_price(params)
    nash_prices = np.full(params.n_agents, nash_price)
    
    return np.mean(np.abs(prices - nash_prices))


def compute_cooperative_distance(prices: np.ndarray, params: SimParams) -> float:
    """
    Compute distance from cooperative equilibrium.
    
    Args:
        prices: Current prices array of shape (n_agents,)
        params: Simulation parameters
        
    Returns:
        Average distance from theoretical cooperative prices
    """
    # Theoretical cooperative price for logit demand (simplified)
    coop_price = compute_theoretical_coop_price(params)
    coop_prices = np.full(params.n_agents, coop_price)
    
    return np.mean(np.abs(prices - coop_prices))


def compute_theoretical_nash_price(params: SimParams) -> float:
    """
    Compute theoretical Nash equilibrium price.
    
    Args:
        params: Simulation parameters
        
    Returns:
        Nash equilibrium price
    """
    # Simplified theoretical calculation
    # In the actual implementation, this would involve solving the Nash equilibrium
    if params.demand_model == "logit":
        # Mock calculation for logit demand
        return 0.5
    else:
        # Default Nash price
        return 0.5


def compute_theoretical_coop_price(params: SimParams) -> float:
    """
    Compute theoretical cooperative (collusive) price.
    
    Args:
        params: Simulation parameters
        
    Returns:
        Cooperative equilibrium price
    """
    # Simplified theoretical calculation
    # Cooperative price is typically higher than Nash
    if params.demand_model == "logit":
        # Mock calculation for cooperative outcome
        return 0.8
    else:
        # Default cooperative price
        return 0.8


def analyze_convergence(
    price_hist: np.ndarray, 
    strategy_hist: List[np.ndarray],
    params: SimParams,
    window: int = 1000,
    tol: float = 1e-4
) -> Dict[str, Any]:
    """
    Comprehensive convergence analysis.
    
    Args:
        price_hist: Price history array
        strategy_hist: Strategy history list
        params: Simulation parameters
        window: Convergence window
        tol: Convergence tolerance
        
    Returns:
        Dictionary with convergence statistics
    """
    result = {}
    
    # Basic convergence flags
    result['price_converged'] = has_converged(price_hist, window, tol)
    result['strategy_converged'] = has_strategy_converged(strategy_hist, window)
    result['overall_converged'] = result['price_converged'] and result['strategy_converged']
    
    # Final prices and strategies
    if len(price_hist) > 0:
        final_prices = price_hist[-1]
        result['final_prices'] = final_prices
        result['nash_distance'] = compute_nash_distance(final_prices, params)
        result['coop_distance'] = compute_cooperative_distance(final_prices, params)
    else:
        result['final_prices'] = np.zeros(params.n_agents)
        result['nash_distance'] = np.inf
        result['coop_distance'] = np.inf
    
    if len(strategy_hist) > 0:
        result['final_strategy'] = strategy_hist[-1]
    else:
        result['final_strategy'] = np.zeros((params.n_states, params.n_agents), dtype=int)
    
    # Convergence time
    result['convergence_time'] = find_convergence_time(price_hist, window, tol)
    
    # Price volatility in final window
    if len(price_hist) >= window:
        final_window = price_hist[-window:]
        result['final_volatility'] = np.mean(np.std(final_window, axis=0))
    else:
        result['final_volatility'] = np.inf
    
    return result


def find_convergence_time(
    price_hist: np.ndarray, 
    window: int = 1000, 
    tol: float = 1e-4
) -> Optional[int]:
    """
    Find the iteration when convergence was first achieved.
    
    Args:
        price_hist: Price history array
        window: Convergence window
        tol: Convergence tolerance
        
    Returns:
        Iteration number when convergence started, or None if not converged
    """
    if len(price_hist) < window:
        return None
    
    # Check each possible starting point
    for start_idx in range(len(price_hist) - window + 1):
        window_prices = price_hist[start_idx:start_idx + window]
        price_ranges = np.max(window_prices, axis=0) - np.min(window_prices, axis=0)
        
        if np.all(price_ranges <= tol):
            return start_idx
    
    return None


def compute_cycle_length(strategy: np.ndarray, start_state: int, params: SimParams) -> Tuple[int, List[int]]:
    """
    Compute cycle length starting from a given state.
    
    Args:
        strategy: Strategy matrix of shape (n_states, n_agents)
        start_state: Starting state
        params: Simulation parameters
        
    Returns:
        Tuple of (cycle_length, cycle_states)
    """
    visited_states = []
    current_state = start_state
    
    while current_state not in visited_states:
        visited_states.append(current_state)
        
        # Get actions for current state
        actions = strategy[current_state, :]
        
        # Compute next state
        next_state = 0
        for i, action in enumerate(actions):
            next_state += action * (params.n_actions ** i)
        
        current_state = next_state
        
        # Safety check to prevent infinite loops
        if len(visited_states) > params.n_states:
            break
    
    if current_state in visited_states:
        cycle_start = visited_states.index(current_state)
        cycle_states = visited_states[cycle_start:]
        cycle_length = len(cycle_states)
    else:
        cycle_states = visited_states
        cycle_length = len(visited_states)
    
    return cycle_length, cycle_states


def save_convergence_stats(file_path: str, result_dict: Dict[str, Any]) -> None:
    """
    Save convergence statistics to file for Phase 2 analysis.
    
    Args:
        file_path: Output file path
        result_dict: Convergence analysis results
    """
    with open(file_path, 'w') as f:
        f.write("Convergence Analysis Results\n")
        f.write("============================\n\n")
        
        # Basic convergence info
        f.write(f"Price Converged: {result_dict['price_converged']}\n")
        f.write(f"Strategy Converged: {result_dict['strategy_converged']}\n")
        f.write(f"Overall Converged: {result_dict['overall_converged']}\n")
        f.write(f"Convergence Time: {result_dict['convergence_time']}\n")
        f.write(f"Final Volatility: {result_dict['final_volatility']:.6f}\n\n")
        
        # Equilibrium distances
        f.write(f"Nash Distance: {result_dict['nash_distance']:.6f}\n")
        f.write(f"Cooperative Distance: {result_dict['coop_distance']:.6f}\n\n")
        
        # Final prices
        f.write("Final Prices:\n")
        for i, price in enumerate(result_dict['final_prices']):
            f.write(f"  Agent {i+1}: {price:.6f}\n")
        f.write("\n")
        
        # Final strategy (sample of first few states)
        f.write("Final Strategy (first 10 states):\n")
        strategy = result_dict['final_strategy']
        for state in range(min(10, strategy.shape[0])):
            actions = " ".join(f"{strategy[state, agent]:2d}" for agent in range(strategy.shape[1]))
            f.write(f"  State {state:2d}: [{actions}]\n") 