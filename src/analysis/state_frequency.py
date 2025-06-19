"""
State frequency and price cycle analysis module.

Implements analysis of state visitation patterns and price cycles in Q-learning.
Based on the paper's analysis of learning dynamics and convergence patterns.

References:
    - Section 4.3: Learning dynamics and state transitions
    - Figure 2: Price trajectory analysis
    - Appendix B: Cycle detection algorithms
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import pandas as pd
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

# Handle imports for both package and standalone usage
try:
    from ..params import SimParams
except ImportError:
    try:
        from params import SimParams
    except ImportError:
        # Fallback for testing
        import numpy as np
        class SimParams:
            def __init__(self, config=None):
                self.n_agents = 2
                self.n_actions = 11
                self.lambda_param = 0.5
                self.a_param = 1.0
                self.demand_model = "logit"



def count_state_freq(price_hist: np.ndarray, price_grid: np.ndarray, 
                    normalize: bool = True) -> np.ndarray:
    """
    Count frequency of price states in simulation history.
    
    Args:
        price_hist: Price history array of shape (time_steps, n_agents)
        price_grid: Discrete price grid used in simulation
        normalize: Whether to normalize frequencies to probabilities
        
    Returns:
        State frequency array of shape (n_price_states,)
        
    Note:
        State is encoded as sum_i (price_index_i * n_prices^i)
    """
    if len(price_hist) == 0:
        return zeros(len(price_grid) ** price_hist.shape[1] if len(price_hist.shape) > 1 else len(price_grid))
    
    n_agents = price_hist.shape[1] if len(price_hist.shape) > 1 else 1
    n_prices = len(price_grid)
    n_states = n_prices ** n_agents
    
    # Initialize frequency counter
    state_freq = zeros(n_states, dtype=int)
    
    # Convert prices to state indices
    for t in range(len(price_hist)):
        if n_agents == 1:
            prices = [price_hist[t]]
        else:
            prices = price_hist[t, :]
        
        # Find closest price grid points
        state_index = 0
        for agent, price in enumerate(prices):
            # Find closest grid point
            price_index = np.argmin(np.abs(price_grid - price))
            state_index += price_index * (n_prices ** agent)
        
        state_freq[state_index] += 1
    
    # Normalize if requested
    if normalize and np.sum(state_freq) > 0:
        state_freq = state_freq.astype(DTYPE) / np.sum(state_freq)
    
    return state_freq


def compute_state_transition_matrix(price_hist: np.ndarray, price_grid: np.ndarray) -> np.ndarray:
    """
    Compute empirical state transition matrix from price history.
    
    Args:
        price_hist: Price history array
        price_grid: Discrete price grid
        
    Returns:
        Transition matrix of shape (n_states, n_states)
    """
    if len(price_hist) < 2:
        n_agents = price_hist.shape[1] if len(price_hist.shape) > 1 else 1
        n_states = len(price_grid) ** n_agents
        return zeros((n_states, n_states))
    
    n_agents = price_hist.shape[1] if len(price_hist.shape) > 1 else 1
    n_prices = len(price_grid)
    n_states = n_prices ** n_agents
    
    # Initialize transition count matrix
    transition_counts = zeros((n_states, n_states), dtype=int)
    
    # Count transitions
    for t in range(len(price_hist) - 1):
        # Current state
        if n_agents == 1:
            current_prices = [price_hist[t]]
            next_prices = [price_hist[t + 1]]
        else:
            current_prices = price_hist[t, :]
            next_prices = price_hist[t + 1, :]
        
        # Convert to state indices
        current_state = prices_to_state_index(current_prices, price_grid)
        next_state = prices_to_state_index(next_prices, price_grid)
        
        transition_counts[current_state, next_state] += 1
    
    # Normalize to probabilities
    transition_matrix = zeros((n_states, n_states))
    for state in range(n_states):
        total_transitions = np.sum(transition_counts[state, :])
        if total_transitions > 0:
            transition_matrix[state, :] = transition_counts[state, :].astype(DTYPE) / total_transitions
    
    return transition_matrix


def prices_to_state_index(prices: List[float], price_grid: np.ndarray) -> int:
    """
    Convert price vector to state index.
    
    Args:
        prices: List of prices for each agent
        price_grid: Discrete price grid
        
    Returns:
        State index
    """
    state_index = 0
    n_prices = len(price_grid)
    
    for agent, price in enumerate(prices):
        price_index = np.argmin(np.abs(price_grid - price))
        state_index += price_index * (n_prices ** agent)
    
    return state_index


def state_index_to_prices(state_index: int, price_grid: np.ndarray, n_agents: int) -> List[float]:
    """
    Convert state index back to price vector.
    
    Args:
        state_index: State index
        price_grid: Discrete price grid
        n_agents: Number of agents
        
    Returns:
        List of prices for each agent
    """
    n_prices = len(price_grid)
    prices = []
    
    remaining_index = state_index
    for agent in range(n_agents):
        price_index = remaining_index % n_prices
        prices.append(price_grid[price_index])
        remaining_index //= n_prices
    
    return prices


def detect_cycles(price_hist: np.ndarray, max_period: int = 5, min_occurrences: int = 2, tol: float = 1e-4) -> Dict[int, int]:
    """
    Detect periodic patterns in price history using array comparison.
    
    Args:
        price_hist: Price history array of shape (time_steps, n_agents)
        max_period: Maximum cycle period to search for
        min_occurrences: Minimum number of cycle occurrences to count
        tol: Tolerance for numerical comparison
        
    Returns:
        Dictionary mapping cycle periods to occurrence counts
    """
    if len(price_hist) < 2 * max_period:
        return {}
    
    cycle_counts = {}
    
    # Search for cycles of different periods
    for period in range(1, max_period + 1):
        occurrences = 0
        
        # Check sliding windows for repeating patterns
        for start in range(len(price_hist) - 2 * period + 1):
            # Extract potential cycle pattern
            pattern1 = price_hist[start:start + period]
            pattern2 = price_hist[start + period:start + 2 * period]
            
            # Check if patterns match within tolerance
            if np.allclose(pattern1, pattern2, atol=tol):
                occurrences += 1
        
        # Only count if minimum occurrences met
        if occurrences >= min_occurrences:
            cycle_counts[period] = occurrences
    
    return cycle_counts


def find_cycles_of_period(sequence: List, period: int) -> Dict[Tuple, int]:
    """
    Find all cycles of a specific period in the sequence.
    
    Args:
        sequence: Price sequence (list of values or tuples)
        period: Cycle period to search for
        
    Returns:
        Dictionary mapping cycle patterns to occurrence counts
    """
    if len(sequence) < 2 * period:
        return {}
    
    cycle_patterns = defaultdict(int)
    
    # Extract all possible cycles of this period
    for start in range(len(sequence) - 2 * period + 1):
        # Extract potential cycle
        pattern = tuple(sequence[start:start + period])
        
        # Check if this pattern repeats immediately
        next_pattern = tuple(sequence[start + period:start + 2 * period])
        
        if pattern == next_pattern:
            cycle_patterns[pattern] += 1
    
    return dict(cycle_patterns)


def analyze_price_volatility(price_hist: np.ndarray, window_size: int = 100) -> Dict[str, Any]:
    """
    Analyze price volatility over time windows.
    
    Args:
        price_hist: Price history array
        window_size: Size of rolling window for volatility calculation
        
    Returns:
        Dictionary with volatility analysis results
    """
    if len(price_hist) < window_size:
        return {'error': 'Insufficient data for volatility analysis'}
    
    n_agents = price_hist.shape[1] if len(price_hist.shape) > 1 else 1
    n_windows = len(price_hist) - window_size + 1
    
    # Calculate rolling volatility (standard deviation)
    volatilities = []
    
    for t in range(n_windows):
        window_data = price_hist[t:t + window_size]
        
        if n_agents == 1:
            volatility = np.std(window_data)
        else:
            # Average volatility across agents
            agent_volatilities = [np.std(window_data[:, agent]) for agent in range(n_agents)]
            volatility = np.mean(agent_volatilities)
        
        volatilities.append(volatility)
    
    volatilities = array(volatilities)
    
    return {
        'mean_volatility': np.mean(volatilities),
        'std_volatility': np.std(volatilities),
        'min_volatility': np.min(volatilities),
        'max_volatility': np.max(volatilities),
        'volatility_series': volatilities.tolist(),
        'declining_volatility': np.mean(volatilities[:len(volatilities)//2]) > np.mean(volatilities[len(volatilities)//2:])
    }


def compute_price_correlation(price_hist: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix between agents' prices.
    
    Args:
        price_hist: Price history array of shape (time_steps, n_agents)
        
    Returns:
        Correlation matrix of shape (n_agents, n_agents)
    """
    if len(price_hist.shape) == 1 or price_hist.shape[1] == 1:
        return array([[1.0]])
    
    return np.corrcoef(price_hist.T)


def analyze_convergence_pattern(price_hist: np.ndarray, convergence_threshold: float = 1e-4) -> Dict[str, Any]:
    """
    Analyze the pattern of convergence in price history.
    
    Args:
        price_hist: Price history array
        convergence_threshold: Threshold for considering prices converged
        
    Returns:
        Dictionary with convergence pattern analysis
    """
    if len(price_hist) < 10:
        return {'error': 'Insufficient data for convergence analysis'}
    
    n_agents = price_hist.shape[1] if len(price_hist.shape) > 1 else 1
    
    # Calculate price ranges in sliding windows
    window_size = min(100, len(price_hist) // 10)
    convergence_times = []
    
    if n_agents == 1:
        # Single agent case
        for t in range(window_size, len(price_hist)):
            window_data = price_hist[t - window_size:t]
            price_range = np.max(window_data) - np.min(window_data)
            
            if price_range <= convergence_threshold:
                convergence_times.append(t - window_size)
                break
    else:
        # Multi-agent case
        for t in range(window_size, len(price_hist)):
            window_data = price_hist[t - window_size:t, :]
            converged = True
            
            for agent in range(n_agents):
                agent_range = np.max(window_data[:, agent]) - np.min(window_data[:, agent])
                if agent_range > convergence_threshold:
                    converged = False
                    break
            
            if converged:
                convergence_times.append(t - window_size)
                break
    
    # Analyze monotonicity (are prices generally moving towards equilibrium?)
    if n_agents == 1:
        trend = np.polyfit(range(len(price_hist)), price_hist, 1)[0]
        monotonic = abs(trend) > 1e-6
    else:
        trends = []
        for agent in range(n_agents):
            trend = np.polyfit(range(len(price_hist)), price_hist[:, agent], 1)[0]
            trends.append(trend)
        
        # Consider convergent if all trends are small
        monotonic = any(abs(t) > 1e-6 for t in trends)
    
    return {
        'converged': len(convergence_times) > 0,
        'convergence_time': convergence_times[0] if convergence_times else None,
        'monotonic_trend': monotonic,
        'final_price_range': np.max(price_hist[-min(50, len(price_hist)):]) - np.min(price_hist[-min(50, len(price_hist)):]) if n_agents == 1 else None
    }


def create_state_frequency_dataframe(state_freq: np.ndarray, price_grid: np.ndarray, 
                                   n_agents: int) -> pd.DataFrame:
    """
    Create a DataFrame with state frequencies for analysis.
    
    Args:
        state_freq: State frequency array
        price_grid: Price grid used
        n_agents: Number of agents
        
    Returns:
        DataFrame with state information and frequencies
    """
    data = []
    
    for state_idx, freq in enumerate(state_freq):
        if freq > 0:  # Only include visited states
            prices = state_index_to_prices(state_idx, price_grid, n_agents)
            
            row = {
                'state_index': state_idx,
                'frequency': freq,
                'probability': freq if isinstance(freq, float) else freq / np.sum(state_freq)
            }
            
            # Add agent-specific prices
            for agent, price in enumerate(prices):
                row[f'price_agent_{agent + 1}'] = price
            
            data.append(row)
    
    return pd.DataFrame(data)


def analyze_state_persistence(price_hist: np.ndarray, price_grid: np.ndarray) -> Dict[str, Any]:
    """
    Analyze how long the system stays in each state.
    
    Args:
        price_hist: Price history array
        price_grid: Price grid used
        
    Returns:
        Dictionary with state persistence analysis
    """
    if len(price_hist) < 2:
        return {'error': 'Insufficient data for persistence analysis'}
    
    n_agents = price_hist.shape[1] if len(price_hist.shape) > 1 else 1
    
    # Convert price history to state sequence
    state_sequence = []
    for t in range(len(price_hist)):
        if n_agents == 1:
            prices = [price_hist[t]]
        else:
            prices = price_hist[t, :]
        
        state_idx = prices_to_state_index(prices, price_grid)
        state_sequence.append(state_idx)
    
    # Analyze persistence (consecutive occurrences of same state)
    state_durations = defaultdict(list)
    current_state = state_sequence[0]
    current_duration = 1
    
    for t in range(1, len(state_sequence)):
        if state_sequence[t] == current_state:
            current_duration += 1
        else:
            state_durations[current_state].append(current_duration)
            current_state = state_sequence[t]
            current_duration = 1
    
    # Add final duration
    state_durations[current_state].append(current_duration)
    
    # Compute statistics
    persistence_stats = {}
    for state, durations in state_durations.items():
        persistence_stats[state] = {
            'mean_duration': np.mean(durations),
            'max_duration': np.max(durations),
            'total_visits': len(durations),
            'total_time': sum(durations)
        }
    
    return {
        'state_persistence': persistence_stats,
        'most_persistent_state': max(persistence_stats.keys(), 
                                   key=lambda s: persistence_stats[s]['mean_duration']),
        'most_visited_state': max(persistence_stats.keys(), 
                                key=lambda s: persistence_stats[s]['total_visits'])
    } 