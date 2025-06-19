"""
Impulse response analysis module for Phase 2.
Analyzes how the system responds to price shocks.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, TYPE_CHECKING, Any
from dataclasses import dataclass

if TYPE_CHECKING:
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


@dataclass
class ImpulseResponseResult:
    """Container for impulse response analysis results."""
    shock_time: int
    shock_magnitude: float
    affected_agent: int
    price_response: np.ndarray  # Price trajectory after shock
    profit_response: np.ndarray  # Profit trajectory after shock
    convergence_time: Optional[int]  # Time to return to equilibrium
    max_deviation: float  # Maximum deviation from pre-shock state
    recovery_rate: Optional[float]  # Rate of return to equilibrium

def analyze_impulse_response(
    price_history: np.ndarray,
    profit_history: np.ndarray,
    shock_time: int,
    shock_magnitude: float,
    affected_agent: int,
    convergence_threshold: float = 1e-3,
    max_recovery_time: int = 100
) -> ImpulseResponseResult:
    """
    Analyze system response to a price shock.
    
    Args:
        price_history: Array of price trajectories
        profit_history: Array of profit trajectories
        shock_time: Time step when shock occurs
        shock_magnitude: Size of price shock
        affected_agent: Index of agent receiving shock
        convergence_threshold: Threshold for considering system converged
        max_recovery_time: Maximum time to track recovery
        
    Returns:
        ImpulseResponseResult containing analysis results
    """
    if not isinstance(price_history, np.ndarray) or not isinstance(profit_history, np.ndarray):
        raise TypeError("Price and profit histories must be numpy arrays")
    
    if price_history.shape != profit_history.shape:
        raise ValueError("Price and profit histories must have same shape")
    
    if shock_time < 0 or shock_time >= len(price_history):
        raise ValueError("Shock time must be within history range")
    
    if affected_agent < 0 or affected_agent >= price_history.shape[1]:
        raise ValueError("Affected agent index out of range")
    
    # Extract pre-shock equilibrium
    pre_shock_prices = price_history[shock_time]
    pre_shock_profits = profit_history[shock_time]
    
    # Calculate post-shock trajectories
    post_shock_prices = price_history[shock_time+1:shock_time+max_recovery_time+1]
    post_shock_profits = profit_history[shock_time+1:shock_time+max_recovery_time+1]
    
    if len(post_shock_prices) == 0:
        return ImpulseResponseResult(
            shock_time=shock_time,
            shock_magnitude=shock_magnitude,
            affected_agent=affected_agent,
            price_response=array([]),
            profit_response=array([]),
            convergence_time=None,
            max_deviation=0.0,
            recovery_rate=None
        )
    
    # Calculate deviations from pre-shock state
    price_deviations = np.abs(post_shock_prices - pre_shock_prices)
    profit_deviations = np.abs(post_shock_profits - pre_shock_profits)
    
    # Find maximum deviation
    max_price_dev = np.max(price_deviations)
    max_profit_dev = np.max(profit_deviations)
    max_deviation = max(max_price_dev, max_profit_dev)
    
    # Check for convergence
    convergence_time = None
    recovery_rate = None
    
    for t in range(len(post_shock_prices)):
        if np.all(price_deviations[t] <= convergence_threshold) and \
           np.all(profit_deviations[t] <= convergence_threshold):
            convergence_time = t
            break
    
    if convergence_time is not None:
        # Calculate recovery rate (inverse of convergence time)
        recovery_rate = 1.0 / (convergence_time + 1)
    
    return ImpulseResponseResult(
        shock_time=shock_time,
        shock_magnitude=shock_magnitude,
        affected_agent=affected_agent,
        price_response=post_shock_prices,
        profit_response=post_shock_profits,
        convergence_time=convergence_time,
        max_deviation=max_deviation,
        recovery_rate=recovery_rate
    )

def analyze_multiple_shocks(
    price_history: np.ndarray,
    profit_history: np.ndarray,
    shock_times: List[int],
    shock_magnitudes: List[float],
    affected_agents: List[int],
    convergence_threshold: float = 1e-3,
    max_recovery_time: int = 100
) -> List[ImpulseResponseResult]:
    """
    Analyze system response to multiple price shocks.
    
    Args:
        price_history: Array of price trajectories
        profit_history: Array of profit trajectories
        shock_times: List of time steps when shocks occur
        shock_magnitudes: List of shock magnitudes
        affected_agents: List of affected agent indices
        convergence_threshold: Threshold for considering system converged
        max_recovery_time: Maximum time to track recovery
        
    Returns:
        List of ImpulseResponseResult objects
    """
    if len(shock_times) != len(shock_magnitudes) or len(shock_times) != len(affected_agents):
        raise ValueError("Shock parameters must have same length")
    
    results = []
    for t, mag, agent in zip(shock_times, shock_magnitudes, affected_agents):
        result = analyze_impulse_response(
            price_history=price_history,
            profit_history=profit_history,
            shock_time=t,
            shock_magnitude=mag,
            affected_agent=agent,
            convergence_threshold=convergence_threshold,
            max_recovery_time=max_recovery_time
        )
        results.append(result)
    
    return results

def calculate_shock_statistics(
    results: List[ImpulseResponseResult]
) -> Dict[str, float]:
    """
    Calculate aggregate statistics from multiple shock responses.
    
    Args:
        results: List of ImpulseResponseResult objects
        
    Returns:
        Dictionary of aggregate statistics
    """
    if not results:
        return {
            'mean_convergence_time': 0.0,
            'mean_max_deviation': 0.0,
            'mean_recovery_rate': 0.0,
            'convergence_rate': 0.0
        }
    
    # Calculate statistics
    conv_times = [r.convergence_time for r in results if r.convergence_time is not None]
    max_deviations = [r.max_deviation for r in results]
    recovery_rates = [r.recovery_rate for r in results if r.recovery_rate is not None]
    
    stats = {
        'mean_convergence_time': np.mean(conv_times) if conv_times else 0.0,
        'mean_max_deviation': np.mean(max_deviations),
        'mean_recovery_rate': np.mean(recovery_rates) if recovery_rates else 0.0,
        'convergence_rate': len(conv_times) / len(results)
    }
    
    return stats

def find_optimal_shock_time(
    price_history: np.ndarray,
    profit_history: np.ndarray,
    shock_magnitude: float,
    affected_agent: int,
    convergence_threshold: float = 1e-3,
    max_recovery_time: int = 100,
    min_stable_period: int = 10
) -> Tuple[int, ImpulseResponseResult]:
    """
    Find the optimal time to apply a shock for maximum system response.
    
    Args:
        price_history: Array of price trajectories
        profit_history: Array of profit trajectories
        shock_magnitude: Size of price shock
        affected_agent: Index of agent receiving shock
        convergence_threshold: Threshold for considering system converged
        max_recovery_time: Maximum time to track recovery
        min_stable_period: Minimum period of stability before shock
        
    Returns:
        Tuple of (optimal_time, ImpulseResponseResult)
    """
    if len(price_history) < min_stable_period + max_recovery_time:
        raise ValueError("Price history too short for analysis")
    
    best_time = None
    best_result = None
    max_response = -float('inf')
    
    # Check each possible shock time
    for t in range(min_stable_period, len(price_history) - max_recovery_time):
        # Verify stability before shock
        pre_shock_prices = price_history[t-min_stable_period:t]
        if not np.all(np.diff(pre_shock_prices, axis=0) <= convergence_threshold):
            continue
        
        # Analyze response
        result = analyze_impulse_response(
            price_history=price_history,
            profit_history=profit_history,
            shock_time=t,
            shock_magnitude=shock_magnitude,
            affected_agent=affected_agent,
            convergence_threshold=convergence_threshold,
            max_recovery_time=max_recovery_time
        )
        
        # Update best result if this one has larger response
        if result.max_deviation > max_response:
            max_response = result.max_deviation
            best_time = t
            best_result = result
    
    if best_time is None:
        raise ValueError("No suitable shock time found")
    
    return best_time, best_result


def simulate_impulse(params, shock_price: float, shock_agent: int, 
                    shock_duration: int = 1, steps: int = 50) -> Any:
    """
    Simulate impulse response with price shock.
    
    Args:
        params: Simulation parameters
        shock_price: Price level for shock
        shock_agent: Agent receiving the shock
        shock_duration: Duration of shock in periods
        steps: Number of time steps to simulate
        
    Returns:
        DataFrame with time series of system response
    """
    import pandas as pd
    
    # Create time series data
    time_series = []
    
    for t in range(steps):
        # Determine if shock is active
        shock_active = t < shock_duration
        
        # Simulate basic price dynamics (simplified)
        if shock_active:
            # During shock period
            if hasattr(params, 'n_agents'):
                prices = [shock_price if i == shock_agent else 0.5 for i in range(params.n_agents)]
            else:
                prices = [shock_price, 0.5]  # Default to 2 agents
        else:
            # Post-shock convergence (simplified exponential decay)
            if hasattr(params, 'n_agents'):
                base_prices = [0.5] * params.n_agents
            else:
                base_prices = [0.5, 0.5]  # Default to 2 agents
            
            # Add some decay towards equilibrium
            decay_factor = np.exp(-0.1 * (t - shock_duration)) if t >= shock_duration else 1.0
            prices = [p + 0.1 * decay_factor * np.sin(0.3 * t) for p in base_prices]
        
        # Calculate simplified Nash and cooperative gaps
        nash_equilibrium = 0.5  # Simplified assumption
        coop_equilibrium = 0.6  # Simplified assumption
        
        avg_price = np.mean(prices)
        nash_gap = abs(avg_price - nash_equilibrium)
        coop_gap = abs(avg_price - coop_equilibrium)
        
        # Store time step data
        time_series.append({
            'time': t,
            'shock_active': shock_active,
            'agent_0_price': prices[0],
            'agent_1_price': prices[1] if len(prices) > 1 else prices[0],
            'nash_gap': nash_gap,
            'coop_gap': coop_gap
        })
    
    return pd.DataFrame(time_series) 