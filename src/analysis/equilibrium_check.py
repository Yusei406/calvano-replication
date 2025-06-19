"""
Equilibrium check module.

Implements functions to verify Nash and cooperative equilibria.
Based on the paper's equilibrium definitions and convergence criteria.

References:
    - Section 3.1: Nash equilibrium definition
    - Section 3.3: Cooperative equilibrium definition  
    - Section 4.1: Convergence criteria and validation
"""

import numpy as np
from typing import List, Union, Tuple
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

# Handle imports for both package and standalone usage
try:
    from ..dtype_policy import DTYPE, array
except ImportError:
    try:
        from dtype_policy import DTYPE, array
    except ImportError:
        # Fallback for testing
        import numpy as np
        DTYPE = np.float64
        array = lambda x: np.array(x, dtype=DTYPE)
        zeros = lambda n: np.zeros(n, dtype=DTYPE)

# Handle imports for both package and standalone usage
try:
    from ..convergence import compute_theoretical_nash_price, compute_theoretical_coop_price
except ImportError:
    try:
        from convergence import compute_theoretical_nash_price, compute_theoretical_coop_price
    except ImportError:
        # Fallback for testing
        import numpy as np

from .profit_gain import compute_demands_from_prices, calc_profit_vector, zeros


def check_nash(price_pair: Union[List[float], np.ndarray], params: SimParams, 
               tol: float = 1e-4) -> bool:
    """
    Check if a price pair constitutes a Nash equilibrium.
    
    Args:
        price_pair: Prices for each agent
        params: Simulation parameters
        tol: Tolerance for equilibrium check
        
    Returns:
        True if prices are Nash equilibrium within tolerance
    """
    prices = array(price_pair)
    n_agents = len(prices)
    
    # Check each agent's best response condition
    for agent in range(n_agents):
        # Current profit
        demands = compute_demands_from_prices(prices, params)
        current_profit = calc_profit_vector(prices, zeros(n_agents), demands)[agent]
        
        # Check if any deviation improves profit significantly
        is_best_response = True
        
        # Test deviations on a fine grid
        test_prices = np.linspace(0.0, 1.0, 101)  # Fine grid for testing
        
        for test_price in test_prices:
            # Create deviated price vector
            deviated_prices = prices.copy()
            deviated_prices[agent] = test_price
            
            # Compute profit with deviation
            dev_demands = compute_demands_from_prices(deviated_prices, params)
            dev_profit = calc_profit_vector(deviated_prices, zeros(n_agents), dev_demands)[agent]
            
            # Check if deviation improves profit beyond tolerance
            if dev_profit > current_profit + tol:
                is_best_response = False
                break
        
        if not is_best_response:
            return False
    
    return True


def check_coop(price_pair: Union[List[float], np.ndarray], params: SimParams,
               tol: float = 1e-4) -> bool:
    """
    Check if a price pair constitutes a cooperative equilibrium.
    
    Args:
        price_pair: Prices for each agent
        params: Simulation parameters
        tol: Tolerance for equilibrium check
        
    Returns:
        True if prices are cooperative equilibrium within tolerance
    """
    prices = array(price_pair)
    n_agents = len(prices)
    
    # Compute current total welfare
    demands = compute_demands_from_prices(prices, params)
    profits = calc_profit_vector(prices, zeros(n_agents), demands)
    current_welfare = np.sum(profits)
    
    # Check if any joint deviation improves total welfare significantly
    # Test symmetric deviations (both agents move to same price)
    test_prices = np.linspace(0.0, 1.0, 101)
    
    for test_price in test_prices:
        # All agents set same price
        test_price_vector = array([test_price] * n_agents)
        
        # Compute welfare with this price
        test_demands = compute_demands_from_prices(test_price_vector, params)
        test_profits = calc_profit_vector(test_price_vector, zeros(n_agents), test_demands)
        test_welfare = np.sum(test_profits)
        
        # Check if this improves welfare beyond tolerance
        if test_welfare > current_welfare + tol:
            return False
    
    return True


def check_theoretical_nash(price_pair: Union[List[float], np.ndarray], 
                          params: SimParams, tol: float = 1e-4) -> bool:
    """
    Check if prices match theoretical Nash equilibrium.
    
    Args:
        price_pair: Prices for each agent
        params: Simulation parameters
        tol: Tolerance for comparison
        
    Returns:
        True if prices match theoretical Nash within tolerance
    """
    prices = array(price_pair)
    theoretical_nash = compute_theoretical_nash_price(params)
    
    # For symmetric games, all agents should have same price
    return all(abs(price - theoretical_nash) <= tol for price in prices)


def check_theoretical_coop(price_pair: Union[List[float], np.ndarray],
                          params: SimParams, tol: float = 1e-4) -> bool:
    """
    Check if prices match theoretical cooperative equilibrium.
    
    Args:
        price_pair: Prices for each agent
        params: Simulation parameters
        tol: Tolerance for comparison
        
    Returns:
        True if prices match theoretical cooperative equilibrium within tolerance
    """
    prices = array(price_pair)
    theoretical_coop = compute_theoretical_coop_price(params)
    
    # For symmetric games, all agents should have same price
    return all(abs(price - theoretical_coop) <= tol for price in prices)


def compute_nash_distance(price_pair: Union[List[float], np.ndarray],
                         params: SimParams) -> float:
    """
    Compute distance from Nash equilibrium.
    
    Args:
        price_pair: Prices for each agent
        params: Simulation parameters
        
    Returns:
        Euclidean distance from theoretical Nash equilibrium
    """
    prices = array(price_pair)
    theoretical_nash = compute_theoretical_nash_price(params)
    nash_vector = array([theoretical_nash] * len(prices))
    
    return float(np.linalg.norm(prices - nash_vector))


def compute_coop_distance(price_pair: Union[List[float], np.ndarray],
                         params: SimParams) -> float:
    """
    Compute distance from cooperative equilibrium.
    
    Args:
        price_pair: Prices for each agent
        params: Simulation parameters
        
    Returns:
        Euclidean distance from theoretical cooperative equilibrium
    """
    prices = array(price_pair)
    theoretical_coop = compute_theoretical_coop_price(params)
    coop_vector = array([theoretical_coop] * len(prices))
    
    return float(np.linalg.norm(prices - coop_vector))


def classify_equilibrium(price_pair: Union[List[float], np.ndarray],
                        params: SimParams, tol: float = 1e-4) -> str:
    """
    Classify the type of equilibrium (if any).
    
    Args:
        price_pair: Prices for each agent
        params: Simulation parameters
        tol: Tolerance for equilibrium checks
        
    Returns:
        String describing equilibrium type: 'nash', 'cooperative', 'both', 'neither'
    """
    is_nash = check_theoretical_nash(price_pair, params, tol)
    is_coop = check_theoretical_coop(price_pair, params, tol)
    
    if is_nash and is_coop:
        return 'both'
    elif is_nash:
        return 'nash'
    elif is_coop:
        return 'cooperative'
    else:
        return 'neither'


def analyze_equilibrium_properties(price_pair: Union[List[float], np.ndarray],
                                  params: SimParams) -> dict:
    """
    Comprehensive analysis of equilibrium properties.
    
    Args:
        price_pair: Prices for each agent
        params: Simulation parameters
        
    Returns:
        Dictionary with equilibrium analysis results
    """
    prices = array(price_pair)
    
    # Basic equilibrium checks
    is_nash_theoretical = check_theoretical_nash(prices, params)
    is_coop_theoretical = check_theoretical_coop(prices, params)
    is_nash_behavioral = check_nash(prices, params)
    is_coop_behavioral = check_coop(prices, params)
    
    # Distance measures
    nash_dist = compute_nash_distance(prices, params)
    coop_dist = compute_coop_distance(prices, params)
    
    # Efficiency measures
    demands = compute_demands_from_prices(prices, params)
    profits = calc_profit_vector(prices, zeros(len(prices)), demands)
    total_welfare = np.sum(profits)
    
    # Theoretical benchmarks
    nash_price = compute_theoretical_nash_price(params)
    coop_price = compute_theoretical_coop_price(params)
    
    nash_vector = array([nash_price] * len(prices))
    coop_vector = array([coop_price] * len(prices))
    
    nash_demands = compute_demands_from_prices(nash_vector, params)
    coop_demands = compute_demands_from_prices(coop_vector, params)
    
    nash_profits = calc_profit_vector(nash_vector, zeros(len(prices)), nash_demands)
    coop_profits = calc_profit_vector(coop_vector, zeros(len(prices)), coop_demands)
    
    nash_welfare = np.sum(nash_profits)
    coop_welfare = np.sum(coop_profits)
    
    # Efficiency ratio
    if abs(coop_welfare - nash_welfare) > 1e-10:
        efficiency_ratio = (total_welfare - nash_welfare) / (coop_welfare - nash_welfare)
    else:
        efficiency_ratio = 1.0 if abs(total_welfare - nash_welfare) < 1e-10 else 0.0
    
    return {
        'equilibrium_type': classify_equilibrium(prices, params),
        'is_nash_theoretical': is_nash_theoretical,
        'is_coop_theoretical': is_coop_theoretical,
        'is_nash_behavioral': is_nash_behavioral,
        'is_coop_behavioral': is_coop_behavioral,
        'nash_distance': nash_dist,
        'coop_distance': coop_dist,
        'total_welfare': total_welfare,
        'nash_welfare': nash_welfare,
        'coop_welfare': coop_welfare,
        'efficiency_ratio': efficiency_ratio,
        'welfare_loss_vs_coop': coop_welfare - total_welfare,
        'welfare_gain_vs_nash': total_welfare - nash_welfare,
        'prices': prices.tolist(),
        'profits': profits.tolist(),
        'demands': demands.tolist()
    }


def validate_equilibrium_theory(params: SimParams, tol: float = 1e-6) -> dict:
    """
    Validate theoretical equilibrium calculations.
    
    Args:
        params: Simulation parameters
        tol: Tolerance for validation
        
    Returns:
        Dictionary with validation results
    """
    # Get theoretical equilibria
    nash_price = compute_theoretical_nash_price(params)
    coop_price = compute_theoretical_coop_price(params)
    
    # Test Nash equilibrium
    nash_vector = [nash_price] * params.n_agents
    nash_valid = check_nash(nash_vector, params, tol)
    
    # Test cooperative equilibrium
    coop_vector = [coop_price] * params.n_agents
    coop_valid = check_coop(coop_vector, params, tol)
    
    # Cross-check: Nash should not be cooperative (unless coincidental)
    nash_is_coop = check_coop(nash_vector, params, tol)
    coop_is_nash = check_nash(coop_vector, params, tol)
    
    return {
        'nash_price': nash_price,
        'coop_price': coop_price,
        'nash_equilibrium_valid': nash_valid,
        'coop_equilibrium_valid': coop_valid,
        'nash_is_also_coop': nash_is_coop,
        'coop_is_also_nash': coop_is_nash,
        'equilibria_coincide': abs(nash_price - coop_price) < tol
    }


def check_price_symmetry(price_pair: Union[List[float], np.ndarray], 
                        tol: float = 1e-6) -> bool:
    """
    Check if prices are symmetric across agents.
    
    Args:
        price_pair: Prices for each agent
        tol: Tolerance for symmetry check
        
    Returns:
        True if all prices are equal within tolerance
    """
    prices = array(price_pair)
    
    if len(prices) <= 1:
        return True
    
    mean_price = np.mean(prices)
    return all(abs(price - mean_price) <= tol for price in prices)


def compute_price_dispersion(price_pair: Union[List[float], np.ndarray]) -> float:
    """
    Compute price dispersion measure.
    
    Args:
        price_pair: Prices for each agent
        
    Returns:
        Coefficient of variation of prices
    """
    prices = array(price_pair)
    
    if len(prices) <= 1:
        return 0.0
    
    mean_price = np.mean(prices)
    std_price = np.std(prices)
    
    if abs(mean_price) > 1e-10:
        return std_price / mean_price
    else:
        return 0.0 if std_price < 1e-10 else float('inf') 