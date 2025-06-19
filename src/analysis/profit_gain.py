"""
Profit gain analysis module.

Implements profit calculations and comparisons with theoretical benchmarks.
Based on the paper's profit analysis and equilibrium comparisons (Section 4.2).

References:
    - Section 4.2: Profit analysis and welfare implications
    - Equation (5): Profit function definition
    - Table 2: Profit comparisons across different strategies
"""

import numpy as np
from typing import List, Union, Optional, Dict, Any, Callable
from scipy.optimize import minimize_scalar

# Handle imports for both package and standalone usage
try:
    from ..dtype_policy import DTYPE, array, zeros
    from ..params import SimParams
    from ..convergence import compute_theoretical_nash_price, compute_theoretical_coop_price
except ImportError:
    try:
        from dtype_policy import DTYPE, array, zeros
        from params import SimParams
        from convergence import compute_theoretical_nash_price, compute_theoretical_coop_price
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
        
        def compute_theoretical_nash_price(params):
            return 0.5  # Dummy value
        
        def compute_theoretical_coop_price(params):
            return 0.5  # Dummy value


def calc_profit(price: float, cost: float, demand: float) -> float:
    """
    Calculate single agent profit.
    
    Args:
        price: Price set by the agent
        cost: Marginal cost (typically 0 in the paper)
        demand: Demand quantity for this agent
        
    Returns:
        Profit = (price - cost) * demand
    """
    return DTYPE((price - cost) * demand)


def calc_profit_vector(prices: np.ndarray, costs: np.ndarray, demands: np.ndarray) -> np.ndarray:
    """
    Calculate profits for multiple agents.
    
    Args:
        prices: Price vector for all agents
        costs: Cost vector for all agents
        demands: Demand vector for all agents
        
    Returns:
        Profit vector for all agents
    """
    return array([(p - c) * d for p, c, d in zip(prices, costs, demands)])


def compute_nash_profits(params: SimParams, costs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute theoretical Nash equilibrium profits.
    
    Args:
        params: Simulation parameters
        costs: Cost vector (defaults to zero costs)
        
    Returns:
        Nash equilibrium profits for each agent
    """
    if costs is None:
        costs = zeros(params.n_agents)
    
    # Get Nash equilibrium price
    nash_price = compute_theoretical_nash_price(params)
    nash_prices = array([nash_price] * params.n_agents)
    
    # Compute demands at Nash prices
    demands = compute_demands_from_prices(nash_prices, params)
    
    # Compute profits
    nash_profits = calc_profit_vector(nash_prices, costs, demands)
    
    return nash_profits


def compute_coop_profits(params: SimParams, costs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute theoretical cooperative equilibrium profits.
    
    Args:
        params: Simulation parameters
        costs: Cost vector (defaults to zero costs)
        
    Returns:
        Cooperative equilibrium profits for each agent
    """
    if costs is None:
        costs = zeros(params.n_agents)
    
    # Get cooperative equilibrium price
    coop_price = compute_theoretical_coop_price(params)
    coop_prices = array([coop_price] * params.n_agents)
    
    # Compute demands at cooperative prices
    demands = compute_demands_from_prices(coop_prices, params)
    
    # Compute profits
    coop_profits = calc_profit_vector(coop_prices, costs, demands)
    
    return coop_profits


def compute_random_profits(params: SimParams, n_samples: int = 1000, 
                          costs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute expected profits under random pricing strategy.
    
    Args:
        params: Simulation parameters
        n_samples: Number of random price samples
        costs: Cost vector (defaults to zero costs)
        
    Returns:
        Expected profits under random pricing
    """
    if costs is None:
        costs = zeros(params.n_agents)
    
    # Generate random prices uniformly from [0, 1]
    total_profits = zeros(params.n_agents)
    
    for _ in range(n_samples):
        random_prices = np.random.uniform(0, 1, params.n_agents)
        demands = compute_demands_from_prices(random_prices, params)
        profits = calc_profit_vector(random_prices, costs, demands)
        total_profits += profits
    
    # Return average profits
    return total_profits / n_samples


def gain_vs_nash(profits: Union[np.ndarray, List[float]], params: SimParams) -> np.ndarray:
    """
    Calculate profit gain relative to Nash equilibrium (%).
    
    Args:
        profits: Observed profits for each agent
        params: Simulation parameters for Nash calculation
        
    Returns:
        Percentage gain relative to Nash profits for each agent
    """
    profits = array(profits)
    nash_profits = compute_nash_profits(params)
    
    # Handle division by zero
    gains = zeros(len(profits))
    for i, (profit, nash_profit) in enumerate(zip(profits, nash_profits)):
        if abs(nash_profit) > 1e-10:
            gains[i] = ((profit - nash_profit) / nash_profit) * 100.0
        else:
            gains[i] = 0.0 if abs(profit) < 1e-10 else float('inf')
    
    return gains


def gain_vs_random(profits: Union[np.ndarray, List[float]], params: SimParams) -> np.ndarray:
    """
    Calculate profit gain relative to random pricing strategy (%).
    
    Args:
        profits: Observed profits for each agent
        params: Simulation parameters for random strategy calculation
        
    Returns:
        Percentage gain relative to random strategy profits for each agent
    """
    profits = array(profits)
    random_profits = compute_random_profits(params)
    
    # Handle division by zero
    gains = zeros(len(profits))
    for i, (profit, random_profit) in enumerate(zip(profits, random_profits)):
        if abs(random_profit) > 1e-10:
            gains[i] = ((profit - random_profit) / random_profit) * 100.0
        else:
            gains[i] = 0.0 if abs(profit) < 1e-10 else float('inf')
    
    return gains


def gain_vs_coop(profits: Union[np.ndarray, List[float]], params: SimParams) -> np.ndarray:
    """
    Calculate profit gain relative to cooperative equilibrium (%).
    
    Args:
        profits: Observed profits for each agent
        params: Simulation parameters for cooperative calculation
        
    Returns:
        Percentage gain relative to cooperative profits for each agent
    """
    profits = array(profits)
    coop_profits = compute_coop_profits(params)
    
    # Handle division by zero
    gains = zeros(len(profits))
    for i, (profit, coop_profit) in enumerate(zip(profits, coop_profits)):
        if abs(coop_profit) > 1e-10:
            gains[i] = ((profit - coop_profit) / coop_profit) * 100.0
        else:
            gains[i] = 0.0 if abs(profit) < 1e-10 else float('inf')
    
    return gains


def compute_demands_from_prices(prices: np.ndarray, params: SimParams) -> np.ndarray:
    """
    Compute demand quantities from prices using the specified demand model.
    
    Args:
        prices: Price vector
        params: Simulation parameters
        
    Returns:
        Demand vector
    """
    # Use logit demand model (matching the paper)
    try:
        from ..PI_routines import compute_demands_logit
    except ImportError:
        try:
            from PI_routines import compute_demands_logit
        except ImportError:
            # Fallback: simple linear demand
            def compute_demands_logit(price_list, params):
                # Simple linear demand: q_i = a - b*p_i + c*sum(p_j for j != i)
                demands = []
                for i, price in enumerate(price_list):
                    other_prices = [p for j, p in enumerate(price_list) if j != i]
                    demand = params.a_param - price + params.lambda_param * sum(other_prices)
                    demands.append(max(0, demand))  # Non-negative demand
                return demands
    
    price_list = prices.tolist()
    demands = compute_demands_logit(price_list, params)
    
    return array(demands)


def analyze_profit_distribution(profit_series: List[np.ndarray], params: SimParams) -> Dict[str, Any]:
    """
    Analyze distribution of profits across multiple runs.
    
    Args:
        profit_series: List of profit vectors from different runs
        params: Simulation parameters
        
    Returns:
        Dictionary with profit distribution analysis
    """
    if not profit_series:
        return {'error': 'No profit data provided'}
    
    # Convert to array
    profits_array = np.array(profit_series)
    
    # Basic statistics
    mean_profits = np.mean(profits_array, axis=0)
    std_profits = np.std(profits_array, axis=0)
    min_profits = np.min(profits_array, axis=0)
    max_profits = np.max(profits_array, axis=0)
    
    # Benchmark comparisons
    nash_profits = compute_nash_profits(params)
    coop_profits = compute_coop_profits(params)
    random_profits = compute_random_profits(params)
    
    # Gain calculations
    mean_nash_gains = gain_vs_nash(mean_profits, params)
    mean_coop_gains = gain_vs_coop(mean_profits, params)
    mean_random_gains = gain_vs_random(mean_profits, params)
    
    # Welfare analysis
    total_welfare = np.sum(mean_profits)
    nash_welfare = np.sum(nash_profits)
    coop_welfare = np.sum(coop_profits)
    
    welfare_vs_nash = ((total_welfare - nash_welfare) / nash_welfare) * 100.0 if nash_welfare > 0 else 0.0
    welfare_vs_coop = ((total_welfare - coop_welfare) / coop_welfare) * 100.0 if coop_welfare > 0 else 0.0
    
    return {
        'mean_profits': mean_profits.tolist(),
        'std_profits': std_profits.tolist(),
        'min_profits': min_profits.tolist(),
        'max_profits': max_profits.tolist(),
        'nash_profits': nash_profits.tolist(),
        'coop_profits': coop_profits.tolist(),
        'random_profits': random_profits.tolist(),
        'nash_gains_percent': mean_nash_gains.tolist(),
        'coop_gains_percent': mean_coop_gains.tolist(),
        'random_gains_percent': mean_random_gains.tolist(),
        'total_welfare': total_welfare,
        'nash_welfare': nash_welfare,
        'coop_welfare': coop_welfare,
        'welfare_gain_vs_nash_percent': welfare_vs_nash,
        'welfare_gain_vs_coop_percent': welfare_vs_coop
    }


def compute_efficiency_ratio(profits: np.ndarray, params: SimParams) -> float:
    """
    Compute efficiency ratio relative to first-best (cooperative) outcome.
    
    Args:
        profits: Observed profits
        params: Simulation parameters
        
    Returns:
        Efficiency ratio (0 = Nash, 1 = Cooperative)
    """
    nash_profits = compute_nash_profits(params)
    coop_profits = compute_coop_profits(params)
    
    total_observed = np.sum(profits)
    total_nash = np.sum(nash_profits)
    total_coop = np.sum(coop_profits)
    
    if abs(total_coop - total_nash) > 1e-10:
        efficiency = (total_observed - total_nash) / (total_coop - total_nash)
        return max(0.0, min(1.0, efficiency))  # Clamp to [0, 1]
    else:
        return 1.0 if abs(total_observed - total_nash) < 1e-10 else 0.0


def profit_decomposition(profits: np.ndarray, prices: np.ndarray, params: SimParams) -> Dict[str, Any]:
    """
    Decompose profits into price and demand effects.
    
    Args:
        profits: Observed profits
        prices: Observed prices
        params: Simulation parameters
        
    Returns:
        Dictionary with profit decomposition
    """
    demands = compute_demands_from_prices(prices, params)
    
    # Compute counterfactual profits at Nash prices with observed demands
    nash_price = compute_theoretical_nash_price(params)
    nash_prices = array([nash_price] * params.n_agents)
    counterfactual_profits = calc_profit_vector(nash_prices, zeros(params.n_agents), demands)
    
    # Decomposition
    price_effect = profits - counterfactual_profits
    demand_effect = counterfactual_profits - compute_nash_profits(params)
    
    return {
        'total_profit_effect': profits - compute_nash_profits(params),
        'price_effect': price_effect,
        'demand_effect': demand_effect,
        'observed_profits': profits.tolist(),
        'observed_prices': prices.tolist(),
        'observed_demands': demands.tolist()
    } 