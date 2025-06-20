"""
Cooperative benchmark calculation module.

Implements exact calculation of cooperative (collusive) equilibrium
for comparison with Q-learning results.
"""

import numpy as np
from typing import List, Dict
from scipy.optimize import minimize_scalar
from params import SimParams, load_config

# Handle imports for both package and standalone usage
try:
    from .params import SimParams
    from .profit import DemandParams, compute_demands_logit, calc_profit
    from .dtype_policy import DTYPE, array
except ImportError:
    from params import SimParams
    from profit import DemandParams, compute_demands_logit, calc_profit


def compute_cooperative_price_exact(demand_params: DemandParams, 
                                  n_agents: int = 2) -> float:
    """
    Compute exact cooperative (collusive) price by solving joint profit maximization.
    
    For logit demand: max π = Σ_i (p_i - c) * D_i(p)
    where D_i(p) = exp(u_i/μ) / Σ_j exp(u_j/μ)
    and u_i = a0 - a*p_i + c
    
    Args:
        demand_params: Demand parameters
        n_agents: Number of agents
        
    Returns:
        Cooperative equilibrium price (symmetric)
    """
    def joint_profit_negative(price: float) -> float:
        """Negative joint profit for minimization."""
        # Ensure float conversion for array operations
        price = float(price)
        prices = np.asarray([price] * n_agents, dtype=float)
        demands = compute_demands_logit(prices.tolist(), demand_params)
        demands = np.asarray(demands, dtype=float)
        
        total_profit = 0.0
        for i in range(n_agents):
            profit = calc_profit(float(prices[i]), float(demand_params.c), float(demands[i]))
            total_profit += float(profit)
        
        return -float(total_profit)  # Negative for minimization
    
    # Solve for optimal symmetric price
    result = minimize_scalar(
        joint_profit_negative,
        bounds=(float(demand_params.c), 2.0),  # Reasonable bounds
        method='bounded'
    )
    
    return float(result.x)


def compute_nash_price_exact(demand_params: DemandParams, 
                           n_agents: int = 2) -> float:
    """
    Compute exact Nash equilibrium price by solving best response conditions.
    
    For symmetric logit demand, solve first-order condition:
    ∂π_i/∂p_i = D_i + (p_i - c) * ∂D_i/∂p_i = 0
    
    Args:
        demand_params: Demand parameters
        n_agents: Number of agents
        
    Returns:
        Nash equilibrium price (symmetric)
    """
    def best_response_condition(price: float) -> float:
        """First-order condition for Nash equilibrium."""
        price = float(price)
        prices = np.asarray([price] * n_agents, dtype=float)
        
        # Compute demand and its derivative
        demands = compute_demands_logit(prices.tolist(), demand_params)
        demands = np.asarray(demands, dtype=float)
        demand_i = float(demands[0])  # Symmetric case
        
        # Compute demand derivative (approximate with finite difference)
        epsilon = 1e-6
        prices_plus = prices.copy()
        prices_plus[0] += epsilon
        demands_plus = compute_demands_logit(prices_plus.tolist(), demand_params)
        demands_plus = np.asarray(demands_plus, dtype=float)
        demand_derivative = float((demands_plus[0] - demand_i) / epsilon)
        
        # First-order condition
        foc = demand_i + (price - float(demand_params.c)) * demand_derivative
        
        return float(foc)
    
    # Solve first-order condition
    from scipy.optimize import fsolve
    nash_price = fsolve(best_response_condition, x0=0.5)[0]
    
    return float(nash_price)


def compute_cooperation_gap(observed_profit: float, 
                          nash_profit: float, 
                          coop_profit: float) -> float:
    """
    Compute cooperation gap: (π_observed - π_nash) / (π_coop - π_nash)
    
    Args:
        observed_profit: Observed profit from simulation
        nash_profit: Theoretical Nash profit
        coop_profit: Theoretical cooperative profit
        
    Returns:
        Cooperation gap (0 = Nash, 1 = full cooperation)
    """
    # Ensure all inputs are floats
    observed_profit = float(observed_profit)
    nash_profit = float(nash_profit)
    coop_profit = float(coop_profit)
    
    if abs(coop_profit - nash_profit) < 1e-10:
        return 0.0  # Avoid division by zero
    
    gap = (observed_profit - nash_profit) / (coop_profit - nash_profit)
    
    # Clamp to reasonable range
    return float(max(0.0, min(2.0, gap)))


def coop_price_exact(demand_params: DemandParams, n_agents: int = 2) -> float:
    """
    Compute exact cooperative price for linear demand model.
    
    For linear demand D(p) = a0 - a*p, cooperative price is:
    p_coll = (a0 + c) / 2
    
    Args:
        demand_params: Demand parameters
        n_agents: Number of agents (for compatibility)
        
    Returns:
        Exact cooperative price
    """
    if demand_params.mu != 0.0:
        # For logit demand, use numerical optimization
        return compute_cooperative_price_exact(demand_params, n_agents)
    
    # For linear demand (mu = 0), use exact formula
    # D(p) = a0 - a*p, joint profit maximization gives:
    # p_coll = (a0 + c) / 2
    a0 = float(demand_params.a0)
    c = float(demand_params.c)
    
    coop_price = (a0 + c) / 2.0
    
    return float(coop_price)


def compute_cooperation_gap_exact(observed_profit: float,
                                demand_params: DemandParams,
                                n_agents: int = 2) -> float:
    """
    Compute cooperation gap using exact theoretical benchmarks.
    
    Args:
        observed_profit: Observed profit from simulation
        demand_params: Demand parameters
        n_agents: Number of agents
        
    Returns:
        Cooperation gap computed with exact benchmarks
    """
    # Use exact formulas
    nash_price = compute_nash_price_exact(demand_params, n_agents)
    coop_price = coop_price_exact(demand_params, n_agents)
    
    # Compute corresponding profits
    nash_prices = np.asarray([nash_price] * n_agents, dtype=float)
    coop_prices = np.asarray([coop_price] * n_agents, dtype=float)
    
    nash_demands = compute_demands_logit(nash_prices.tolist(), demand_params)
    coop_demands = compute_demands_logit(coop_prices.tolist(), demand_params)
    
    nash_demands = np.asarray(nash_demands, dtype=float)
    coop_demands = np.asarray(coop_demands, dtype=float)
    
    nash_profit = calc_profit(float(nash_price), float(demand_params.c), float(nash_demands[0]))
    coop_profit = calc_profit(float(coop_price), float(demand_params.c), float(coop_demands[0]))
    
    return compute_cooperation_gap(observed_profit, nash_profit, coop_profit)


def compute_theoretical_benchmarks(demand_params: DemandParams, 
                                 n_agents: int = 2) -> Dict[str, float]:
    """
    Compute all theoretical benchmarks for comparison.
    
    Args:
        demand_params: Demand parameters
        n_agents: Number of agents
        
    Returns:
        Dictionary with theoretical prices and profits
    """
    # Compute equilibrium prices
    nash_price = compute_nash_price_exact(demand_params, n_agents)
    coop_price = compute_cooperative_price_exact(demand_params, n_agents)
    
    # Compute profits at equilibria
    nash_prices = np.asarray([nash_price] * n_agents, dtype=float)
    coop_prices = np.asarray([coop_price] * n_agents, dtype=float)
    
    nash_demands = compute_demands_logit(nash_prices.tolist(), demand_params)
    coop_demands = compute_demands_logit(coop_prices.tolist(), demand_params)
    
    nash_demands = np.asarray(nash_demands, dtype=float)
    coop_demands = np.asarray(coop_demands, dtype=float)
    
    nash_profit = calc_profit(float(nash_price), float(demand_params.c), float(nash_demands[0]))
    coop_profit = calc_profit(float(coop_price), float(demand_params.c), float(coop_demands[0]))
    
    return {
        'nash_price': float(nash_price),
        'coop_price': float(coop_price),
        'nash_profit': float(nash_profit),
        'coop_profit': float(coop_profit),
        'nash_demands': nash_demands.tolist(),
        'coop_demands': coop_demands.tolist()
    }


def analyze_cooperation_gap_detailed(observed_results: List[Dict], 
                                   demand_params: DemandParams) -> Dict[str, float]:
    """
    Detailed cooperation gap analysis for multiple simulation runs.
    
    Args:
        observed_results: List of simulation result dictionaries
        demand_params: Demand parameters
        
    Returns:
        Dictionary with cooperation gap statistics
    """
    # Compute theoretical benchmarks
    benchmarks = compute_theoretical_benchmarks(demand_params)
    
    cooperation_gaps = []
    observed_profits = []
    
    for result in observed_results:
        if 'final_profits' in result:
            # Ensure array conversion for profit calculation
            final_profits = np.asarray(result['final_profits'], dtype=float)
            mean_profit = float(np.mean(final_profits))
            observed_profits.append(mean_profit)
            
            # Compute cooperation gap
            gap = compute_cooperation_gap(
                mean_profit, 
                benchmarks['nash_profit'],
                benchmarks['coop_profit']
            )
            cooperation_gaps.append(float(gap))
    
    if not cooperation_gaps:
        return {}
    
    cooperation_gaps = np.asarray(cooperation_gaps, dtype=float)
    observed_profits = np.asarray(observed_profits, dtype=float)
    
    return {
        'mean_coop_gap': float(np.mean(cooperation_gaps)),
        'std_coop_gap': float(np.std(cooperation_gaps)),
        'min_coop_gap': float(np.min(cooperation_gaps)),
        'max_coop_gap': float(np.max(cooperation_gaps)),
        'theoretical_nash_profit': float(benchmarks['nash_profit']),
        'theoretical_coop_profit': float(benchmarks['coop_profit']),
        'observed_mean_profit': float(np.mean(observed_profits)),
        'observed_std_profit': float(np.std(observed_profits)) if len(observed_profits) > 1 else 0.0
    }


def demand_function(price1, price2, params):
    """
    Compute demand using logit or linear model from paper.
    
    Args:
        price1: Agent 1's price
        price2: Agent 2's price  
        params: SimParams object
        
    Returns:
        Tuple of (demand1, demand2)
    """
    if params.demand_model == "linear":
        # Linear demand model
        d1 = params.a0 - params.a * price1 + params.a * price2
        d2 = params.a0 - params.a * price2 + params.a * price1
        return max(0, d1), max(0, d2)
    
    elif params.demand_model == "logit":
        # Logit demand model (Equation 1 in paper)
        if params.mu == 0:
            # Linear demand as limit case when mu=0
            d1 = params.a0 - params.a * price1 + params.a * price2  
            d2 = params.a0 - params.a * price2 + params.a * price1
        else:
            # True logit model following Equation (1):
            # q_i = exp((a0 - a*p_i + μ*ln(p_j))/(1+μ)) / Σ_j exp((a0 - a*p_j + μ*ln(p_i))/(1+μ))
            
            # Ensure positive prices for log
            price1 = max(price1, 1e-6)
            price2 = max(price2, 1e-6)
            
            # Compute utilities with interaction terms
            u1 = (params.a0 - params.a * price1 + params.mu * np.log(price2)) / (1 + params.mu)
            u2 = (params.a0 - params.a * price2 + params.mu * np.log(price1)) / (1 + params.mu)
            
            exp1 = np.exp(u1)
            exp2 = np.exp(u2)
            exp0 = 1  # Outside option
            
            total = exp1 + exp2 + exp0
            d1 = exp1 / total
            d2 = exp2 / total
            
        return max(0, d1), max(0, d2)
    
    else:
        raise ValueError(f"Unknown demand model: {params.demand_model}")


def profit_function(price1, price2, params):
    """
    Compute profit for both agents.
    
    Args:
        price1: Agent 1's price
        price2: Agent 2's price
        params: SimParams object
        
    Returns:
        Tuple of (profit1, profit2)
    """
    d1, d2 = demand_function(price1, price2, params)
    
    profit1 = (price1 - params.c) * d1
    profit2 = (price2 - params.c) * d2
    
    return profit1, profit2


def compute_cooperative_benchmark(params=None, config_path="config.json"):
    """
    Compute cooperative (joint profit maximizing) benchmark.
    
    Args:
        params: SimParams object (optional)
        config_path: Path to config file (default: config.json)
        
    Returns:
        Dictionary with cooperative prices and profits
    """
    if params is None:
        config = load_config(config_path)
        params = SimParams(config)
    
    # Use actual price grid from configuration
    price_grid = np.array(params.price_grid)
    
    best_joint_profit = -np.inf
    best_prices = None
    best_profits = None
    
    for p1 in price_grid:
        for p2 in price_grid:
            profit1, profit2 = profit_function(p1, p2, params)
            joint_profit = profit1 + profit2
            
            if joint_profit > best_joint_profit:
                best_joint_profit = joint_profit
                best_prices = (p1, p2)
                best_profits = (profit1, profit2)
    
    return {
        'cooperative_prices': best_prices,
        'cooperative_profits': best_profits,
        'joint_profit': best_joint_profit
    }


def compute_nash_benchmark(params=None, config_path="config.json"):
    """
    Compute Nash equilibrium benchmark using best response iteration.
    
    Args:
        params: SimParams object (optional) 
        config_path: Path to config file (default: config.json)
        
    Returns:
        Dictionary with Nash prices and profits
    """
    if params is None:
        config = load_config(config_path)
        params = SimParams(config)
    
    # Use actual price grid from configuration
    price_grid = np.array(params.price_grid)
    
    # Initial guess - start at middle of price grid
    p1, p2 = price_grid[len(price_grid)//2], price_grid[len(price_grid)//2]
    
    # Best response iteration
    for iteration in range(1000):
        p1_old, p2_old = p1, p2
        
        # Best response for agent 1
        best_profit1 = -np.inf
        
        for p in price_grid:
            profit1, _ = profit_function(p, p2, params)
            if profit1 > best_profit1:
                best_profit1 = profit1
                p1 = p
        
        # Best response for agent 2
        best_profit2 = -np.inf
        for p in price_grid:
            _, profit2 = profit_function(p1, p, params)
            if profit2 > best_profit2:
                best_profit2 = profit2
                p2 = p
        
        # Check convergence
        if abs(p1 - p1_old) < 1e-6 and abs(p2 - p2_old) < 1e-6:
            break
    
    profit1, profit2 = profit_function(p1, p2, params)
    
    return {
        'nash_prices': (p1, p2),
        'nash_profits': (profit1, profit2),
        'iterations': iteration + 1
    }


def compute_monopoly_benchmark(params=None, config_path="config.json"):
    """
    Compute monopoly benchmark (single agent maximizing profit).
    
    Args:
        params: SimParams object (optional)
        config_path: Path to config file (default: config.json)
        
    Returns:
        Dictionary with monopoly price and profit
    """
    if params is None:
        config = load_config(config_path)
        params = SimParams(config)
    
    # Use actual price grid from configuration
    price_grid = np.array(params.price_grid)
    best_profit = -np.inf
    best_price = None
    
    for p in price_grid:
        # Monopolist sets same price, captures all demand
        d1, d2 = demand_function(p, p, params)
        total_demand = d1 + d2
        profit = (p - params.c) * total_demand
        
        if profit > best_profit:
            best_profit = profit
            best_price = p
    
    return {
        'monopoly_price': best_price,
        'monopoly_profit': best_profit
    }


if __name__ == "__main__":
    # Test with different configurations
    print("Computing benchmarks...")
    
    # Default configuration
    coop = compute_cooperative_benchmark()
    nash = compute_nash_benchmark()
    monopoly = compute_monopoly_benchmark()
    
    print(f"Cooperative: prices={coop['cooperative_prices']}, joint_profit={coop['joint_profit']}")
    print(f"Nash: prices={nash['nash_prices']}, profits={nash['nash_profits']}")
    print(f"Monopoly: price={monopoly['monopoly_price']}, profit={monopoly['monopoly_profit']}") 