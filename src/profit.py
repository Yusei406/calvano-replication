"""
Unified profit calculation module.

This module provides the single authoritative implementation for profit calculations
in the Calvano Q-learning simulation. All other profit calculation functions are
deprecated in favor of these implementations.
"""

import numpy as np
from typing import List
from dataclasses import dataclass

try:
    from .params import SimParams
    from .dtype_policy import DTYPE, array
except ImportError:
    from params import SimParams
    from dtype_policy import DTYPE, array

@dataclass
class DemandParams:
    """Parameters for demand function calculation."""
    a0: float = 2.0      # Market size parameter
    a: float = 1.0       # Own-price sensitivity
    c: float = 0.0       # Marginal cost
    mu: float = 0.25     # Logit scaling parameter

def calc_profit(price: float, cost: float, demand: float) -> float:
    """The canonical profit calculation function."""
    return DTYPE((price - cost) * demand)

def calc_profit_vector(prices: np.ndarray, costs: np.ndarray, demands: np.ndarray) -> np.ndarray:
    """Calculate profits for multiple agents."""
    return array([(p - c) * d for p, c, d in zip(prices, costs, demands)])

def compute_demands_logit(prices: List[float], demand_params: DemandParams) -> List[float]:
    """
    Compute demands using logit demand model following Equation (1) from Calvano et al. (2020).
    
    Equation (1): q_i = exp((a0 - a*p_i + μ*ln(p_j))/(1+μ)) / Σ_j exp((a0 - a*p_j + μ*ln(p_i))/(1+μ))
    
    Args:
        prices: List of prices [p1, p2, ...]
        demand_params: DemandParams object
        
    Returns:
        List of demand quantities
    """
    n_agents = len(prices)
    if n_agents != 2:
        raise ValueError("This implementation assumes exactly 2 agents")
    
    p1, p2 = prices[0], prices[1]
    
    # Ensure positive prices for log calculation
    p1 = max(p1, 1e-6)
    p2 = max(p2, 1e-6)
    
    # Compute utilities with interaction terms (Equation 1)
    u1 = (demand_params.a0 - demand_params.a * p1 + demand_params.mu * np.log(p2)) / (1 + demand_params.mu)
    u2 = (demand_params.a0 - demand_params.a * p2 + demand_params.mu * np.log(p1)) / (1 + demand_params.mu)
    
    # Include outside option (normalized to 0)
    exp1 = np.exp(u1)
    exp2 = np.exp(u2)
    exp0 = 1.0  # Outside option
    
    total_exp = exp1 + exp2 + exp0
    
    # Market shares (demands)
    d1 = exp1 / total_exp
    d2 = exp2 / total_exp
    
    return [float(d1), float(d2)]

def get_demand_params_from_config(params: SimParams) -> DemandParams:
    """Extract demand parameters from simulation parameters."""
    return DemandParams(
        a0=getattr(params, "a0", 2.0),
        a=getattr(params, "a", 1.0), 
        c=getattr(params, "c", 0.0),
        mu=getattr(params, "mu", 0.25)
    )
