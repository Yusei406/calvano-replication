import numpy as np
from typing import List, Tuple, Optional
from .globals import GlobalVars
from . import globals
from .generic_routines import RandomNumberGenerator
from .params import SimParams

def compute_pi_matrices_singh_vives(globals: GlobalVars) -> None:
    """
    Compute the Singh&Vives common payoff matrix PI
    """
    # Extract demand parameters
    gamma = globals.demand_parameters[0]
    extend = globals.demand_parameters[1:3]
    
    # 1. Compute repeated Nash profits
    globals.nash_market_shares = linear_demands(gamma, globals.nash_prices)
    globals.nash_profits = globals.nash_prices * globals.nash_market_shares
    
    # 2. Compute cooperation profits
    globals.coop_market_shares = linear_demands(gamma, globals.coop_prices)
    globals.coop_profits = globals.coop_prices * globals.coop_market_shares
    
    # 3. Compute price grid
    # Upper and lower bounds
    if np.all(extend > 0.0):
        # Lower bound = pNash - extend[0]*(pCoop - pNash)
        # Upper bound = pCoop + extend[1]*(pCoop - pNash)
        for i_agent in range(globals.num_agents):
            globals.prices_grids[0, i_agent] = max(0.0, globals.nash_prices[i_agent] - 
                extend[0] * (globals.coop_prices[i_agent] - globals.nash_prices[i_agent]))
            globals.prices_grids[-1, i_agent] = max(0.0, globals.coop_prices[i_agent] + 
                extend[1] * (globals.coop_prices[i_agent] - globals.nash_prices[i_agent]))
    
    elif extend[0] < 0.0 and extend[1] >= -np.finfo(float).eps:
        # Lower bound = 0
        # Upper bound = (1+extend[1])*pCoop
        for i_agent in range(globals.num_agents):
            globals.prices_grids[0, i_agent] = 0.0
            globals.prices_grids[-1, i_agent] = max(0.0, (1.0 + extend[1]) * globals.coop_prices[i_agent])
    
    # Grids
    step_prices = (globals.prices_grids[-1, :] - globals.prices_grids[0, :]) / (globals.num_prices - 1)
    for i in range(1, globals.num_prices - 1):
        globals.prices_grids[i, :] = globals.prices_grids[i-1, :] + step_prices
    
    # 4. Compute Pi matrices
    prices = np.zeros(globals.num_agents)
    for i in range(globals.num_actions):
        for j in range(globals.num_agents):
            prices[j] = globals.prices_grids[globals.index_actions[i, j] - 1, j]
        
        d = linear_demands(gamma, prices)
        globals.pi[i, :] = prices * d
    
    # 5. With linear demand, the repeated Nash prices do not necessarily belong to the
    # prices grid. Hence, the indexNashPrices vector is empty.
    globals.index_nash_prices = np.zeros(globals.num_agents, dtype=int)
    globals.index_coop_prices = np.zeros(globals.num_agents, dtype=int)

def compute_pi_matrices_logit(globals: GlobalVars) -> None:
    """
    Compute the Logit common payoff matrix PI
    """
    # Extract demand parameters
    a0 = globals.demand_parameters[0]
    a = globals.demand_parameters[1:1+globals.num_agents]
    c = globals.demand_parameters[1+globals.num_agents:1+2*globals.num_agents]
    mu = globals.demand_parameters[1+2*globals.num_agents]
    extend = globals.demand_parameters[2+2*globals.num_agents:3+2*globals.num_agents]
    
    # 1. Compute repeated Nash profits
    globals.nash_market_shares = logit_demands(a0, a, c, mu, globals.nash_prices)
    globals.nash_profits = (globals.nash_prices - c) * globals.nash_market_shares
    
    # 2. Compute cooperation profits
    globals.coop_market_shares = logit_demands(a0, a, c, mu, globals.coop_prices)
    globals.coop_profits = (globals.coop_prices - c) * globals.coop_market_shares
    
    # 3. Compute price grid
    # Upper and lower bounds
    if np.all(extend > 0.0):
        # Lower bound = pNash - extend[0]*(pCoop - pNash)
        # Upper bound = pCoop + extend[1]*(pCoop - pNash)
        globals.prices_grids[0, :] = globals.nash_prices - extend[0] * (globals.coop_prices - globals.nash_prices)
        globals.prices_grids[-1, :] = globals.coop_prices + extend[1] * (globals.coop_prices - globals.nash_prices)
    
    elif extend[0] < 0.0 and extend[1] >= -np.finfo(float).eps:
        # Lower bound = cost + extend[0]*cost
        # Upper bound = pCoop + extend[1]*(CoopPrices-NashPrices)
        globals.prices_grids[0, :] = c + extend[0] * c
        globals.prices_grids[-1, :] = globals.coop_prices + extend[1] * (globals.coop_prices - globals.nash_prices)
    
    # Grids
    step_prices = (globals.prices_grids[-1, :] - globals.prices_grids[0, :]) / (globals.num_prices - 1)
    for i in range(1, globals.num_prices - 1):
        globals.prices_grids[i, :] = globals.prices_grids[i-1, :] + step_prices
    
    # 4. Compute Pi matrices
    prices = np.zeros(globals.num_agents)
    for i in range(globals.num_actions):
        for j in range(globals.num_agents):
            prices[j] = globals.prices_grids[globals.index_actions[i, j] - 1, j]
        
        d = logit_demands(a0, a, c, mu, prices)
        globals.pi[i, :] = (prices - c) * d
    
    # 5. With logit demand, the repeated Nash prices do not necessarily belong to the
    # prices grid. Hence, the indexNashPrices vector is empty.
    globals.index_nash_prices = np.zeros(globals.num_agents, dtype=int)
    globals.index_coop_prices = np.zeros(globals.num_agents, dtype=int)

def compute_pi_matrices_logit_mu0(globals: GlobalVars) -> None:
    """
    Compute the Logit common payoff matrix PI with mu = 0
    """
    # Extract demand parameters
    a0 = globals.demand_parameters[0]
    a = globals.demand_parameters[1:1+globals.num_agents]
    c = globals.demand_parameters[1+globals.num_agents:1+2*globals.num_agents]
    extend = globals.demand_parameters[2+2*globals.num_agents:3+2*globals.num_agents]
    
    # 1. Compute repeated Nash profits
    globals.nash_market_shares = logit_demands_mu0(a0, a, c, globals.nash_prices)
    globals.nash_profits = (globals.nash_prices - c) * globals.nash_market_shares
    
    # 2. Compute cooperation profits
    globals.coop_market_shares = logit_demands_mu0(a0, a, c, globals.coop_prices)
    globals.coop_profits = (globals.coop_prices - c) * globals.coop_market_shares
    
    # 3. Compute price grid
    # Upper and lower bounds
    if np.all(extend > 0.0):
        # Lower bound = pNash - extend[0]*(pCoop - pNash)
        # Upper bound = pCoop + extend[1]*(pCoop - pNash)
        globals.prices_grids[0, :] = globals.nash_prices - extend[0] * (globals.coop_prices - globals.nash_prices)
        globals.prices_grids[-1, :] = globals.coop_prices + extend[1] * (globals.coop_prices - globals.nash_prices)
    
    elif extend[0] < 0.0 and extend[1] >= -np.finfo(float).eps:
        # Lower bound = cost + extend[0]*cost
        # Upper bound = pCoop + extend[1]*(CoopPrices-NashPrices)
        globals.prices_grids[0, :] = c + extend[0] * c
        globals.prices_grids[-1, :] = globals.coop_prices + extend[1] * (globals.coop_prices - globals.nash_prices)
    
    # Grids
    step_prices = (globals.prices_grids[-1, :] - globals.prices_grids[0, :]) / (globals.num_prices - 1)
    for i in range(1, globals.num_prices - 1):
        globals.prices_grids[i, :] = globals.prices_grids[i-1, :] + step_prices
    
    # 4. Compute Pi matrices
    prices = np.zeros(globals.num_agents)
    for i in range(globals.num_actions):
        for j in range(globals.num_agents):
            prices[j] = globals.prices_grids[globals.index_actions[i, j] - 1, j]
        
        d = logit_demands_mu0(a0, a, c, prices)
        globals.pi[i, :] = (prices - c) * d
    
    # 5. With logit demand, the repeated Nash prices do not necessarily belong to the
    # prices grid. Hence, the indexNashPrices vector is empty.
    globals.index_nash_prices = np.zeros(globals.num_agents, dtype=int)
    globals.index_coop_prices = np.zeros(globals.num_agents, dtype=int)

def linear_demands(gamma: float, prices: np.ndarray) -> np.ndarray:
    """Compute linear demands"""
    num_agents = len(prices)
    d = np.zeros(num_agents)
    
    for i in range(num_agents):
        d[i] = 1.0 - prices[i] + gamma * np.sum(prices) / num_agents
    
    return d

def logit_demands(a0: float, a: np.ndarray, c: np.ndarray, mu: float, prices: np.ndarray) -> np.ndarray:
    """Compute logit demands"""
    num_agents = len(prices)
    d = np.zeros(num_agents)
    
    # Compute denominator
    den = a0
    for i in range(num_agents):
        den += np.exp((a[i] - prices[i]) / mu)
    
    # Compute demands
    for i in range(num_agents):
        d[i] = np.exp((a[i] - prices[i]) / mu) / den
    
    return d

def logit_demands_mu0(a0: float, a: np.ndarray, c: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Compute logit demands with mu = 0"""
    num_agents = len(prices)
    d = np.zeros(num_agents)
    
    # Find best product
    best_idx = np.argmax(a - prices)
    
    # Set demand to 1 for best product, 0 for others
    d[best_idx] = 1.0
    
    return d

def run_policy_iteration(
    Q_matrices: List[np.ndarray],
    strategies: List[np.ndarray],
    delta: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Run policy iteration algorithm.
    
    Args:
        Q_matrices: List of Q matrices for each agent
        strategies: List of strategies for each agent
        delta: Discount factor
        
    Returns:
        Tuple containing:
        - Updated Q matrices
        - Updated strategies
    """
    # Initialize random number generator
    rng = RandomNumberGenerator()
    
    # Policy iteration loop
    while True:
        # Policy evaluation
        Q_matrices = evaluate_policy(Q_matrices, strategies, delta)
        
        # Policy improvement
        new_strategies = improve_policy(Q_matrices)
        
        # Check convergence
        if all(np.array_equal(s1, s2) for s1, s2 in zip(strategies, new_strategies)):
            break
        
        strategies = new_strategies
    
    return Q_matrices, strategies

def evaluate_policy(
    Q_matrices: List[np.ndarray],
    strategies: List[np.ndarray],
    delta: float
) -> List[np.ndarray]:
    """Evaluate current policy."""
    n_states = Q_matrices[0].shape[0]
    n_actions = Q_matrices[0].shape[1]
    
    # Initialize new Q matrices
    new_Q_matrices = [np.zeros((n_states, n_actions)) for _ in range(globals.n_agents)]
    
    # Evaluate policy for each state
    for state in range(n_states):
        # Get current actions
        actions = [strategy[state] for strategy in strategies]
        
        # Compute next state
        next_state = compute_state_number(actions)
        
        # Compute rewards
        rewards = compute_rewards(actions)
        
        # Update Q values
        for agent in range(globals.n_agents):
            new_Q_matrices[agent][state, actions[agent]] = (
                rewards[agent] + delta * np.max(Q_matrices[agent][next_state])
            )
    
    return new_Q_matrices

def improve_policy(Q_matrices: List[np.ndarray]) -> List[np.ndarray]:
    """Improve policy based on Q values."""
    n_states = Q_matrices[0].shape[0]
    new_strategies = [np.zeros(n_states, dtype=int) for _ in range(globals.n_agents)]
    
    # Improve policy for each state
    for state in range(n_states):
        for agent in range(globals.n_agents):
            new_strategies[agent][state] = np.argmax(Q_matrices[agent][state])
    
    return new_strategies

def compute_state_number(actions: List[int]) -> int:
    """Compute state number from actions."""
    state = 0
    for i, action in enumerate(actions):
        state += action * (globals.n_actions ** i)
    return state

def compute_rewards(actions: List[int]) -> List[float]:
    """Compute rewards for each agent."""
    # Convert actions to prices
    prices = [action / (globals.n_actions - 1) for action in actions]
    
    # Compute demands
    demands = compute_demands(prices)
    
    # Compute profits
    profits = [p * d for p, d in zip(prices, demands)]
    
    return profits

def compute_demands(prices: List[float]) -> List[float]:
    """Compute demands using logit model."""
    # Logit demand parameters
    a0 = 1.0
    a = 2.0
    c = 0.5
    mu = 0.1
    
    # Compute utilities
    utilities = [a0 - a * p + c for p in prices]
    
    # Compute demands
    exp_utilities = np.exp(np.array(utilities) / mu)
    total_exp = np.sum(exp_utilities)
    demands = exp_utilities / total_exp
    
    return demands.tolist()

def compute_demands_logit(price_list: List[float], params: SimParams) -> List[float]:
    """
    Compute demand quantities using logit demand model.
    
    Args:
        price_list: List of prices for all agents
        params: Simulation parameters
        
    Returns:
        List of demand quantities
    """
    n_agents = len(price_list)
    demands = []
    
    for i, price in enumerate(price_list):
        # Simple logit demand calculation
        other_prices = [price_list[j] for j in range(n_agents) if j != i]
        
        # Base demand with price sensitivity and cross-price effects
        base_demand = params.a_param - price
        cross_effect = params.lambda_param * sum(other_prices) / max(1, len(other_prices))
        
        demand = base_demand + cross_effect
        demand = max(0.0, demand)  # Non-negative demand
        demands.append(demand)
    
    return demands 