import numpy as np
from typing import List, Tuple, Optional, Any
from .params import SimParams
from .worker import get_worker_rng
from .dtype_policy import DTYPE, zeros, array

def init_q_matrices(
    params: SimParams,
    rng: Any = None
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Initialize Q matrices for all agents.
    
    Args:
        params: Simulation parameters
        rng: Random number generator (optional)
        
    Returns:
        Tuple containing:
        - Q matrices for each agent
        - Maximum Q values for each state and agent
        - Locations of maximum Q values
    """
    if rng is None:
        rng = get_worker_rng()
    
    q_matrices = []
    q_max = []
    q_max_loc = []
    
    for agent in range(params.n_agents):
        q_matrix = np.zeros((params.n_states, params.n_actions))
        q_max_agent = np.zeros(params.n_states)
        q_max_loc_agent = np.zeros(params.n_states, dtype=int)
        
        # Initialize Q values based on type (with fallback)
        q_init_type = getattr(params, 'q_init_type', 0)
        
        if q_init_type == 1:  # Random initialization
            for state in range(params.n_states):
                for action in range(params.n_actions):
                    if hasattr(rng, 'ran2'):
                        q_matrix[state, action] = rng.ran2()
                    else:
                        q_matrix[state, action] = np.random.random()
        elif q_init_type == 2:  # Nash initialization
            # Initialize to Nash equilibrium values
            for state in range(params.n_states):
                for action in range(params.n_actions):
                    q_matrix[state, action] = compute_nash_value(state, action, agent, params)
        elif q_init_type == 3:  # Cooperative initialization
            # Initialize to cooperative values
            for state in range(params.n_states):
                for action in range(params.n_actions):
                    q_matrix[state, action] = compute_coop_value(state, action, agent, params)
        else:
            # Default to zero initialization
            pass
        
        # Find maximum values and locations
        for state in range(params.n_states):
            max_val, max_loc = max_loc_break_ties(q_matrix[state, :], params)
            q_max_agent[state] = max_val
            q_max_loc_agent[state] = max_loc
        
        q_matrices.append(q_matrix)
        q_max.append(q_max_agent)
        q_max_loc.append(q_max_loc_agent)
    
    return q_matrices, q_max, q_max_loc

def init_state(params: SimParams) -> int:
    """Initialize state randomly."""
    try:
        rng = get_worker_rng()
        actions = [int(rng.ran2() * params.n_actions) for _ in range(params.n_agents)]
    except:
        # Fallback to numpy random
        actions = [int(np.random.random() * params.n_actions) for _ in range(params.n_agents)]
    return compute_state_number(actions, params)

def compute_state_number(actions: List[int], params: SimParams) -> int:
    """Compute state number from actions."""
    state = 0
    for i, action in enumerate(actions):
        state += action * (params.n_actions ** i)
    return state

def compute_action_number(actions: List[int], params: SimParams) -> int:
    """Compute action number from individual actions."""
    action_num = 0
    for i, action in enumerate(actions):
        action_num += action * (params.n_actions ** i)
    return action_num

def max_loc_break_ties(array: np.ndarray, params: SimParams) -> Tuple[float, int]:
    """Find index of maximum value, breaking ties randomly."""
    try:
        rng = get_worker_rng()
        max_val = np.max(array)
        max_indices = np.where(array == max_val)[0]
        max_loc = max_indices[int(rng.ran2() * len(max_indices))]
    except:
        # Fallback
        max_val = np.max(array)
        max_indices = np.where(array == max_val)[0]
        max_loc = max_indices[int(np.random.random() * len(max_indices))]
    return max_val, max_loc

def convert_number_base(number: int, base: int, n_digits: int) -> List[int]:
    """Convert number to base-n representation."""
    digits = []
    for _ in range(n_digits):
        digits.append(number % base)
        number //= base
    return digits

def generate_u_ini_price(
    u_ini_price: np.ndarray
) -> None:
    """Generate random numbers for initial price generation."""
    try:
        rng = get_worker_rng()
        shape = u_ini_price.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    u_ini_price[i, j, k] = rng.ran2()
    except:
        # Fallback
        u_ini_price[:] = np.random.random(u_ini_price.shape)

def generate_u_exploration(
    u_exploration: np.ndarray
) -> None:
    """Generate random numbers for exploration."""
    try:
        rng = get_worker_rng()
        shape = u_exploration.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                u_exploration[i, j] = rng.ran2()
    except:
        # Fallback
        u_exploration[:] = np.random.random(u_exploration.shape)

def compute_nash_value(state: int, action: int, agent: int, params: SimParams) -> float:
    """Compute Nash equilibrium value for given state, action, and agent."""
    # Convert action to price
    price = action / (params.n_actions - 1)
    
    # For Nash equilibrium, assume other agents play Nash strategies
    # This is a simplified implementation
    nash_price = 0.5  # Simplified Nash price
    other_prices = [nash_price] * params.n_agents
    other_prices[agent] = price
    
    # Compute profit
    demands = compute_demands(other_prices, params)
    profit = price * demands[agent]
    
    return profit

def compute_coop_value(state: int, action: int, agent: int, params: SimParams) -> float:
    """Compute cooperative value for given state, action, and agent."""
    # Convert action to price
    price = action / (params.n_actions - 1)
    
    # For cooperative equilibrium, assume all agents cooperate
    coop_price = 0.8  # Simplified cooperative price
    prices = [coop_price] * params.n_agents
    prices[agent] = price
    
    # Compute profit
    demands = compute_demands(prices, params)
    profit = price * demands[agent]
    
    return profit

def compute_demands(prices: List[float], params: SimParams) -> List[float]:
    """Compute demands using logit model."""
    # Use parameter defaults if not available
    a0 = getattr(params, 'a0', 2.0)
    a = getattr(params, 'a', 1.0)
    c = getattr(params, 'c', 0.0)
    mu = getattr(params, 'mu', 0.25)
    
    # Compute utilities
    utilities = [a0 - a * p + c for p in prices]
    
    # Compute demands
    exp_utilities = np.exp(array(utilities) / mu)
    total_exp = np.sum(exp_utilities)
    demands = exp_utilities / total_exp
    
    return demands.tolist()

def compute_profit_gain(
    Q_matrices: List[np.ndarray],
    strategies: List[np.ndarray]
) -> float:
    """Compute profit gain relative to Nash equilibrium."""
    # TODO: Implement profit gain computation
    return 0.0

def compute_incentive_compatibility(
    Q_matrices: List[np.ndarray],
    strategies: List[np.ndarray]
) -> float:
    """Compute incentive compatibility measure."""
    # TODO: Implement incentive compatibility computation
    return 0.0

def compute_incentive_ratio(
    Q_matrices: List[np.ndarray],
    strategies: List[np.ndarray]
) -> float:
    """Compute incentive ratio."""
    # TODO: Implement incentive ratio computation
    return 0.0 