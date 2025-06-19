"""
Q-value initialization strategies for Q-learning.
Implements all 6 strategies from the original Fortran code.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import os

# Handle imports for both standalone and package usage
try:
    from ..params import SimParams
    from ..rng.Lecuyer import LecuyerCombined
except ImportError:
    # Fallback for standalone usage
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from params import SimParams
    from rng.Lecuyer import LecuyerCombined

# Strategy type definitions
STRATEGY_TYPES = ['F', 'G', 'O', 'T', 'R', 'U']


def init_Q(strategy: str, params: SimParams, rng: LecuyerCombined, 
           agent_idx: int = 0, **kwargs) -> np.ndarray:
    """
    Initialize Q-matrix based on specified strategy.
    
    Args:
        strategy: Strategy type ('F', 'G', 'O', 'T', 'R', 'U')
        params: Simulation parameters
        rng: Random number generator
        agent_idx: Index of the agent (for strategy-specific parameters)
        **kwargs: Additional parameters specific to each strategy
        
    Returns:
        Q-matrix of shape (n_states, n_actions)
    """
    if strategy not in STRATEGY_TYPES:
        raise ValueError(f"Unknown strategy '{strategy}'. Must be one of {STRATEGY_TYPES}")
    
    if strategy == 'F':
        return _init_Q_fixed(params, agent_idx, **kwargs)
    elif strategy == 'G':
        return _init_Q_grim_trigger(params, rng, agent_idx, **kwargs)
    elif strategy == 'O':
        return _init_Q_opponent_random(params, rng, agent_idx, **kwargs)
    elif strategy == 'T':
        return _init_Q_pretrained(params, agent_idx, **kwargs)
    elif strategy == 'R':
        return _init_Q_random(params, rng, agent_idx, **kwargs)
    elif strategy == 'U':
        return _init_Q_uniform(params, agent_idx, **kwargs)


def _init_Q_fixed(params: SimParams, agent_idx: int, **kwargs) -> np.ndarray:
    """
    'F': Fixed strategy initialization.
    Agent assumes other agents play fixed prices.
    Q matrix is set to 0 for the fixed price, -inf for others.
    """
    fixed_price = kwargs.get('fixed_price', 0.5)
    
    # Convert fixed price to action index
    fixed_action = int(round(fixed_price * (params.n_actions - 1)))
    fixed_action = np.clip(fixed_action, 0, params.n_actions - 1)
    
    # Initialize Q matrix
    Q = np.full((params.n_states, params.n_actions), -np.inf, dtype=np.float64)
    
    # Set Q values for fixed action to 0 (neutral)
    Q[:, fixed_action] = 0.0
    
    return Q


def _init_Q_grim_trigger(params: SimParams, rng: LecuyerCombined, 
                        agent_idx: int, **kwargs) -> np.ndarray:
    """
    'G': Grim Trigger strategy initialization.
    2x2 format: cooperation price gets high value, deviation gets low value.
    """
    coop_price = kwargs.get('coop_price', 0.8)
    punish_price = kwargs.get('punish_price', 0.0)
    coop_value = kwargs.get('coop_value', 1.0)
    punish_value = kwargs.get('punish_value', -1.0)
    
    # Convert prices to action indices
    coop_action = int(round(coop_price * (params.n_actions - 1)))
    punish_action = int(round(punish_price * (params.n_actions - 1)))
    
    coop_action = np.clip(coop_action, 0, params.n_actions - 1)
    punish_action = np.clip(punish_action, 0, params.n_actions - 1)
    
    # Initialize Q matrix with neutral values
    Q = np.zeros((params.n_states, params.n_actions), dtype=np.float64)
    
    # Set cooperation and punishment values
    Q[:, coop_action] = coop_value
    Q[:, punish_action] = punish_value
    
    # Add small random noise to break ties
    noise_scale = 0.01
    for state in range(params.n_states):
        for action in range(params.n_actions):
            Q[state, action] += noise_scale * (rng.ran2() - 0.5)
    
    return Q


def _init_Q_opponent_random(params: SimParams, rng: LecuyerCombined, 
                           agent_idx: int, **kwargs) -> np.ndarray:
    """
    'O': Opponent randomization initialization.
    Self-agent Q values set to 0, opponent parts randomized.
    """
    random_scale = kwargs.get('random_scale', 1.0)
    
    # Initialize Q matrix
    Q = np.zeros((params.n_states, params.n_actions), dtype=np.float64)
    
    # Randomize Q values (standard normal distribution)
    for state in range(params.n_states):
        for action in range(params.n_actions):
            # Use Box-Muller transform for normal distribution
            if state % 2 == 0:  # Generate pairs of normal variables
                u1 = rng.ran2()
                u2 = rng.ran2()
                z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
                z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
                Q[state, action] = random_scale * z1
                if state + 1 < params.n_states:
                    Q[state + 1, action] = random_scale * z2
            # Note: odd states are handled in the previous iteration
    
    return Q


def _init_Q_pretrained(params: SimParams, agent_idx: int, **kwargs) -> np.ndarray:
    """
    'T': Pre-trained Q-matrix initialization.
    Load Q-matrix from file.
    """
    q_init_path = kwargs.get('q_init_path')
    if q_init_path is None:
        raise ValueError("q_init_path must be provided for 'T' strategy")
    
    if not os.path.exists(q_init_path):
        raise FileNotFoundError(f"Q initialization file not found: {q_init_path}")
    
    # Load Q matrix
    try:
        Q = np.load(q_init_path)
    except Exception as e:
        raise ValueError(f"Failed to load Q matrix from {q_init_path}: {e}")
    
    # Validate shape
    expected_shape = (params.n_states, params.n_actions)
    if Q.shape != expected_shape:
        raise ValueError(f"Q matrix shape {Q.shape} does not match expected {expected_shape}")
    
    # Ensure finite values
    if not np.isfinite(Q).all():
        raise ValueError("Q matrix contains non-finite values")
    
    return Q.astype(np.float64)


def _init_Q_random(params: SimParams, rng: LecuyerCombined, 
                  agent_idx: int, **kwargs) -> np.ndarray:
    """
    'R': Random initialization.
    Q values uniformly distributed in [min_val, max_val].
    """
    min_val = kwargs.get('min_val', 0.0)
    max_val = kwargs.get('max_val', 1.0)
    
    # Initialize Q matrix
    Q = np.zeros((params.n_states, params.n_actions), dtype=np.float64)
    
    # Fill with random values
    for state in range(params.n_states):
        for action in range(params.n_actions):
            Q[state, action] = min_val + (max_val - min_val) * rng.ran2()
    
    return Q


def _init_Q_uniform(params: SimParams, agent_idx: int, **kwargs) -> np.ndarray:
    """
    'U': Uniform initialization.
    All Q values set to the same constant (not random).
    """
    constant_value = kwargs.get('constant_value', 0.0)
    
    # Initialize Q matrix with constant value
    Q = np.full((params.n_states, params.n_actions), constant_value, dtype=np.float64)
    
    return Q


def init_all_agents_Q(strategies: List[str], params: SimParams, rng: LecuyerCombined,
                     **strategy_kwargs) -> List[np.ndarray]:
    """
    Initialize Q-matrices for all agents.
    
    Args:
        strategies: List of strategy types for each agent
        params: Simulation parameters
        rng: Random number generator
        **strategy_kwargs: Additional parameters for strategies
        
    Returns:
        List of Q-matrices, one for each agent
    """
    if len(strategies) != params.n_agents:
        raise ValueError(f"Number of strategies ({len(strategies)}) must match n_agents ({params.n_agents})")
    
    Q_matrices = []
    for agent_idx, strategy in enumerate(strategies):
        # Get agent-specific kwargs
        agent_kwargs = {}
        for key, value in strategy_kwargs.items():
            if isinstance(value, (list, tuple)) and len(value) == params.n_agents:
                agent_kwargs[key] = value[agent_idx]
            else:
                agent_kwargs[key] = value
        
        Q = init_Q(strategy, params, rng, agent_idx, **agent_kwargs)
        Q_matrices.append(Q)
    
    return Q_matrices


def validate_Q_matrix(Q: np.ndarray, params: SimParams) -> bool:
    """
    Validate Q-matrix shape and content.
    
    Args:
        Q: Q-matrix to validate
        params: Simulation parameters
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    expected_shape = (params.n_states, params.n_actions)
    
    if Q.shape != expected_shape:
        raise ValueError(f"Q matrix shape {Q.shape} does not match expected {expected_shape}")
    
    # Allow -inf values (used by fixed strategy) but not +inf or NaN
    if np.isnan(Q).any() or np.isposinf(Q).any():
        raise ValueError("Q matrix contains NaN or +inf values")
    
    if Q.dtype != np.float64:
        raise ValueError(f"Q matrix dtype {Q.dtype} should be float64")
    
    return True 