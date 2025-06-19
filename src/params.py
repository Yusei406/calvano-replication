"""
Parameter management for Calvano Q-learning simulation.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Return default configuration
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration matching paper parameters.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "n_agents": 2,
        "n_actions": 11,
        "max_iterations": 1000000,
        "convergence_window": 10000,
        "convergence_tolerance": 0.02,
        "demand_model": "logit",
        "a0": 1.0,    # Market size parameter
        "a": 1.0,     # Own-price sensitivity 
        "c": 0.0,     # Marginal cost
        "mu": 0.0,    # Logit scaling (0 = linear demand)
        "discount_factor": 0.95,
        "learning_rate": 0.1,
        "rng_seed": 12345
    }


class SimParams:
    """Simulation parameters container with backward compatibility."""
    
    def __init__(self, *args, **kwargs):
        """
        Initialize parameters from configuration dictionary or individual keywords.
        
        Args:
            *args: Positional arguments (first can be config dict)
            **kwargs: Individual parameter values
        """
        # Handle dict input (new API)
        if len(args) == 1 and isinstance(args[0], dict):
            config = args[0]
            kwargs.update(config)
        
        # Load from config file if available
        if 'config_path' in kwargs:
            file_config = load_config(kwargs.pop('config_path'))
            # File config has lower priority than explicit kwargs
            for key, value in file_config.items():
                if key not in kwargs:
                    kwargs[key] = value
        
        # Set defaults first
        self.n_agents = kwargs.pop('n_agents', 2)
        self.n_actions = kwargs.pop('n_actions', 11) 
        self.n_prices = kwargs.pop('n_prices', 11)  # For backward compatibility
        self.n_states = kwargs.pop('n_states', 121)
        self.state_depth = kwargs.pop('state_depth', 1)
        self.q_strategy = kwargs.pop('q_strategy', "R")
        
        # Demand model parameters (from paper Table A1)
        self.a0 = kwargs.pop('a0', 1.0)
        self.a = kwargs.pop('a', 1.0)  
        self.c = kwargs.pop('c', 0.0)
        self.mu = kwargs.pop('mu', 0.0)
        self.demand_model = kwargs.pop('demand_model', "logit")
        
        # Convergence parameters
        self.convergence_window = kwargs.pop('convergence_window', 10000)
        self.convergence_tolerance = kwargs.pop('convergence_tolerance', 0.02)
        
        # Additional parameters for backward compatibility
        self.n_runs = kwargs.pop('n_runs', 50)
        self.max_episodes = kwargs.pop('max_episodes', 2000)
        self.alpha = kwargs.pop('alpha', 0.1)
        self.delta = kwargs.pop('delta', 0.95)
        self.epsilon = kwargs.pop('epsilon', 0.1)
        self.lambda_param = kwargs.pop('lambda_param', 0.5)
        self.a_param = kwargs.pop('a_param', 1.0)
        self.rng_seed = kwargs.pop('rng_seed', 42)
        self.q_init_strategy = kwargs.pop('q_init_strategy', "R")
        self.conv_window = kwargs.pop('conv_window', self.convergence_window)
        self.conv_tolerance = kwargs.pop('conv_tolerance', self.convergence_tolerance)
        self.save_q_tables = kwargs.pop('save_q_tables', False)
        self.save_detailed_logs = kwargs.pop('save_detailed_logs', True)
        
        # Price grid handling
        self.price_grid = kwargs.pop('price_grid', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        # Store any extra parameters
        self.extra = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return {
            'n_agents': self.n_agents,
            'n_actions': self.n_actions,
            'n_prices': self.n_prices,
            'n_states': self.n_states,
            'state_depth': self.state_depth,
            'q_strategy': self.q_strategy,
            'a0': self.a0,
            'a': self.a,
            'c': self.c,
            'mu': self.mu,
            'demand_model': self.demand_model,
            'convergence_window': self.convergence_window,
            'convergence_tolerance': self.convergence_tolerance,
            'n_runs': self.n_runs,
            'max_episodes': self.max_episodes,
            'alpha': self.alpha,
            'delta': self.delta,
            'epsilon': self.epsilon,
            'lambda_param': self.lambda_param,
            'a_param': self.a_param,
            'rng_seed': self.rng_seed,
            'q_init_strategy': self.q_init_strategy,
            'conv_window': self.conv_window,
            'conv_tolerance': self.conv_tolerance,
            'save_q_tables': self.save_q_tables,
            'save_detailed_logs': self.save_detailed_logs,
            'price_grid': self.price_grid,
            **self.extra
        }


@dataclass
class SimResults:
    """Simulation results container."""
    
    # Convergence results
    converged: Optional[np.ndarray] = None
    time_to_convergence: Optional[np.ndarray] = None
    index_strategies: Optional[np.ndarray] = None
    cycle_length: Optional[np.ndarray] = None
    cycle_states: Optional[np.ndarray] = None
    
    # Profit matrices
    pi: Optional[np.ndarray] = None
    nash_profits: Optional[np.ndarray] = None
    coop_profits: Optional[np.ndarray] = None
    pg: Optional[np.ndarray] = None
    
    # Q-learning results
    Q_matrices: Optional[list] = None
    strategies: Optional[list] = None 