import numpy as np
from typing import Dict, Any
from .params import SimParams

def load_params_from_file(file_path: str) -> SimParams:
    """Load parameters from text file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    # Skip comments and empty lines
    lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    
    # Parse parameters
    params_dict = {}
    params_dict['n_agents'] = int(lines[0])
    params_dict['n_prices'] = int(lines[1])
    params_dict['state_depth'] = int(lines[2])
    params_dict['delta'] = float(lines[3])
    params_dict['alpha'] = float(lines[4])
    params_dict['exploration_parameter'] = float(lines[5])
    
    # Demand parameters
    params_dict['gamma'] = float(lines[6])  # Linear demand
    logit_params = list(map(float, lines[7].split()))
    params_dict['a0'] = logit_params[0]
    params_dict['a'] = logit_params[1]
    params_dict['c'] = logit_params[2]
    params_dict['mu'] = logit_params[3]
    
    # Simulation parameters
    params_dict['n_sessions'] = int(lines[8])
    params_dict['n_iterations'] = int(lines[9])
    params_dict['n_perfect_measurement'] = int(lines[10])
    params_dict['max_iterations'] = int(lines[11])
    params_dict['exploration_type'] = int(lines[12])
    params_dict['n_cores'] = int(lines[13])
    params_dict['experiment_number'] = int(lines[14])
    params_dict['q_init_type'] = int(lines[15])
    
    return SimParams(**params_dict)

def load_params_from_dict(config: Dict[str, Any]) -> SimParams:
    """Load parameters from dictionary."""
    return SimParams(**config)

def create_default_params() -> SimParams:
    """Create default parameters for testing."""
    return SimParams(
        n_agents=2,
        n_prices=11,
        state_depth=1,
        delta=0.95,
        alpha=0.1,
        exploration_parameter=1.0,
        gamma=0.5,
        a0=1.0,
        a=2.0,
        c=0.5,
        mu=0.1,
        n_sessions=10,
        n_iterations=1000,
        n_perfect_measurement=100,
        max_iterations=100000,
        exploration_type=1,
        n_cores=4,
        experiment_number=1,
        q_init_type=1
    ) 