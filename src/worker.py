"""
Worker initialization for multiprocessing safety.
Ensures each worker process has its own RNG state.
"""

from typing import Any
from .params import SimParams
from .rng.Lecuyer import set_global_rng

# Global variables for worker processes
RNG = None
PARAMS = None


def init_worker(rank: int, base_seed: int, sim_params: SimParams):
    """
    Initialize worker process with unique RNG state and parameters.
    
    Args:
        rank: Worker process rank/index
        base_seed: Base random seed
        sim_params: Simulation parameters
    """
    global RNG, PARAMS
    
    # Set unique RNG for this worker
    set_global_rng(base_seed, rank)
    
    # Store parameters globally for this worker
    PARAMS = sim_params
    
    # For backward compatibility with existing code
    from .rng.Lecuyer import get_global_raw_rng
    RNG = get_global_raw_rng()


def get_worker_rng():
    """Get the worker's RNG instance."""
    global RNG
    if RNG is None:
        # Fallback initialization
        init_worker(0, 12345, None)
    return RNG


def get_worker_params() -> SimParams:
    """Get the worker's parameters."""
    global PARAMS
    if PARAMS is None:
        # Should not happen in normal usage
        raise RuntimeError("Worker not properly initialized. Call init_worker() first.")
    return PARAMS 