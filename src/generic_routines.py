import numpy as np
from typing import List, Tuple, Optional
import random
from .rng.Lecuyer import get_lecuyer_raw, LecuyerCombined

def compute_row_summary_statistics(x: np.ndarray) -> np.ndarray:
    """
    Computes summary statistics on a (m x n) matrix X by rows
    Returns a (m x 9) matrix with columns equal to:
    1: average
    2: standard deviation
    3: minimum
    4: 0.025 percentile
    5: 0.25 percentile
    6: 0.5 percentile
    7: 0.75 percentile
    8: 0.975 percentile
    9: maximum
    """
    m, n = x.shape
    y = np.zeros((m, 9))
    
    # Mean and standard deviation
    y[:, 0] = np.mean(x, axis=1)
    y[:, 1] = np.std(x, axis=1)
    
    # Minimum and maximum
    y[:, 2] = np.min(x, axis=1)
    y[:, 9-1] = np.max(x, axis=1)
    
    # Percentiles
    y[:, 3] = np.percentile(x, 2.5, axis=1)
    y[:, 4] = np.percentile(x, 25, axis=1)
    y[:, 5] = np.percentile(x, 50, axis=1)
    y[:, 6] = np.percentile(x, 75, axis=1)
    y[:, 7] = np.percentile(x, 97.5, axis=1)
    
    return y

def are_equal_reals(a: float, b: float) -> bool:
    """Tests the equality between a and b, two float values"""
    return abs(a - b) <= np.finfo(float).eps

def convert_number_base(n: int, b: int, l: int) -> np.ndarray:
    """Converts an integer n from base 10 to base b, generating a vector of length l"""
    result = np.zeros(l, dtype=int)
    tmp = n
    for i in range(l):
        result[l-i-1] = tmp % b + 1
        tmp = tmp // b
    return result

class RandomNumberGenerator:
    """
    Wrapper for L'Ecuyer RNG to maintain backward compatibility.
    This class provides the same interface as the original RandomNumberGenerator
    but uses the high-quality L'Ecuyer Combined generator internally.
    """
    
    def __init__(self, seed: int = 12345):
        """Initialize with L'Ecuyer generator."""
        self._lecuyer = get_lecuyer_raw(seed)
    
    def ran2(self) -> float:
        """Generate random number between 0 and 1 (exclusive)."""
        return self._lecuyer.ran2()
    
    def random(self) -> float:
        """Generate random number between 0 and 1 (exclusive)."""
        return self.ran2()
    
    def randint(self, low: int, high: int) -> int:
        """Generate random integer between low and high (inclusive)."""
        return low + int(self.ran2() * (high - low + 1))

def generate_combinations(x: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """
    Computes all possible combinations of the columns of the (rows x cols) integer matrix X,
    considering only the first LENGTHS elements in each column.
    Returns a (totrows x cols) matrix where totrows is the product of the elements of LENGTHS.
    """
    rows, cols = x.shape
    totrows = np.prod(lengths)
    comb = np.zeros((totrows, cols), dtype=int)
    
    for itotrows in range(totrows):
        itmp = itotrows
        for icol in range(cols-1, -1, -1):
            index = itmp % lengths[icol]
            itmp = itmp // lengths[icol]
            comb[itotrows, icol] = x[index, icol]
    
    return comb

def compute_row_summary_statistics_parallel(x: np.ndarray, num_cores: int) -> np.ndarray:
    """
    Parallel version of compute_row_summary_statistics using multiprocessing
    """
    from multiprocessing import Pool
    
    def process_chunk(chunk):
        return compute_row_summary_statistics(chunk)
    
    # Split the data into chunks for parallel processing
    chunk_size = x.shape[0] // num_cores
    chunks = [x[i:i+chunk_size] for i in range(0, x.shape[0], chunk_size)]
    
    # Process chunks in parallel
    with Pool(num_cores) as pool:
        results = pool.map(process_chunk, chunks)
    
    # Combine results
    return np.vstack(results)

def get_states_at_convergence(strategies: List[np.ndarray]) -> np.ndarray:
    """Get states at convergence from strategies."""
    n_states = strategies[0].shape[0]
    states = np.zeros(n_states, dtype=int)
    
    for state in range(n_states):
        actions = [strategy[state] for strategy in strategies]
        states[state] = compute_state_number(actions)
    
    return states

def get_prices_at_convergence(states: np.ndarray) -> np.ndarray:
    """Get prices at convergence from states."""
    n_states = len(states)
    prices = np.zeros((n_states, globals.n_agents))
    
    for state in range(n_states):
        actions = convert_number_base(state, globals.n_actions, globals.n_agents)
        prices[state] = [action / (globals.n_actions - 1) for action in actions]
    
    return prices

def compute_state_number(actions: List[int]) -> int:
    """Compute state number from actions."""
    state = 0
    for i, action in enumerate(actions):
        state += action * (globals.n_actions ** i)
    return state

def convert_number_base(number: int, base: int, n_digits: int) -> List[int]:
    """Convert number to base-n representation."""
    digits = []
    for _ in range(n_digits):
        digits.append(number % base)
        number //= base
    return digits 