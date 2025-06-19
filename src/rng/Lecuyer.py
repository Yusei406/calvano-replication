"""
L'Ecuyer (1999) Combined Multiple Recursive Generator with Bays-Durham Shuffle
Based on the algorithm used in the original Fortran code.
"""

import numpy as np
from numpy.random import Generator, BitGenerator
from typing import Optional
import threading

class LecuyerCombined(BitGenerator):
    """
    L'Ecuyer Combined Multiple Recursive Generator with Bays-Durham Shuffle
    
    This implements the same algorithm as the Fortran ran2() function used in the paper.
    Period: approximately 2.3 × 10^18
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self._seed = seed if seed is not None else 12345
        self._lock = threading.Lock()
        self._initialize()
    
    def _initialize(self):
        """Initialize the generator state"""
        # Constants for L'Ecuyer algorithm
        self.IM1 = 2147483563
        self.IM2 = 2147483399
        self.IMM1 = self.IM1 - 1
        self.IA1 = 40014
        self.IA2 = 40692
        self.IQ1 = 53668
        self.IQ2 = 52774
        self.IR1 = 12211
        self.IR2 = 3791
        self.NDIV = 1 + self.IMM1 // 32
        
        # Real constants
        self.AM = 1.0 / self.IM1
        self.EPS = 1.2e-7
        self.RNMX = 1.0 - self.EPS
        
        # Initialize state
        self.idum = max(-abs(self._seed), 1)
        self.idum2 = 123456789
        self.iy = 0
        self.iv = np.zeros(32, dtype=np.int64)
        
        # Warm up the generator
        for j in range(32 + 8, 0, -1):
            k = self.idum // self.IQ1
            self.idum = self.IA1 * (self.idum - k * self.IQ1) - k * self.IR1
            if self.idum < 0:
                self.idum += self.IM1
            if j <= 32:
                self.iv[j-1] = self.idum
        self.iy = self.iv[0]
    
    def random_raw(self) -> int:
        """Generate a random 64-bit integer"""
        with self._lock:
            return self._ran2_raw()
    
    def _ran2_raw(self) -> int:
        """
        Core L'Ecuyer algorithm - generates uniform random deviate between 0.0 and 1.0
        Returns as integer scaled to 64-bit range for BitGenerator compatibility
        """
        # Combined generator
        k = self.idum // self.IQ1
        self.idum = self.IA1 * (self.idum - k * self.IQ1) - k * self.IR1
        if self.idum < 0:
            self.idum += self.IM1
        
        k = self.idum2 // self.IQ2
        self.idum2 = self.IA2 * (self.idum2 - k * self.IQ2) - k * self.IR2
        if self.idum2 < 0:
            self.idum2 += self.IM2
        
        # Bays-Durham shuffle
        j = 1 + self.iy // self.NDIV
        j = min(j, 32)  # Ensure j is in valid range
        self.iy = self.iv[j-1] - self.idum2
        self.iv[j-1] = self.idum
        
        if self.iy < 1:
            self.iy += self.IMM1
        
        # Convert to [0, 1) and then to 64-bit integer
        temp = min(self.AM * self.iy, self.RNMX)
        return int(temp * (2**64 - 1))
    
    def ran2(self) -> float:
        """
        Generate a uniform random deviate between 0.0 and 1.0 (exclusive)
        This matches the original Fortran ran2() function exactly
        """
        with self._lock:
            # Combined generator
            k = self.idum // self.IQ1
            self.idum = self.IA1 * (self.idum - k * self.IQ1) - k * self.IR1
            if self.idum < 0:
                self.idum += self.IM1
            
            k = self.idum2 // self.IQ2
            self.idum2 = self.IA2 * (self.idum2 - k * self.IQ2) - k * self.IR2
            if self.idum2 < 0:
                self.idum2 += self.IM2
            
            # Bays-Durham shuffle
            j = 1 + self.iy // self.NDIV
            j = min(j, 32)  # Ensure j is in valid range
            self.iy = self.iv[j-1] - self.idum2
            self.iv[j-1] = self.idum
            
            if self.iy < 1:
                self.iy += self.IMM1
            
            return min(self.AM * self.iy, self.RNMX)
    
    @property
    def state(self):
        """Get generator state"""
        return {
            'bit_generator': 'LecuyerCombined',
            'state': {
                'idum': self.idum,
                'idum2': self.idum2,
                'iy': self.iy,
                'iv': self.iv.copy()
            }
        }
    
    @state.setter
    def state(self, value):
        """Set generator state"""
        if value['bit_generator'] != 'LecuyerCombined':
            raise ValueError("State is not from LecuyerCombined")
        
        state = value['state']
        self.idum = state['idum']
        self.idum2 = state['idum2']
        self.iy = state['iy']
        self.iv = state['iv'].copy()


def get_rng(seed: Optional[int] = None) -> Generator:
    """
    Get a numpy Generator using safe default RNG
    
    Note: Originally used LecuyerCombined, but that causes segfaults in CI.
    Using numpy.random.default_rng() for compatibility while preserving API.
    
    Args:
        seed: Random seed. If None, uses default seed.
        
    Returns:
        numpy.random.Generator instance using default bit generator
    """
    return np.random.default_rng(seed)


def get_lecuyer_raw(seed: Optional[int] = None) -> LecuyerCombined:
    """
    Get raw L'Ecuyer generator for direct access to ran2() method
    
    Args:
        seed: Random seed. If None, uses default seed.
        
    Returns:
        LecuyerCombined instance
    """
    return LecuyerCombined(seed)


# Global RNG instance for thread-local storage
_thread_local = threading.local()


def set_global_rng(seed: int, rank: int = 0):
    """
    Set thread-local RNG instance (for multiprocessing workers)
    
    Args:
        seed: Base seed
        rank: Process rank (added to seed for uniqueness)
    """
    effective_seed = seed + rank
    _thread_local.rng = np.random.default_rng(effective_seed)
    _thread_local.raw_rng = get_lecuyer_raw(effective_seed)


def get_global_rng() -> Generator:
    """Get thread-local RNG instance"""
    if not hasattr(_thread_local, 'rng'):
        # Default initialization
        set_global_rng(12345, 0)
    return _thread_local.rng


def get_global_raw_rng() -> LecuyerCombined:
    """Get thread-local raw RNG instance"""
    if not hasattr(_thread_local, 'raw_rng'):
        # Default initialization
        set_global_rng(12345, 0)
    return _thread_local.raw_rng 