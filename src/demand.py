"""
Demand calculation module for Calvano Q-learning simulation.
"""

import numpy as np
from typing import List
from params import SimParams


class DemandCalculator:
    """Calculator for demand functions."""
    
    def __init__(self, params: SimParams):
        """Initialize with simulation parameters."""
        self.params = params
    
    def compute_demands(self, prices: List[float]) -> List[float]:
        """Compute demands for given prices."""
        if self.params.demand_model == "logit" and self.params.mu > 0:
            return self._logit_demands(prices)
        else:
            return self._linear_demands(prices)
    
    def _logit_demands(self, prices: List[float]) -> List[float]:
        """Compute logit demands."""
        utilities = [self.params.a0 - self.params.a * p for p in prices]
        exp_utilities = np.exp(np.array(utilities) / self.params.mu)
        total_exp = np.sum(exp_utilities) + 1  # Include outside option
        demands = exp_utilities / total_exp
        return demands.tolist()
    
    def _linear_demands(self, prices: List[float]) -> List[float]:
        """Compute linear demands."""
        demands = []
        for i, p in enumerate(prices):
            other_prices_sum = sum(prices) - p
            demand = (self.params.a0 - self.params.a * p + 
                     self.params.a * other_prices_sum / (len(prices) - 1))
            demands.append(max(0, demand))
        return demands 