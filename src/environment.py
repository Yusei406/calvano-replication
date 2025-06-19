"""
Environment implementation for Calvano Q-learning simulation.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from params import SimParams


class CalvanoEnvironment:
    """Basic environment for Calvano duopoly game."""
    
    def __init__(self, params: SimParams):
        """
        Initialize environment.
        
        Args:
            params: Simulation parameters
        """
        self.params = params
        self.n_agents = params.n_agents
        self.n_actions = params.n_actions
        self.n_states = params.n_states
        
        # Price grid (actions map to prices)
        if hasattr(params, 'price_grid') and params.price_grid:
            self.price_grid = params.price_grid
        else:
            # Default price grid from 0 to 1
            self.price_grid = [i / (self.n_actions - 1) for i in range(self.n_actions)]
        
        # Current state
        self.current_state = 0
        
    def reset(self) -> int:
        """Reset environment and return initial state."""
        # Random initial state
        self.current_state = np.random.randint(0, self.n_states)
        return self.current_state
    
    def step(self, actions: List[int]) -> Tuple[int, List[float], bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            actions: List of actions (price indices) for each agent
            
        Returns:
            Tuple of (next_state, rewards, done, info)
        """
        # Convert actions to prices
        prices = [self.price_grid[action] for action in actions]
        
        # Compute demands and profits
        demands = self._compute_demands(prices)
        profits = self._compute_profits(prices, demands)
        
        # Compute next state (simple: based on current actions)
        next_state = self._compute_next_state(actions)
        
        # Rewards are profits
        rewards = profits
        
        # Episode never ends in this simple implementation
        done = False
        
        # Info dictionary
        info = {
            'prices': prices,
            'demands': demands,
            'actions': actions
        }
        
        self.current_state = next_state
        return next_state, rewards, done, info
    
    def _compute_demands(self, prices: List[float]) -> List[float]:
        """Compute demand for each agent using logit model."""
        if self.params.demand_model == "logit" and self.params.mu > 0:
            # Logit demand model
            utilities = [self.params.a0 - self.params.a * p for p in prices]
            exp_utilities = np.exp(np.array(utilities) / self.params.mu)
            total_exp = np.sum(exp_utilities) + 1  # Include outside option
            demands = exp_utilities / total_exp
        else:
            # Linear demand model (or logit with mu=0)
            demands = []
            for i, p in enumerate(prices):
                # Linear demand: d_i = a0 - a*p_i + a*sum(p_j for j != i)
                other_prices_sum = sum(prices) - p
                demand = self.params.a0 - self.params.a * p + self.params.a * other_prices_sum / (len(prices) - 1)
                demands.append(max(0, demand))  # Non-negative demand
        
        return demands
    
    def _compute_profits(self, prices: List[float], demands: List[float]) -> List[float]:
        """Compute profit for each agent."""
        profits = []
        for p, d in zip(prices, demands):
            profit = (p - self.params.c) * d
            profits.append(profit)
        return profits
    
    def _compute_next_state(self, actions: List[int]) -> int:
        """Compute next state based on actions."""
        # Simple state computation: combine actions into state index
        if self.params.state_depth == 1:
            # State is just the action combination
            state = 0
            for i, action in enumerate(actions):
                state += action * (self.n_actions ** i)
            return state % self.n_states
        else:
            # For deeper states, would need price history
            # For now, just use action combination
            return sum(actions) % self.n_states


class LogitEnvironment(CalvanoEnvironment):
    """Logit-specific environment implementation."""
    
    def _compute_demands(self, prices: List[float]) -> List[float]:
        """Compute demand using logit model specifically."""
        utilities = [self.params.a0 - self.params.a * p for p in prices]
        
        if self.params.mu == 0:
            # Linear demand as limit case
            demands = []
            for i, p in enumerate(prices):
                other_prices_sum = sum(prices) - p
                demand = self.params.a0 - self.params.a * p + self.params.a * other_prices_sum / (len(prices) - 1)
                demands.append(max(0, demand))
        else:
            # True logit model
            exp_utilities = np.exp(np.array(utilities) / self.params.mu)
            total_exp = np.sum(exp_utilities) + 1  # Include outside option
            demands = exp_utilities / total_exp
        
        return demands.tolist() if isinstance(demands, np.ndarray) else demands 