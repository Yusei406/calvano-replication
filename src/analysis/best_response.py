"""
Best response analysis module.

Implements static and dynamic best response functions for analyzing strategic behavior.
Based on the paper's game-theoretic analysis and optimal response calculations.

References:
    - Section 3.2: Game-theoretic framework and best responses
    - Equation (6): Best response function definition
    - Appendix A: Analytical solutions for linear demand
"""

import numpy as np
from typing import List, Optional, Dict, Any, Callable
from scipy.optimize import minimize_scalar
# Handle imports for both package and standalone usage
try:
    from ..params import SimParams
except ImportError:
    try:
        from params import SimParams
    except ImportError:
        # Fallback for testing
        import numpy as np
        class SimParams:
            def __init__(self, config=None):
                self.n_agents = 2
                self.n_actions = 11
                self.lambda_param = 0.5
                self.a_param = 1.0
                self.demand_model = "logit"

# Handle imports for both package and standalone usage
try:
    from ..dtype_policy import DTYPE, array, zeros
except ImportError:
    try:
        from dtype_policy import DTYPE, array, zeros
    except ImportError:
        # Fallback for testing
        import numpy as np
        DTYPE = np.float64
        array = lambda x: np.array(x, dtype=DTYPE)
        zeros = lambda n: np.zeros(n, dtype=DTYPE)

from .profit_gain import compute_demands_from_prices, calc_profit


def static_best_response(opponent_price: float, demand_func: Callable, cost: float = 0.0,
                        price_range: tuple = (0.0, 1.0)) -> float:
    """
    Compute static best response to opponent's price.
    
    Args:
        opponent_price: Price set by opponent
        demand_func: Function that returns demand given (own_price, opponent_price)
        cost: Marginal cost (default 0)
        price_range: Valid price range (min, max)
        
    Returns:
        Optimal price response
    """
    def profit_function(own_price):
        # Compute demand for this agent
        demand = demand_func(own_price, opponent_price)
        # Return negative profit for minimization
        return -calc_profit(own_price, cost, demand)
    
    # Find optimal price using scalar optimization
    result = minimize_scalar(profit_function, bounds=price_range, method='bounded')
    
    return DTYPE(result.x)


def static_best_response_linear(opponent_price: float, a: float = 1.0, b: float = 1.0, 
                               c: float = 0.5, cost: float = 0.0) -> float:
    """
    Analytical best response for linear demand model.
    
    For linear demand: q_i = a - b*p_i + c*p_j
    Best response: p_i* = (a + c*p_j + b*cost) / (2*b)
    
    Args:
        opponent_price: Opponent's price (p_j)
        a: Demand intercept parameter
        b: Own price sensitivity
        c: Cross-price sensitivity
        cost: Marginal cost
        
    Returns:
        Analytical best response price
    """
    if b <= 0:
        raise ValueError("Own price sensitivity (b) must be positive")
    
    best_response = (a + c * opponent_price + b * cost) / (2 * b)
    
    # Ensure non-negative price
    return DTYPE(max(0.0, best_response))


def static_best_response_logit(opponent_price: float, params: SimParams, 
                              cost: float = 0.0) -> float:
    """
    Numerical best response for logit demand model.
    
    Args:
        opponent_price: Opponent's price
        params: Simulation parameters (contains logit parameters)
        cost: Marginal cost
        
    Returns:
        Best response price
    """
    def logit_demand_func(own_price, opp_price):
        # Create price vector
        prices = [own_price, opp_price]
        demands = compute_demands_from_prices(array(prices), params)
        return demands[0]  # Return own demand
    
    return static_best_response(opponent_price, logit_demand_func, cost)


def dynamic_best_response(Q_matrix: np.ndarray, state: int, params: SimParams,
                         epsilon: float = 0.1) -> int:
    """
    Compute dynamic best response action from Q-matrix.
    
    Args:
        Q_matrix: Q-value matrix for this agent
        state: Current state
        params: Simulation parameters
        epsilon: Exploration parameter (for epsilon-greedy)
        
    Returns:
        Best response action index
    """
    if state >= Q_matrix.shape[0]:
        # Invalid state, return random action
        return np.random.randint(0, Q_matrix.shape[1])
    
    # Get Q-values for current state
    q_values = Q_matrix[state, :]
    
    # Epsilon-greedy policy
    if np.random.random() < epsilon:
        # Explore: random action
        best_action = np.random.randint(0, len(q_values))
    else:
        # Exploit: best action
        best_action = np.argmax(q_values)
    
    return best_action


def compute_best_response_correspondence(price_grid: np.ndarray, demand_func: Callable,
                                       cost: float = 0.0) -> np.ndarray:
    """
    Compute best response correspondence over a price grid.
    
    Args:
        price_grid: Grid of prices to evaluate
        demand_func: Demand function
        cost: Marginal cost
        
    Returns:
        Array of best response prices for each opponent price in grid
    """
    best_responses = []
    
    for opp_price in price_grid:
        br_price = static_best_response(opp_price, demand_func, cost,
                                      price_range=(price_grid[0], price_grid[-1]))
        best_responses.append(br_price)
    
    return array(best_responses)


def find_nash_equilibrium_intersection(price_grid: np.ndarray, demand_func: Callable,
                                     cost: float = 0.0, tolerance: float = 1e-6) -> List[float]:
    """
    Find Nash equilibrium by finding intersections of best response correspondences.
    
    Args:
        price_grid: Price grid for evaluation
        demand_func: Demand function (assumes symmetric agents)
        cost: Marginal cost
        tolerance: Convergence tolerance
        
    Returns:
        List of Nash equilibrium prices
    """
    def br_func_symmetric(opp_price):
        # For symmetric agents, both have same best response function
        return static_best_response(opp_price, 
                                  lambda p1, p2: demand_func(p1, p2), cost)
    
    nash_prices = []
    
    # Search for fixed points: p* = BR(p*)
    for p0 in price_grid:
        try:
            # Use fixed-point iteration
            p_current = p0
            for _ in range(100):  # Max iterations
                p_next = br_func_symmetric(p_current)
                if abs(p_next - p_current) < tolerance:
                    # Found equilibrium
                    if p_current not in nash_prices:  # Avoid duplicates
                        nash_prices.append(p_current)
                    break
                p_current = p_next
        except:
            continue
    
    return sorted(nash_prices)


def analyze_best_response_dynamics(Q_matrices: List[np.ndarray], params: SimParams,
                                  n_steps: int = 1000) -> Dict[str, Any]:
    """
    Analyze best response dynamics from Q-matrices.
    
    Args:
        Q_matrices: List of Q-matrices for each agent
        params: Simulation parameters
        n_steps: Number of simulation steps
        
    Returns:
        Dictionary with dynamics analysis
    """
    n_agents = len(Q_matrices)
    
    # Initialize tracking
    state = 0
    action_history = []
    price_history = []
    
    for step in range(n_steps):
        # Compute best responses
        actions = []
        for agent in range(n_agents):
            action = dynamic_best_response(Q_matrices[agent], state, params, epsilon=0.0)
            actions.append(action)
        
        # Convert to prices
        prices = [action / (params.n_actions - 1) for action in actions]
        
        # Store history
        action_history.append(actions.copy())
        price_history.append(prices.copy())
        
        # Update state (simplified)
        state = sum(action * (params.n_actions ** i) for i, action in enumerate(actions))
        state = state % params.n_states
    
    # Analyze convergence
    price_array = array(price_history)
    
    # Check if prices converged
    final_window = price_array[-100:] if len(price_array) >= 100 else price_array
    price_ranges = []
    for agent in range(n_agents):
        agent_prices = final_window[:, agent]
        price_range = np.max(agent_prices) - np.min(agent_prices)
        price_ranges.append(price_range)
    
    converged = all(pr < 1e-3 for pr in price_ranges)
    
    # Compute final equilibrium
    if converged:
        final_prices = [np.mean(final_window[:, agent]) for agent in range(n_agents)]
    else:
        final_prices = price_array[-1, :].tolist()
    
    # Analyze cycles
    cycle_analysis = analyze_price_cycles(price_array)
    
    return {
        'converged': converged,
        'final_prices': final_prices,
        'price_ranges': price_ranges,
        'cycle_detected': cycle_analysis['has_cycles'],
        'cycle_length': cycle_analysis.get('dominant_cycle_length'),
        'price_history': price_array.tolist(),
        'action_history': action_history
    }


def analyze_price_cycles(price_array: np.ndarray) -> Dict[str, Any]:
    """
    Detect cycles in price dynamics.
    
    Args:
        price_array: Array of shape (time, agents) with price history
        
    Returns:
        Dictionary with cycle analysis
    """
    if len(price_array) < 10:
        return {'has_cycles': False}
    
    # Look for repeating patterns in the last part of the series
    analysis_window = price_array[-200:] if len(price_array) >= 200 else price_array
    
    cycles_detected = {}
    
    # Check for cycles of different lengths
    for cycle_len in range(2, min(20, len(analysis_window) // 3)):
        # Extract potential cycle patterns
        n_complete_cycles = len(analysis_window) // cycle_len
        if n_complete_cycles >= 2:
            # Compare consecutive cycles
            patterns = []
            for i in range(n_complete_cycles):
                start_idx = len(analysis_window) - (i + 1) * cycle_len
                end_idx = len(analysis_window) - i * cycle_len
                pattern = analysis_window[start_idx:end_idx]
                patterns.append(pattern)
            
            # Check if patterns are similar
            if len(patterns) >= 2:
                pattern_similarity = np.mean([
                    np.allclose(patterns[i], patterns[i+1], atol=1e-3)
                    for i in range(len(patterns) - 1)
                ])
                
                if pattern_similarity > 0.8:  # 80% of comparisons match
                    cycles_detected[cycle_len] = {
                        'similarity': pattern_similarity,
                        'n_cycles': n_complete_cycles
                    }
    
    # Determine dominant cycle
    has_cycles = len(cycles_detected) > 0
    dominant_cycle_length = None
    
    if has_cycles:
        # Choose cycle with highest similarity score
        best_cycle = max(cycles_detected.items(), 
                        key=lambda x: x[1]['similarity'])
        dominant_cycle_length = best_cycle[0]
    
    return {
        'has_cycles': has_cycles,
        'cycles_detected': cycles_detected,
        'dominant_cycle_length': dominant_cycle_length
    }


def compute_reaction_function_slope(price_grid: np.ndarray, demand_func: Callable,
                                  cost: float = 0.0) -> float:
    """
    Compute slope of best response function (reaction function).
    
    Args:
        price_grid: Price grid for numerical differentiation
        demand_func: Demand function
        cost: Marginal cost
        
    Returns:
        Average slope of reaction function
    """
    best_responses = compute_best_response_correspondence(price_grid, demand_func, cost)
    
    # Numerical differentiation
    slopes = []
    for i in range(1, len(price_grid)):
        dp_opp = price_grid[i] - price_grid[i-1]
        dp_own = best_responses[i] - best_responses[i-1]
        if abs(dp_opp) > 1e-10:
            slope = dp_own / dp_opp
            slopes.append(slope)
    
    return np.mean(slopes) if slopes else 0.0


def validate_best_response_linear_demand():
    """
    Unit test: Validate analytical best response for linear demand.
    
    Returns:
        True if test passes, False otherwise
    """
    # Test parameters
    a, b, c = 1.0, 1.0, 0.5
    cost = 0.0
    opponent_price = 0.5
    
    # Analytical solution
    analytical_br = static_best_response_linear(opponent_price, a, b, c, cost)
    
    # Numerical solution
    def linear_demand(own_price, opp_price):
        return a - b * own_price + c * opp_price
    
    numerical_br = static_best_response(opponent_price, linear_demand, cost)
    
    # Check if they match within tolerance
    tolerance = 1e-6
    return abs(analytical_br - numerical_br) < tolerance


def compute_strategic_complementarity(price_grid: np.ndarray, demand_func: Callable) -> str:
    """
    Determine if prices are strategic complements or substitutes.
    
    Args:
        price_grid: Price grid
        demand_func: Demand function
        
    Returns:
        'complements' if slope > 0, 'substitutes' if slope < 0, 'independent' if slope ≈ 0
    """
    slope = compute_reaction_function_slope(price_grid, demand_func)
    
    if slope > 1e-6:
        return 'complements'
    elif slope < -1e-6:
        return 'substitutes'
    else:
        return 'independent' 