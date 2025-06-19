"""
Phase 2 analysis modules.

Provides comprehensive analysis tools for Q-learning simulation results.
"""

from .convergence_results import ConvergenceStats, aggregate_runs, to_dataframe
from .profit_gain import calc_profit, gain_vs_nash, gain_vs_random, analyze_profit_distribution
from .state_frequency import count_state_freq, detect_cycles
from .impulse_response import analyze_impulse_response, analyze_multiple_shocks, calculate_shock_statistics
from .best_response import static_best_response, dynamic_best_response
from .equilibrium_check import check_nash, check_coop, check_price_symmetry

__all__ = [
    # Convergence analysis
    'ConvergenceStats', 'aggregate_runs', 'to_dataframe',
    
    # Profit analysis
    'calc_profit', 'gain_vs_nash', 'gain_vs_random', 'analyze_profit_distribution',
    
    # State frequency analysis
    'count_state_freq', 'detect_cycles',
    
    # Impulse response analysis
    'analyze_impulse_response', 'analyze_multiple_shocks', 'calculate_shock_statistics',
    
    # Best response analysis
    'static_best_response', 'dynamic_best_response',
    
    # Equilibrium checking
    'check_nash', 'check_coop', 'check_price_symmetry'
] 