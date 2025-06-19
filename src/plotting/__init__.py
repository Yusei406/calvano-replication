"""
Plotting module for Q-learning simulation results.
Implements visualization functions for the paper's figures and analysis.
"""

# Try to import with graceful fallback
try:
    from .figures import plot_convergence_trajectory, plot_price_distribution, plot_impulse_response
    FIGURES_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Figures module not fully available: {e}")
    FIGURES_AVAILABLE = False
    # Define fallback functions
    def plot_convergence_trajectory(*args, **kwargs):
        print("plot_convergence_trajectory not available")
        return None
    def plot_price_distribution(*args, **kwargs):
        print("plot_price_distribution not available")
        return None
    def plot_impulse_response(*args, **kwargs):
        print("plot_impulse_response not available")
        return None

try:
    from .tables import generate_table1, generate_table2, generate_table3
    TABLES_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Tables module not fully available: {e}")
    TABLES_AVAILABLE = False
    # Define fallback functions
    def generate_table1(*args, **kwargs):
        print("generate_table1 not available")
        return None
    def generate_table2(*args, **kwargs):
        print("generate_table2 not available")
        return None
    def generate_table3(*args, **kwargs):
        print("generate_table3 not available")
        return None

__all__ = [
    'plot_convergence_trajectory', 'plot_price_distribution', 'plot_impulse_response',
    'generate_table1', 'generate_table2', 'generate_table3',
    'FIGURES_AVAILABLE', 'TABLES_AVAILABLE'
] 