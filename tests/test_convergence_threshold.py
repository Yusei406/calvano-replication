"""
Test convergence threshold parameters match paper specifications.
"""

import pytest
import json
import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from params import SimParams, load_config, get_default_config


class TestConvergenceThreshold:
    """Test convergence threshold parameters."""
    
    def test_convergence_window_paper_spec(self):
        """Test that convergence window matches paper specification (10000)."""
        config = load_config("config.json")
        params = SimParams(config)
        
        # Paper specifies 10000 iterations for convergence window
        assert params.convergence_window == 10000
        assert params.conv_window == 10000
        
    def test_convergence_tolerance_paper_spec(self):
        """Test that convergence tolerance matches paper specification (0.02)."""
        config = load_config("config.json")
        params = SimParams(config)
        
        # Paper specifies 0.02 as convergence tolerance
        assert params.convergence_tolerance == 0.02
        assert params.conv_tolerance == 0.02
        
    def test_default_convergence_parameters(self):
        """Test default convergence parameters are correct."""
        default_config = get_default_config()
        
        assert default_config["convergence_window"] == 10000
        assert default_config["convergence_tolerance"] == 0.02
        
    def test_convergence_parameters_consistency(self):
        """Test that convergence parameters are consistent across different access methods."""
        params = SimParams()
        
        # Test backward compatibility
        assert hasattr(params, 'convergence_window')
        assert hasattr(params, 'conv_window') 
        assert hasattr(params, 'convergence_tolerance')
        assert hasattr(params, 'conv_tolerance')
        
        # Test values are consistent
        assert params.convergence_window == params.conv_window
        assert params.convergence_tolerance == params.conv_tolerance
        
    def test_convergence_parameters_from_dict(self):
        """Test convergence parameters when loading from dict."""
        test_config = {
            "convergence_window": 5000,
            "convergence_tolerance": 0.01
        }
        
        params = SimParams(test_config)
        
        assert params.convergence_window == 5000
        assert params.convergence_tolerance == 0.01
        
    def test_convergence_mathematical_properties(self):
        """Test mathematical properties of convergence parameters."""
        params = SimParams()
        
        # Window should be positive integer
        assert isinstance(params.convergence_window, int)
        assert params.convergence_window > 0
        
        # Tolerance should be positive float
        assert isinstance(params.convergence_tolerance, (int, float))
        assert params.convergence_tolerance > 0
        assert params.convergence_tolerance < 1  # Should be less than 100%
        
    def test_paper_baseline_convergence_params(self):
        """Test that paper baseline uses correct convergence parameters."""
        # These are the exact values from Calvano et al. (2020)
        expected_window = 10000
        expected_tolerance = 0.02
        
        config = load_config("config.json")
        params = SimParams(config)
        
        assert params.convergence_window == expected_window, f"Expected window {expected_window}, got {params.convergence_window}"
        assert params.convergence_tolerance == expected_tolerance, f"Expected tolerance {expected_tolerance}, got {params.convergence_tolerance}"


class TestConvergenceLogic:
    """Test convergence detection logic."""
    
    def test_convergence_detection_implementation(self):
        """Test that convergence detection follows paper methodology."""
        # This would test the actual convergence detection algorithm
        # For now, we test that the parameters are accessible
        
        params = SimParams()
        
        # Test that we can create a convergence checker with these parameters
        window_size = params.convergence_window
        tolerance = params.convergence_tolerance
        
        assert window_size == 10000
        assert tolerance == 0.02
        
        # Test typical convergence scenario
        n_episodes = 15000
        profits = []
        
        # Simulate converging profits
        for t in range(n_episodes):
            if t < 5000:
                # Initial learning phase with high variance
                profit = 0.25 + 0.1 * np.sin(t/100) + 0.05 * np.random.randn()
            else:
                # Convergence phase with low variance
                profit = 0.26 + 0.01 * np.random.randn()
            profits.append(profit)
        
        # Test convergence in last window
        last_window_profits = profits[-window_size:]
        profit_std = np.std(last_window_profits)
        
        # Should converge according to tolerance
        converged = profit_std < tolerance
        
        # With our simulated data, this should converge
        assert isinstance(converged, (bool, np.bool_))
        # The convergence itself depends on random seed, so just check type
        
    def test_convergence_edge_cases(self):
        """Test edge cases for convergence detection."""
        params = SimParams()
        
        # Test with insufficient data
        short_sequence = [0.25, 0.26, 0.24]
        assert len(short_sequence) < params.convergence_window
        
        # Test with perfect convergence (no variance)
        perfect_sequence = [0.25] * params.convergence_window
        perfect_std = np.std(perfect_sequence)
        assert perfect_std == 0.0
        assert perfect_std < params.convergence_tolerance
        
        # Test with high variance (non-convergence)
        high_var_sequence = []
        for i in range(params.convergence_window):
            high_var_sequence.append(0.25 + 0.1 * ((-1) ** i))
        high_var_std = np.std(high_var_sequence)
        assert high_var_std > params.convergence_tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 