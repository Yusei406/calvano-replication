#!/usr/bin/env python3
"""
Unit tests for cooperative gap data type handling.

Tests that the cooperative benchmark calculation handles various data types
correctly and returns proper float values.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from profit import DemandParams
from cooperative_benchmark import (
    compute_cooperation_gap,
    compute_theoretical_benchmarks,
    analyze_cooperation_gap_detailed
)


class TestCooperationGapDataTypes(unittest.TestCase):
    """Test data type handling in cooperation gap calculations."""
    
    def setUp(self):
        """Set up test parameters."""
        self.demand_params = DemandParams(
            a0=1.0,
            a=1.0,
            c=0.0,
            mu=0.25
        )
    
    def test_cooperation_gap_float_inputs(self):
        """Test cooperation gap with float inputs."""
        observed = 0.3
        nash = 0.25
        coop = 0.4
        
        gap = compute_cooperation_gap(observed, nash, coop)
        
        self.assertIsInstance(gap, float)
        self.assertAlmostEqual(gap, (0.3 - 0.25) / (0.4 - 0.25), places=6)
    
    def test_cooperation_gap_array_inputs(self):
        """Test cooperation gap with numpy array inputs."""
        observed = np.array([0.3])
        nash = np.array([0.25])
        coop = np.array([0.4])
        
        gap = compute_cooperation_gap(observed[0], nash[0], coop[0])
        
        self.assertIsInstance(gap, float)
        self.assertAlmostEqual(gap, (0.3 - 0.25) / (0.4 - 0.25), places=6)
    
    def test_cooperation_gap_list_inputs(self):
        """Test cooperation gap with list inputs (should convert to float)."""
        observed = [0.3]
        nash = [0.25]
        coop = [0.4]
        
        gap = compute_cooperation_gap(observed[0], nash[0], coop[0])
        
        self.assertIsInstance(gap, float)
        self.assertAlmostEqual(gap, (0.3 - 0.25) / (0.4 - 0.25), places=6)
    
    def test_theoretical_benchmarks_return_types(self):
        """Test that theoretical benchmarks return proper float types."""
        benchmarks = compute_theoretical_benchmarks(self.demand_params)
        
        # Check that all values are floats
        self.assertIsInstance(benchmarks['nash_price'], float)
        self.assertIsInstance(benchmarks['coop_price'], float)
        self.assertIsInstance(benchmarks['nash_profit'], float)
        self.assertIsInstance(benchmarks['coop_profit'], float)
        
        # Check that demands are lists of floats
        self.assertIsInstance(benchmarks['nash_demands'], list)
        self.assertIsInstance(benchmarks['coop_demands'], list)
        for demand in benchmarks['nash_demands']:
            self.assertIsInstance(demand, (float, np.floating))
        for demand in benchmarks['coop_demands']:
            self.assertIsInstance(demand, (float, np.floating))
    
    def test_analyze_cooperation_gap_detailed_types(self):
        """Test detailed cooperation gap analysis with various input types."""
        # Mock simulation results with different data types
        observed_results = [
            {'final_profits': [0.3, 0.32]},  # list
            {'final_profits': np.array([0.28, 0.29])},  # numpy array
            {'final_profits': (0.31, 0.33)},  # tuple
        ]
        
        result = analyze_cooperation_gap_detailed(observed_results, self.demand_params)
        
        # Check that all outputs are floats
        for key, value in result.items():
            self.assertIsInstance(value, float, f"Key {key} is not float: {type(value)}")
    
    def test_cooperation_gap_edge_cases(self):
        """Test cooperation gap calculation edge cases."""
        # Case 1: Nash equals cooperative (zero denominator)
        gap = compute_cooperation_gap(0.3, 0.25, 0.25)
        self.assertEqual(gap, 0.0)
        
        # Case 2: Observed below Nash
        gap = compute_cooperation_gap(0.2, 0.25, 0.4)
        self.assertGreaterEqual(gap, 0.0)  # Should be clamped to 0
        
        # Case 3: Observed above cooperative
        gap = compute_cooperation_gap(0.5, 0.25, 0.4)
        self.assertLessEqual(gap, 2.0)  # Should be clamped to 2
    
    def test_no_list_subtraction_error(self):
        """Test that list subtraction error is resolved."""
        # This was the original error: "unsupported operand type(s) for -: 'list' and 'list'"
        benchmarks = compute_theoretical_benchmarks(self.demand_params)
        
        # These operations should not raise TypeError
        nash_profit = benchmarks['nash_profit']
        coop_profit = benchmarks['coop_profit']
        
        # This should work without error
        difference = coop_profit - nash_profit
        self.assertIsInstance(difference, float)
        self.assertGreater(difference, 0)  # Cooperative profit should be higher


class TestCooperationGapMathematicalProperties(unittest.TestCase):
    """Test mathematical properties of cooperation gap."""
    
    def setUp(self):
        """Set up test parameters."""
        self.demand_params = DemandParams(a0=1.0, a=1.0, c=0.0, mu=0.25)
    
    def test_gap_monotonicity(self):
        """Test that gap increases with observed profit."""
        nash = 0.25
        coop = 0.4
        
        low_observed = 0.3
        high_observed = 0.35
        
        gap_low = compute_cooperation_gap(low_observed, nash, coop)
        gap_high = compute_cooperation_gap(high_observed, nash, coop)
        
        self.assertLess(gap_low, gap_high)
    
    def test_gap_bounds(self):
        """Test cooperation gap boundary conditions."""
        nash = 0.25
        coop = 0.4
        
        # At Nash equilibrium
        gap_nash = compute_cooperation_gap(nash, nash, coop)
        self.assertAlmostEqual(gap_nash, 0.0, places=6)
        
        # At cooperative equilibrium
        gap_coop = compute_cooperation_gap(coop, nash, coop)
        self.assertAlmostEqual(gap_coop, 1.0, places=6)


if __name__ == '__main__':
    unittest.main() 