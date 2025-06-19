#!/usr/bin/env python3
"""
Unit tests for unified profit calculation module.

Tests the canonical calc_profit function and demand computation functions
against hand-calculated values and mathematical properties.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from profit import calc_profit, calc_profit_vector, DemandParams, compute_demands_logit


class TestProfitCalculations(unittest.TestCase):
    """Test unified profit calculation functions."""
    
    def setUp(self):
        """Set up test parameters."""
        self.demand_params = DemandParams(
            a0=2.0,
            a=1.0,
            c=0.0,
            mu=0.25
        )
    
    def test_calc_profit_basic(self):
        """Test basic profit calculation."""
        # Hand calculation: profit = (price - cost) * demand
        price = 0.5
        cost = 0.1
        demand = 0.3
        
        expected_profit = (0.5 - 0.1) * 0.3  # = 0.12
        actual_profit = calc_profit(price, cost, demand)
        
        self.assertAlmostEqual(actual_profit, expected_profit, places=6)
    
    def test_calc_profit_zero_cost(self):
        """Test profit calculation with zero marginal cost."""
        price = 0.8
        cost = 0.0
        demand = 0.25
        
        expected_profit = 0.8 * 0.25  # = 0.2
        actual_profit = calc_profit(price, cost, demand)
        
        self.assertAlmostEqual(actual_profit, expected_profit, places=6)
    
    def test_calc_profit_vector(self):
        """Test vector profit calculation."""
        prices = np.array([0.5, 0.6])
        costs = np.array([0.1, 0.1])
        demands = np.array([0.3, 0.2])
        
        expected_profits = np.array([
            (0.5 - 0.1) * 0.3,  # = 0.12
            (0.6 - 0.1) * 0.2   # = 0.10
        ])
        
        actual_profits = calc_profit_vector(prices, costs, demands)
        
        np.testing.assert_array_almost_equal(actual_profits, expected_profits, decimal=6)
    
    def test_logit_demands_symmetric(self):
        """Test logit demand calculation for symmetric prices."""
        # Symmetric prices should yield equal demands
        prices = [0.5, 0.5]
        demands = compute_demands_logit(prices, self.demand_params)
        
        # Should be equal demands
        self.assertAlmostEqual(demands[0], demands[1], places=6)
        
        # Should sum to approximately 1 (market shares)
        self.assertAlmostEqual(sum(demands), 1.0, places=6)
    
    def test_logit_demands_price_sensitivity(self):
        """Test that higher prices lead to lower demand."""
        prices_low = [0.3, 0.5]
        prices_high = [0.7, 0.5]
        
        demands_low = compute_demands_logit(prices_low, self.demand_params)
        demands_high = compute_demands_logit(prices_high, self.demand_params)
        
        # Agent 0 has lower price in first case, so should have higher demand
        self.assertGreater(demands_low[0], demands_high[0])
        # Agent 1 should have correspondingly lower demand in first case
        self.assertLess(demands_low[1], demands_high[1])
    
    def test_logit_demands_market_share_property(self):
        """Test that logit demands always sum to 1."""
        test_cases = [
            [0.1, 0.9],
            [0.5, 0.5], 
            [0.3, 0.7],
            [0.0, 1.0]
        ]
        
        for prices in test_cases:
            demands = compute_demands_logit(prices, self.demand_params)
            self.assertAlmostEqual(sum(demands), 1.0, places=6, 
                                 msg=f"Failed for prices {prices}")
    
    def test_demand_params_extraction(self):
        """Test demand parameter extraction."""
        # This test would need actual SimParams implementation
        # For now, just test the DemandParams dataclass
        params = DemandParams(a0=1.5, a=0.8, c=0.05, mu=0.3)
        
        self.assertEqual(params.a0, 1.5)
        self.assertEqual(params.a, 0.8)
        self.assertEqual(params.c, 0.05)
        self.assertEqual(params.mu, 0.3)


class TestProfitMathematicalProperties(unittest.TestCase):
    """Test mathematical properties of profit functions."""
    
    def test_profit_linearity_in_demand(self):
        """Test that profit is linear in demand for fixed price and cost."""
        price = 0.6
        cost = 0.1
        
        demand1 = 0.2
        demand2 = 0.4  # Double the demand
        
        profit1 = calc_profit(price, cost, demand1)
        profit2 = calc_profit(price, cost, demand2)
        
        # Profit should double when demand doubles
        self.assertAlmostEqual(profit2, 2 * profit1, places=6)
    
    def test_profit_zero_at_cost(self):
        """Test that profit is zero when price equals cost."""
        cost = 0.3
        price = cost  # Price equals cost
        demand = 0.5
        
        profit = calc_profit(price, cost, demand)
        self.assertAlmostEqual(profit, 0.0, places=6)
    
    def test_profit_negative_below_cost(self):
        """Test that profit is negative when price is below cost."""
        cost = 0.5
        price = 0.3  # Below cost
        demand = 0.4
        
        profit = calc_profit(price, cost, demand)
        self.assertLess(profit, 0.0)


if __name__ == '__main__':
    unittest.main() 