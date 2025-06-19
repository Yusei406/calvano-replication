#!/usr/bin/env python3
"""
Symbolic vs Numeric verification tests.

This module uses SymPy to compute analytical expressions and compares
them against numerical implementations to detect fundamental discrepancies.
"""

import numpy as np
import sympy as sp
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cooperative_benchmark import demand_function, profit_function
from params import SimParams


class TestSymbolicVsNumeric:
    """Test suite for symbolic vs numeric verification."""
    
    def setup_method(self):
        """Setup test parameters."""
        self.params = SimParams({
            'a0': 2.0,
            'a': 1.0, 
            'c': 0.5,
            'mu': 0.1,
            'demand_model': 'logit'
        })
        
        # Define symbolic variables
        self.p1, self.p2 = sp.symbols('p1 p2', real=True, positive=True)
        self.a0, self.a, self.c, self.mu = sp.symbols('a0 a c mu', real=True, positive=True)
    
    def test_logit_demand_symbolic(self):
        """Test logit demand function against symbolic computation."""
        print("\n=== SYMBOLIC LOGIT DEMAND TEST ===")
        
        # Symbolic logit demand (Equation 1 from paper)
        # u1 = (a0 - a*p1 + mu*ln(p2)) / (1 + mu)
        # u2 = (a0 - a*p2 + mu*ln(p1)) / (1 + mu)
        # d1 = exp(u1) / (exp(u1) + exp(u2) + 1)
        
        u1 = (self.a0 - self.a*self.p1 + self.mu*sp.log(self.p2)) / (1 + self.mu)
        u2 = (self.a0 - self.a*self.p2 + self.mu*sp.log(self.p1)) / (1 + self.mu)
        
        exp1 = sp.exp(u1)
        exp2 = sp.exp(u2)
        exp0 = 1  # Outside option
        
        d1_sym = exp1 / (exp1 + exp2 + exp0)
        d2_sym = exp2 / (exp1 + exp2 + exp0)
        
        # Convert to numerical functions
        d1_func = sp.lambdify((self.p1, self.p2, self.a0, self.a, self.c, self.mu), d1_sym, 'numpy')
        d2_func = sp.lambdify((self.p1, self.p2, self.a0, self.a, self.c, self.mu), d2_sym, 'numpy')
        
        # Test points
        test_prices = [(0.5, 0.5), (0.3, 0.7), (1.0, 1.2), (0.8, 0.9)]
        
        for p1_val, p2_val in test_prices:
            # Symbolic computation
            d1_sym_val = d1_func(p1_val, p2_val, self.params.a0, self.params.a, 
                                self.params.c, self.params.mu)
            d2_sym_val = d2_func(p1_val, p2_val, self.params.a0, self.params.a, 
                                self.params.c, self.params.mu)
            
            # Numerical implementation
            d1_num, d2_num = demand_function(p1_val, p2_val, self.params)
            
            print(f"p=({p1_val}, {p2_val}): sym=({d1_sym_val:.6f}, {d2_sym_val:.6f}), "
                  f"num=({d1_num:.6f}, {d2_num:.6f})")
            
            # Assert equality within tolerance
            assert abs(d1_sym_val - d1_num) < 1e-10, f"Demand 1 mismatch at p=({p1_val}, {p2_val})"
            assert abs(d2_sym_val - d2_num) < 1e-10, f"Demand 2 mismatch at p=({p1_val}, {p2_val})"
    
    def test_profit_gradient_symbolic(self):
        """Test profit gradient against symbolic computation."""
        print("\n=== SYMBOLIC PROFIT GRADIENT TEST ===")
        
        # Symbolic profit function π1 = (p1 - c) * d1(p1, p2)
        u1 = (self.a0 - self.a*self.p1 + self.mu*sp.log(self.p2)) / (1 + self.mu)
        u2 = (self.a0 - self.a*self.p2 + self.mu*sp.log(self.p1)) / (1 + self.mu)
        
        exp1 = sp.exp(u1)
        exp2 = sp.exp(u2)
        exp0 = 1
        
        d1 = exp1 / (exp1 + exp2 + exp0)
        profit1 = (self.p1 - self.c) * d1
        
        # Compute gradient symbolically
        grad_p1 = sp.diff(profit1, self.p1)
        grad_func = sp.lambdify((self.p1, self.p2, self.a0, self.a, self.c, self.mu), 
                               grad_p1, 'numpy')
        
        # Test gradient at several points
        test_points = [(0.5, 0.5), (0.8, 0.9), (1.0, 1.1)]
        
        for p1_val, p2_val in test_points:
            # Symbolic gradient
            grad_sym = grad_func(p1_val, p2_val, self.params.a0, self.params.a, 
                               self.params.c, self.params.mu)
            
            # Numerical gradient (finite difference)
            eps = 1e-8
            profit_plus, _ = profit_function(p1_val + eps, p2_val, self.params)
            profit_minus, _ = profit_function(p1_val - eps, p2_val, self.params)
            grad_num = (profit_plus - profit_minus) / (2 * eps)
            
            print(f"p=({p1_val}, {p2_val}): grad_sym={grad_sym:.8f}, grad_num={grad_num:.8f}")
            
            # Assert gradient equality
            assert abs(grad_sym - grad_num) < 1e-6, f"Gradient mismatch at p=({p1_val}, {p2_val})"
    
    def test_demand_monotonicity(self):
        """Test demand monotonicity property."""
        print("\n=== DEMAND MONOTONICITY TEST ===")
        
        # Test that ∂d1/∂p1 < 0 (own-price effect negative)
        p_base = 0.8
        p_other = 0.9
        price_range = np.linspace(0.1, 1.4, 20)
        
        demands = []
        for p in price_range:
            d1, d2 = demand_function(p, p_other, self.params)
            demands.append(d1)
        
        # Check monotonicity
        for i in range(1, len(demands)):
            assert demands[i] <= demands[i-1] + 1e-10, \
                f"Demand not monotonic: d({price_range[i]})={demands[i]} > d({price_range[i-1]})={demands[i-1]}"
        
        print(f"✅ Demand monotonic: d1 decreases from {demands[0]:.4f} to {demands[-1]:.4f}")
    
    def test_profit_bounds(self):
        """Test profit bounds and sanity checks."""
        print("\n=== PROFIT BOUNDS TEST ===")
        
        # Test profit bounds for reasonable price ranges
        price_grid = np.linspace(0.1, 1.5, 15)
        
        for p1 in price_grid[::3]:  # Sample every 3rd point
            for p2 in price_grid[::3]:
                profit1, profit2 = profit_function(p1, p2, self.params)
                
                # Profits should be finite
                assert np.isfinite(profit1), f"Profit1 not finite at p=({p1}, {p2})"
                assert np.isfinite(profit2), f"Profit2 not finite at p=({p1}, {p2})"
                
                # Profits should be reasonable (not extremely large)
                assert abs(profit1) < 10.0, f"Profit1 too large: {profit1} at p=({p1}, {p2})"
                assert abs(profit2) < 10.0, f"Profit2 too large: {profit2} at p=({p1}, {p2})"
                
                # If price > marginal cost, profit could be positive
                if p1 > self.params.c:
                    # Don't enforce positivity as demand might be very low
                    pass
        
        print("✅ Profit bounds check passed")
    
    def test_demand_sum_bounds(self):
        """Test that demand sum is bounded appropriately."""
        print("\n=== DEMAND SUM BOUNDS TEST ===")
        
        # For logit model with outside option, d1 + d2 < 1
        test_points = [(0.5, 0.5), (0.1, 1.4), (1.0, 0.8), (1.5, 1.5)]
        
        for p1, p2 in test_points:
            d1, d2 = demand_function(p1, p2, self.params)
            total_demand = d1 + d2
            
            # Check bounds
            assert 0 <= d1 <= 1, f"d1 out of bounds: {d1} at p=({p1}, {p2})"
            assert 0 <= d2 <= 1, f"d2 out of bounds: {d2} at p=({p1}, {p2})"
            assert total_demand <= 1.01, f"Total demand too high: {total_demand} at p=({p1}, {p2})"
            
            print(f"p=({p1}, {p2}): d1={d1:.4f}, d2={d2:.4f}, total={total_demand:.4f}")
        
        print("✅ Demand sum bounds check passed")


if __name__ == "__main__":
    # Run tests
    test_suite = TestSymbolicVsNumeric()
    test_suite.setup_method()
    
    try:
        test_suite.test_logit_demand_symbolic()
        print("✅ Logit demand symbolic test PASSED")
    except Exception as e:
        print(f"❌ Logit demand symbolic test FAILED: {e}")
    
    try:
        test_suite.test_profit_gradient_symbolic()
        print("✅ Profit gradient symbolic test PASSED")
    except Exception as e:
        print(f"❌ Profit gradient symbolic test FAILED: {e}")
    
    try:
        test_suite.test_demand_monotonicity()
        print("✅ Demand monotonicity test PASSED")
    except Exception as e:
        print(f"❌ Demand monotonicity test FAILED: {e}")
    
    try:
        test_suite.test_profit_bounds()
        print("✅ Profit bounds test PASSED")
    except Exception as e:
        print(f"❌ Profit bounds test FAILED: {e}")
    
    try:
        test_suite.test_demand_sum_bounds()
        print("✅ Demand sum bounds test PASSED")
    except Exception as e:
        print(f"❌ Demand sum bounds test FAILED: {e}")
    
    print("\n🔍 Symbolic verification complete!") 