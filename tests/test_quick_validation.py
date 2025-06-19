#!/usr/bin/env python3
"""
Quick validation tests for core functionality.

Simple tests to verify basic implementation correctness.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from params import SimParams
from cooperative_benchmark import demand_function, profit_function


def test_demand_function_basic():
    """Test basic demand function properties."""
    print("\n=== BASIC DEMAND FUNCTION TEST ===")
    
    params = SimParams({
        'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 0.1,
        'demand_model': 'logit'
    })
    
    # Test equal prices
    d1, d2 = demand_function(1.0, 1.0, params)
    print(f"Equal prices (1.0, 1.0): d1={d1:.6f}, d2={d2:.6f}")
    assert abs(d1 - d2) < 1e-10, f"Equal prices should give equal demand"
    
    # Test monotonicity
    d1_low, _ = demand_function(0.5, 1.0, params)
    d1_high, _ = demand_function(1.5, 1.0, params)
    print(f"Lower price d1={d1_low:.6f} vs Higher price d1={d1_high:.6f}")
    assert d1_low > d1_high, f"Lower price should have higher demand"
    
    print("✅ Basic demand function test PASSED")


def test_profit_function_basic():
    """Test basic profit function properties."""
    print("\n=== BASIC PROFIT FUNCTION TEST ===")
    
    params = SimParams({
        'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 0.1,
        'demand_model': 'logit'
    })
    
    # Test price below cost gives negative profit
    profit1, profit2 = profit_function(0.1, 1.0, params)
    print(f"Price below cost (0.1): profit1={profit1:.6f}")
    assert profit1 < 0, f"Price below cost should give negative profit"
    
    # Test reasonable price gives positive profit
    profit1, profit2 = profit_function(1.0, 1.0, params)
    print(f"Reasonable price (1.0): profit1={profit1:.6f}")
    assert profit1 > 0, f"Price above cost should give positive profit"
    
    print("✅ Basic profit function test PASSED")


def test_extreme_parameters():
    """Test extreme parameter values."""
    print("\n=== EXTREME PARAMETERS TEST ===")
    
    # Test very small mu (deterministic)
    params_det = SimParams({
        'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 1e-8,
        'demand_model': 'logit'
    })
    
    d1, d2 = demand_function(0.5, 1.0, params_det)
    print(f"μ≈0: Lower price d1={d1:.6f}, Higher price d2={d2:.6f}")
    assert d1 > 0.9, f"With tiny μ, lower price should dominate"
    
    # Test large mu (random)
    params_rand = SimParams({
        'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 10.0,
        'demand_model': 'logit'
    })
    
    d1, d2 = demand_function(0.5, 1.0, params_rand)
    print(f"μ=10: d1={d1:.6f}, d2={d2:.6f}, diff={abs(d1-d2):.6f}")
    assert abs(d1 - d2) < 0.5, f"With large μ, demands should be closer"
    
    print("✅ Extreme parameters test PASSED")


def test_price_grid_boundaries():
    """Test price grid boundary behavior."""
    print("\n=== PRICE GRID BOUNDARIES TEST ===")
    
    params = SimParams({
        'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 0.1,
        'price_grid': [0.0, 0.5, 1.0, 1.5],
        'n_actions': 4,
        'n_states': 16
    })
    
    # Test all boundary price combinations
    for p1 in [0.0, 1.5]:  # Min and max prices
        for p2 in [0.0, 1.5]:
            d1, d2 = demand_function(p1, p2, params)
            profit1, profit2 = profit_function(p1, p2, params)
            
            # Check demands are valid
            assert 0 <= d1 <= 1, f"d1 out of bounds at p=({p1}, {p2})"
            assert 0 <= d2 <= 1, f"d2 out of bounds at p=({p1}, {p2})"
            assert np.isfinite(profit1), f"profit1 not finite at p=({p1}, {p2})"
            assert np.isfinite(profit2), f"profit2 not finite at p=({p1}, {p2})"
            
            print(f"p=({p1}, {p2}): d=({d1:.4f}, {d2:.4f}), π=({profit1:.4f}, {profit2:.4f})")
    
    print("✅ Price grid boundaries test PASSED")


def test_symmetry():
    """Test symmetry properties."""
    print("\n=== SYMMETRY TEST ===")
    
    params = SimParams({
        'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 0.1,
        'demand_model': 'logit'
    })
    
    # Test that swapping prices swaps demands and profits
    p1, p2 = 0.8, 1.2
    d1_a, d2_a = demand_function(p1, p2, params)
    profit1_a, profit2_a = profit_function(p1, p2, params)
    
    d1_b, d2_b = demand_function(p2, p1, params)  # Swapped
    profit1_b, profit2_b = profit_function(p2, p1, params)
    
    print(f"({p1}, {p2}): d=({d1_a:.6f}, {d2_a:.6f})")
    print(f"({p2}, {p1}): d=({d1_b:.6f}, {d2_b:.6f})")
    
    # Check symmetry
    assert abs(d1_a - d2_b) < 1e-10, f"Demand symmetry broken"
    assert abs(d2_a - d1_b) < 1e-10, f"Demand symmetry broken"
    assert abs(profit1_a - profit2_b) < 1e-10, f"Profit symmetry broken"
    assert abs(profit2_a - profit1_b) < 1e-10, f"Profit symmetry broken"
    
    print("✅ Symmetry test PASSED")


if __name__ == "__main__":
    tests = [
        test_demand_function_basic,
        test_profit_function_basic,
        test_extreme_parameters,
        test_price_grid_boundaries,
        test_symmetry,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🔍 Quick validation complete: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic validation tests PASSED!")
    else:
        print("⚠️  Some tests FAILED - check implementation!") 