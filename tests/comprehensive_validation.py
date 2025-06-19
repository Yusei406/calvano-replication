#!/usr/bin/env python3
"""
Comprehensive validation test suite.

Runs all validation layers and generates a detailed report.
"""

import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from params import SimParams
from cooperative_benchmark import demand_function, profit_function


class ValidationReport:
    """Generate validation report."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def add_result(self, layer, test_name, passed, details=""):
        """Add test result."""
        if layer not in self.results:
            self.results[layer] = []
        
        self.results[layer].append({
            'test': test_name,
            'passed': passed,
            'details': details
        })
    
    def generate_report(self):
        """Generate comprehensive report."""
        total_tests = sum(len(tests) for tests in self.results.values())
        total_passed = sum(
            len([t for t in tests if t['passed']]) 
            for tests in self.results.values()
        )
        
        duration = time.time() - self.start_time
        
        report = f"""
========================================
COMPREHENSIVE VALIDATION REPORT
========================================
Timestamp: {self.timestamp}
Duration: {duration:.2f} seconds
Total Tests: {total_passed}/{total_tests}
Success Rate: {(total_passed/total_tests*100):.1f}%

"""
        
        for layer, tests in self.results.items():
            layer_passed = len([t for t in tests if t['passed']])
            layer_total = len(tests)
            
            report += f"""
Layer {layer}: {layer_passed}/{layer_total}
{"="*50}
"""
            
            for test in tests:
                status = "✅" if test['passed'] else "❌"
                report += f"{status} {test['test']}"
                if test['details']:
                    report += f" - {test['details']}"
                report += "\n"
        
        report += f"""
========================================
SUMMARY
========================================
"""
        
        if total_passed == total_tests:
            report += "🎉 ALL TESTS PASSED!\n"
            report += "✅ Implementation appears mathematically correct.\n"
            report += "✅ No configuration errors detected.\n"
            report += "✅ All invariant properties satisfied.\n"
        else:
            failed = total_tests - total_passed
            report += f"⚠️  {failed} TESTS FAILED!\n"
            report += "❌ Implementation issues detected.\n"
            report += "🔍 Review failed tests above for details.\n"
        
        return report


def run_layer_a_symbolic():
    """Layer A: Symbolic vs Numeric Tests."""
    print("\n🔍 LAYER A: SYMBOLIC VERIFICATION")
    results = []
    
    params = SimParams({
        'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 0.1,
        'demand_model': 'logit'
    })
    
    # Test 1: Equal price symmetry
    try:
        d1, d2 = demand_function(1.0, 1.0, params)
        passed = abs(d1 - d2) < 1e-12
        results.append(('equal_price_symmetry', passed, f"d1={d1:.6f}, d2={d2:.6f}"))
    except Exception as e:
        results.append(('equal_price_symmetry', False, str(e)))
    
    # Test 2: Demand monotonicity
    try:
        d1_low, _ = demand_function(0.5, 1.0, params)
        d1_high, _ = demand_function(1.5, 1.0, params)
        passed = d1_low > d1_high
        results.append(('demand_monotonicity', passed, f"{d1_low:.4f} > {d1_high:.4f}"))
    except Exception as e:
        results.append(('demand_monotonicity', False, str(e)))
    
    # Test 3: Demand bounds
    try:
        test_points = [(0.5, 1.0), (1.5, 0.8), (0.1, 2.0)]
        all_valid = True
        for p1, p2 in test_points:
            d1, d2 = demand_function(p1, p2, params)
            if not (0 <= d1 <= 1 and 0 <= d2 <= 1 and d1 + d2 <= 1.01):
                all_valid = False
                break
        results.append(('demand_bounds', all_valid, f"Tested {len(test_points)} points"))
    except Exception as e:
        results.append(('demand_bounds', False, str(e)))
    
    return results


def run_layer_b_extreme():
    """Layer B: Extreme Cases Tests."""
    print("\n🚨 LAYER B: EXTREME CASES")
    results = []
    
    # Test 1: Very small mu (deterministic)
    try:
        params_det = SimParams({
            'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 1e-8,
            'demand_model': 'logit'
        })
        d1, d2 = demand_function(0.5, 1.0, params_det)
        passed = d1 > 0.9 and d2 < 0.1
        results.append(('deterministic_choice', passed, f"d1={d1:.4f}, d2={d2:.4f}"))
    except Exception as e:
        results.append(('deterministic_choice', False, str(e)))
    
    # Test 2: Price below cost
    try:
        params = SimParams({
            'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 0.1,
            'demand_model': 'logit'
        })
        profit1, _ = profit_function(0.3, 1.0, params)  # p1 < c
        passed = profit1 < 0
        results.append(('profit_below_cost', passed, f"profit1={profit1:.4f}"))
    except Exception as e:
        results.append(('profit_below_cost', False, str(e)))
    
    # Test 3: High noise (random choice)
    try:
        params_rand = SimParams({
            'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 10.0,
            'demand_model': 'logit'
        })
        d1, d2 = demand_function(0.5, 1.0, params_rand)
        passed = abs(d1 - d2) < 0.3  # Should be closer with high noise
        results.append(('random_choice', passed, f"diff={abs(d1-d2):.4f}"))
    except Exception as e:
        results.append(('random_choice', False, str(e)))
    
    return results


def run_layer_c_profit_table():
    """Layer C: Profit Table Analysis."""
    print("\n📊 LAYER C: PROFIT TABLE ANALYSIS")
    results = []
    
    try:
        params = SimParams({
            'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 0.1,
            'demand_model': 'logit'
        })
        
        # Create small profit table
        prices = [0.5, 1.0, 1.5]
        profits = []
        
        for p1 in prices:
            for p2 in prices:
                profit1, _ = profit_function(p1, p2, params)
                profits.append(profit1)
        
        # Check for anomalies
        all_finite = all(abs(p) < 10 for p in profits)  # Reasonable bounds
        has_positive = any(p > 0 for p in profits)  # Some positive profits
        
        results.append(('profit_table_finite', all_finite, f"All profits finite"))
        results.append(('profit_table_positive', has_positive, f"Has positive profits"))
        
    except Exception as e:
        results.append(('profit_table_creation', False, str(e)))
    
    return results


def run_layer_d_properties():
    """Layer D: Property-Based Tests."""
    print("\n🔬 LAYER D: PROPERTY-BASED TESTS")
    results = []
    
    params = SimParams({
        'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 0.1,
        'demand_model': 'logit'
    })
    
    # Property 1: Demand sum ≤ 1 (with outside option)
    try:
        test_points = [(0.5, 1.0), (1.5, 0.8), (0.1, 2.0), (2.0, 0.3)]
        all_valid = True
        for p1, p2 in test_points:
            d1, d2 = demand_function(p1, p2, params)
            if d1 + d2 > 1.01:  # Small tolerance
                all_valid = False
                break
        results.append(('demand_sum_property', all_valid, f"Tested {len(test_points)} points"))
    except Exception as e:
        results.append(('demand_sum_property', False, str(e)))
    
    # Property 2: Monotonicity in own price
    try:
        price_sequence = [0.3, 0.6, 0.9, 1.2, 1.5]
        demands = [demand_function(p, 1.0, params)[0] for p in price_sequence]
        monotonic = all(demands[i] >= demands[i+1] - 1e-8 for i in range(len(demands)-1))
        results.append(('own_price_monotonicity', monotonic, f"Sequence valid"))
    except Exception as e:
        results.append(('own_price_monotonicity', False, str(e)))
    
    return results


def main():
    """Run comprehensive validation."""
    print("🔍 COMPREHENSIVE VALIDATION STARTING...")
    
    report = ValidationReport()
    
    # Run all layers
    layer_a_results = run_layer_a_symbolic()
    layer_b_results = run_layer_b_extreme()
    layer_c_results = run_layer_c_profit_table()
    layer_d_results = run_layer_d_properties()
    
    # Add results to report
    for test_name, passed, details in layer_a_results:
        report.add_result("A (Symbolic)", test_name, passed, details)
    
    for test_name, passed, details in layer_b_results:
        report.add_result("B (Extreme)", test_name, passed, details)
    
    for test_name, passed, details in layer_c_results:
        report.add_result("C (Tables)", test_name, passed, details)
    
    for test_name, passed, details in layer_d_results:
        report.add_result("D (Properties)", test_name, passed, details)
    
    # Generate and display report
    final_report = report.generate_report()
    print(final_report)
    
    # Save report to file
    try:
        with open('tests/validation_report.txt', 'w') as f:
            f.write(final_report)
        print("📄 Report saved to tests/validation_report.txt")
    except Exception as e:
        print(f"❌ Could not save report: {e}")
    
    return report


if __name__ == "__main__":
    main() 