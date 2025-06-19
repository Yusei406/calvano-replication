#!/usr/bin/env python3
"""
YAML scenario test runner.

Executes Given/When/Then scenarios from YAML files.
"""

import yaml
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from params import SimParams
from cooperative_benchmark import demand_function, profit_function


def load_scenarios(yaml_file="tests/test_scenarios.yaml"):
    """Load scenarios from YAML file."""
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('scenarios', [])
    except FileNotFoundError:
        print(f"❌ YAML file not found: {yaml_file}")
        return []
    except Exception as e:
        print(f"❌ Error loading YAML: {e}")
        return []


def run_scenario(scenario):
    """Run a single scenario."""
    name = scenario['name']
    given = scenario['given']
    when = scenario['when']
    then = scenario['then']
    
    print(f"\n--- Testing scenario: {name} ---")
    
    try:
        if name == "demand_equal_prices":
            # Setup parameters
            params = SimParams({
                'a0': given['a0'],
                'a': given['a'],
                'mu': given['mu'],
                'c': 0.5,
                'demand_model': 'logit'
            })
            
            # Execute
            d1, d2 = demand_function(when['p1'], when['p2'], params)
            
            # Verify
            tolerance = then['tolerance']
            assert abs(d1 - d2) < tolerance, f"Demands not equal: d1={d1}, d2={d2}"
            print(f"✅ Equal demands: d1={d1:.6f}, d2={d2:.6f}")
            
        elif name == "demand_monotonicity":
            params = SimParams({
                'a0': given['a0'],
                'a': given['a'],
                'mu': given['mu'],
                'c': 0.5,
                'demand_model': 'logit'
            })
            
            d1_low, _ = demand_function(when['p1_low'], when['p2_fixed'], params)
            d1_high, _ = demand_function(when['p1_high'], when['p2_fixed'], params)
            
            assert d1_low > d1_high, f"Monotonicity violated: d1_low={d1_low}, d1_high={d1_high}"
            print(f"✅ Monotonicity: d1({when['p1_low']})={d1_low:.6f} > d1({when['p1_high']})={d1_high:.6f}")
            
        elif name == "profit_below_cost":
            params = SimParams({
                'c': given['c'],
                'a0': 2.0, 'a': 1.0, 'mu': 0.1,
                'demand_model': 'logit'
            })
            
            profit1, _ = profit_function(when['p1'], when['p2'], params)
            
            assert profit1 < 0, f"Profit should be negative: profit1={profit1}"
            print(f"✅ Negative profit below cost: profit1={profit1:.6f}")
            
        elif name == "extreme_mu_deterministic":
            params = SimParams({
                'mu': given['mu'],
                'a0': 2.0, 'a': 1.0, 'c': 0.5,
                'demand_model': 'logit'
            })
            
            d1, d2 = demand_function(when['p1'], when['p2'], params)
            
            assert d1 > then['d1_greater_than'], f"d1={d1} not > {then['d1_greater_than']}"
            assert d2 < then['d2_less_than'], f"d2={d2} not < {then['d2_less_than']}"
            print(f"✅ Deterministic choice: d1={d1:.6f}, d2={d2:.6f}")
            
        else:
            print(f"⚠️  Scenario '{name}' not implemented")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Scenario '{name}' failed: {e}")
        return False


def main():
    """Main scenario runner."""
    print("=== YAML SCENARIO TESTING ===")
    
    # Create simple YAML content if file doesn't exist
    yaml_content = """scenarios:
  - name: "demand_equal_prices"
    given:
      a0: 2.0
      a: 1.0
      mu: 0.1
    when:
      p1: 1.0
      p2: 1.0
    then:
      demand_equal: true
      tolerance: 1e-12

  - name: "demand_monotonicity"
    given:
      a0: 2.0
      a: 1.0
      mu: 0.1
    when:
      p1_low: 0.5
      p1_high: 1.5
      p2_fixed: 1.0
    then:
      d1_low_greater_than_d1_high: true

  - name: "profit_below_cost"
    given:
      c: 0.5
    when:
      p1: 0.3
      p2: 1.0
    then:
      profit1_negative: true

  - name: "extreme_mu_deterministic"
    given:
      mu: 1e-6
    when:
      p1: 0.6
      p2: 1.2
    then:
      d1_greater_than: 0.9
      d2_less_than: 0.1
"""
    
    # Write YAML file
    with open('tests/test_scenarios.yaml', 'w') as f:
        f.write(yaml_content)
    
    # Load and run scenarios
    scenarios = load_scenarios()
    
    if not scenarios:
        print("❌ No scenarios loaded")
        return
    
    passed = 0
    total = len(scenarios)
    
    for scenario in scenarios:
        if run_scenario(scenario):
            passed += 1
    
    print(f"\n=== SCENARIO TESTING COMPLETE ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All scenarios PASSED!")
    else:
        print("⚠️  Some scenarios FAILED!")
    
    return passed == total


if __name__ == "__main__":
    main() 