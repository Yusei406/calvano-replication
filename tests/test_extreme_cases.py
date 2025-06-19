#!/usr/bin/env python3
"""
Extreme cases verification tests.

This module tests edge cases and extreme parameter values to detect
configuration errors that would manifest under extreme conditions.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qlearning import QLearningAgent
from environment import LogitEnvironment
from params import SimParams
from cooperative_benchmark import demand_function, profit_function


class TestExtremeCases:
    """Test suite for extreme parameter cases."""
    
    def test_extreme_learning_rate(self):
        """Test α=1 (full update) and α=0 (no learning)."""
        print("\n=== EXTREME LEARNING RATE TEST ===")
        
        # Test α=1: Q should change dramatically
        params_alpha1 = SimParams({
            'alpha': 1.0,  # Full update
            'epsilon': 0.0,  # No exploration
            'delta': 0.0,   # No discount
            'price_grid': [0.5, 1.0, 1.5],
            'n_actions': 3,
            'n_states': 9,
            'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 0.1
        })
        
        env = LogitEnvironment(params_alpha1)
        agent = QLearningAgent(params_alpha1)
        
        # Initialize Q-table to zeros
        agent.q_table.fill(0.0)
        initial_q = agent.q_table[0, 0]
        
        # Take one action with positive reward
        state = env.get_state(0.5, 1.0)  # Some arbitrary state
        action = 1  # Middle price
        env.step([action, action])  # Both agents take same action
        profit1, profit2 = env.get_last_profits()
        
        # Update Q-table
        next_state = env.get_state(env.last_prices[0], env.last_prices[1])
        agent.update_q_table(state, action, profit1, next_state)
        
        updated_q = agent.q_table[state, action]
        change = abs(updated_q - initial_q)
        
        print(f"α=1: Q change = {change:.6f} (reward={profit1:.6f})")
        assert change > 1e-6, f"With α=1, Q should change significantly, got change={change}"
        assert abs(updated_q - profit1) < 1e-10, f"With α=1, δ=0, Q should equal reward, got Q={updated_q}, reward={profit1}"
        
        # Test α=0: Q should not change
        params_alpha0 = SimParams({
            'alpha': 0.0,  # No learning
            'epsilon': 0.0,
            'delta': 0.0,
            'price_grid': [0.5, 1.0, 1.5],
            'n_actions': 3,
            'n_states': 9,
            'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 0.1
        })
        
        agent0 = QLearningAgent(params_alpha0)
        agent0.q_table.fill(5.0)  # Set to some non-zero value
        initial_q0 = agent0.q_table[0, 0]
        
        # Update should not change Q
        agent0.update_q_table(0, 0, 1.0, 1)  # Give positive reward
        after_q0 = agent0.q_table[0, 0]
        
        print(f"α=0: Q change = {abs(after_q0 - initial_q0):.10f}")
        assert abs(after_q0 - initial_q0) < 1e-15, f"With α=0, Q should not change, got {initial_q0} → {after_q0}"
        
        print("✅ Extreme learning rate test PASSED")
    
    def test_greedy_vs_exploration(self):
        """Test ε=0 (pure greedy) vs ε=1 (pure random)."""
        print("\n=== GREEDY VS EXPLORATION TEST ===")
        
        # Create agent with ε=0 (pure greedy)
        params_greedy = SimParams({
            'alpha': 0.1,
            'epsilon': 0.0,  # Pure greedy
            'delta': 0.9,
            'price_grid': [0.5, 1.0, 1.5],
            'n_actions': 3,
            'n_states': 9,
            'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 0.1
        })
        
        agent_greedy = QLearningAgent(params_greedy)
        
        # Set Q-table with clear best action
        agent_greedy.q_table.fill(0.0)
        state = 0
        best_action = 2  # Make action 2 clearly best
        agent_greedy.q_table[state, best_action] = 10.0
        agent_greedy.q_table[state, 0] = 1.0
        agent_greedy.q_table[state, 1] = 2.0
        
        # Test greedy selection multiple times
        actions = []
        for _ in range(20):
            action = agent_greedy.choose_action(state)
            actions.append(action)
        
        unique_actions = set(actions)
        print(f"ε=0: Actions chosen = {unique_actions}")
        assert len(unique_actions) == 1, f"Greedy should always choose same action, got {unique_actions}"
        assert actions[0] == best_action, f"Greedy should choose best action {best_action}, got {actions[0]}"
        
        # Test ε=1 (pure random)
        params_random = SimParams({
            'alpha': 0.1,
            'epsilon': 1.0,  # Pure random
            'delta': 0.9,
            'price_grid': [0.5, 1.0, 1.5],
            'n_actions': 3,
            'n_states': 9,
            'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 0.1
        })
        
        agent_random = QLearningAgent(params_random)
        agent_random.q_table = agent_greedy.q_table.copy()  # Same Q-values
        
        # Test random selection
        random_actions = []
        for _ in range(100):
            action = agent_random.choose_action(state)
            random_actions.append(action)
        
        unique_random = set(random_actions)
        print(f"ε=1: Actions chosen = {unique_random}")
        assert len(unique_random) >= 2, f"Random should explore multiple actions, got {unique_random}"
        
        print("✅ Greedy vs exploration test PASSED")
    
    def test_extreme_prices(self):
        """Test very high and very low prices."""
        print("\n=== EXTREME PRICES TEST ===")
        
        params = SimParams({
            'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 0.1,
            'demand_model': 'logit'
        })
        
        # Test very low prices
        p_low = 0.01
        d1_low, d2_low = demand_function(p_low, 1.0, params)
        profit1_low, profit2_low = profit_function(p_low, 1.0, params)
        
        print(f"Low price p1={p_low}: d1={d1_low:.6f}, profit1={profit1_low:.6f}")
        assert d1_low > 0.5, f"Very low price should have high demand, got {d1_low}"
        assert profit1_low < 0, f"Price below cost should give negative profit, got {profit1_low}"
        
        # Test very high prices
        p_high = 5.0
        d1_high, d2_high = demand_function(p_high, 1.0, params)
        profit1_high, profit2_high = profit_function(p_high, 1.0, params)
        
        print(f"High price p1={p_high}: d1={d1_high:.6f}, profit1={profit1_high:.6f}")
        assert d1_high < 0.1, f"Very high price should have low demand, got {d1_high}"
        
        # Test equal extreme prices
        p_extreme = 10.0
        d1_eq, d2_eq = demand_function(p_extreme, p_extreme, params)
        
        print(f"Equal extreme prices p=({p_extreme}, {p_extreme}): d1={d1_eq:.6f}, d2={d2_eq:.6f}")
        assert abs(d1_eq - d2_eq) < 1e-10, f"Equal prices should give equal demand, got d1={d1_eq}, d2={d2_eq}"
        
        print("✅ Extreme prices test PASSED")
    
    def test_mu_extreme_values(self):
        """Test extreme values of mu (noise parameter)."""
        print("\n=== EXTREME MU TEST ===")
        
        # Test μ → 0 (deterministic)
        params_det = SimParams({
            'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 1e-10,  # Nearly deterministic
            'demand_model': 'logit'
        })
        
        p1, p2 = 0.5, 1.0
        d1_det, d2_det = demand_function(p1, p2, params_det)
        
        print(f"μ≈0: p=({p1}, {p2}) → d=({d1_det:.6f}, {d2_det:.6f})")
        
        # Lower price should get almost all demand
        assert d1_det > 0.9, f"With low μ, lower price should dominate, got d1={d1_det}"
        assert d2_det < 0.1, f"With low μ, higher price should get little demand, got d2={d2_det}"
        
        # Test μ very large (random choice)
        params_random = SimParams({
            'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 100.0,  # Very noisy
            'demand_model': 'logit'
        })
        
        d1_random, d2_random = demand_function(p1, p2, params_random)
        
        print(f"μ=100: p=({p1}, {p2}) → d=({d1_random:.6f}, {d2_random:.6f})")
        
        # With high noise, demands should be closer to equal
        demand_diff = abs(d1_random - d2_random)
        assert demand_diff < 0.3, f"With high μ, demand difference should be small, got {demand_diff}"
        
        print("✅ Extreme mu test PASSED")
    
    def test_state_action_boundaries(self):
        """Test state and action boundary values."""
        print("\n=== STATE ACTION BOUNDARIES TEST ===")
        
        params = SimParams({
            'alpha': 0.1, 'epsilon': 0.1, 'delta': 0.9,
            'price_grid': [0.0, 0.5, 1.0],
            'n_actions': 3,
            'n_states': 9,
            'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 0.1
        })
        
        env = LogitEnvironment(params)
        agent = QLearningAgent(params)
        
        # Test boundary states
        for state in [0, params.n_states - 1]:
            for action in [0, params.n_actions - 1]:
                try:
                    q_value = agent.q_table[state, action]
                    assert np.isfinite(q_value), f"Q-value should be finite at boundary state={state}, action={action}"
                    
                    # Test action selection at boundaries
                    selected_action = agent.choose_action(state)
                    assert 0 <= selected_action < params.n_actions, \
                        f"Selected action {selected_action} out of bounds [0, {params.n_actions})"
                    
                except IndexError as e:
                    assert False, f"Index error at boundary state={state}, action={action}: {e}"
        
        # Test state encoding/decoding at boundaries
        for p1_idx in [0, len(params.price_grid) - 1]:
            for p2_idx in [0, len(params.price_grid) - 1]:
                p1 = params.price_grid[p1_idx]
                p2 = params.price_grid[p2_idx]
                
                state = env.get_state(p1, p2)
                assert 0 <= state < params.n_states, \
                    f"State {state} out of bounds for prices ({p1}, {p2})"
                
                decoded_p1, decoded_p2 = env.get_prices_from_state(state)
                assert abs(decoded_p1 - p1) < 1e-10, f"Price decoding error: {p1} → {decoded_p1}"
                assert abs(decoded_p2 - p2) < 1e-10, f"Price decoding error: {p2} → {decoded_p2}"
        
        print("✅ State action boundaries test PASSED")
    
    def test_convergence_detection(self):
        """Test convergence detection with extreme parameters."""
        print("\n=== CONVERGENCE DETECTION TEST ===")
        
        # Test immediate convergence (no learning)
        params_converged = SimParams({
            'alpha': 0.0,  # No learning
            'epsilon': 0.0,  # No exploration 
            'delta': 0.9,
            'price_grid': [0.5, 1.0],
            'n_actions': 2,
            'n_states': 4,
            'convergence_window': 10,
            'convergence_threshold': 0.01,
            'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 0.1
        })
        
        env = LogitEnvironment(params_converged)
        agent1 = QLearningAgent(params_converged)
        agent2 = QLearningAgent(params_converged)
        
        # Run simulation to test convergence
        prices_history = []
        for episode in range(20):
            # Get states
            if len(prices_history) == 0:
                state1 = state2 = 0
            else:
                last_p1, last_p2 = prices_history[-1]
                state1 = env.get_state(last_p1, last_p2)
                state2 = state1  # Same state for both agents
            
            # Choose actions
            action1 = agent1.choose_action(state1)
            action2 = agent2.choose_action(state2)
            
            # Execute actions
            env.step([action1, action2])
            prices_history.append((env.last_prices[0], env.last_prices[1]))
            
            # With α=0, ε=0, prices should be constant after first episode
            if episode >= 1:
                assert prices_history[-1] == prices_history[-2], \
                    f"With α=0,ε=0, prices should be constant, got {prices_history[-2]} → {prices_history[-1]}"
        
        print(f"✅ No-learning convergence: prices constant at {prices_history[-1]}")
        print("✅ Convergence detection test PASSED")


if __name__ == "__main__":
    # Run tests
    test_suite = TestExtremeCases()
    
    tests = [
        ("extreme_learning_rate", test_suite.test_extreme_learning_rate),
        ("greedy_vs_exploration", test_suite.test_greedy_vs_exploration),
        ("extreme_prices", test_suite.test_extreme_prices),
        ("mu_extreme_values", test_suite.test_mu_extreme_values),
        ("state_action_boundaries", test_suite.test_state_action_boundaries),
        ("convergence_detection", test_suite.test_convergence_detection),
    ]
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name} test PASSED")
        except Exception as e:
            print(f"❌ {test_name} test FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🚨 Extreme cases verification complete!") 