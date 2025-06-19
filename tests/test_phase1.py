"""
Unit tests for Phase 1 implementations:
- L'Ecuyer RNG
- Q initialization strategies
- Convergence detection
- dtype policy
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

# Import using the module names without src prefix after adding to path
from rng.Lecuyer import (
    get_rng, get_lecuyer_raw, LecuyerCombined,
    set_global_rng, get_global_rng
)
from init.QInit import (
    init_Q, init_all_agents_Q, validate_Q_matrix,
    STRATEGY_TYPES
)
from convergence import (
    has_converged, compute_nash_distance, compute_cooperative_distance,
    analyze_convergence
)
from params import SimParams
import dtype_policy as dp


class TestLecuyerRNG(unittest.TestCase):
    """Test L'Ecuyer random number generator."""
    
    def test_rng_repeatable(self):
        """Test that same seed produces same sequence."""
        # Test with numpy Generator interface
        rng1 = get_rng(12345)
        rng2 = get_rng(12345)
        
        # Generate sequences
        seq1 = [rng1.random() for _ in range(100)]
        seq2 = [rng2.random() for _ in range(100)]
        
        # Should be identical
        np.testing.assert_array_equal(seq1, seq2)
    
    def test_rng_different_seeds(self):
        """Test that different seeds produce different sequences."""
        rng1 = get_rng(12345)
        rng2 = get_rng(12346)
        
        # Generate sequences
        seq1 = [rng1.random() for _ in range(100)]
        seq2 = [rng2.random() for _ in range(100)]
        
        # Should be different
        self.assertFalse(np.array_equal(seq1, seq2))
    
    def test_raw_lecuyer(self):
        """Test raw L'Ecuyer generator with ran2() method."""
        rng1 = get_lecuyer_raw(12345)
        rng2 = get_lecuyer_raw(12345)
        
        # Generate sequences using ran2()
        seq1 = [rng1.ran2() for _ in range(100)]
        seq2 = [rng2.ran2() for _ in range(100)]
        
        # Should be identical
        np.testing.assert_array_almost_equal(seq1, seq2, decimal=15)
    
    def test_range_validity(self):
        """Test that generated numbers are in valid range [0, 1)."""
        rng = get_lecuyer_raw(12345)
        
        for _ in range(1000):
            val = rng.ran2()
            self.assertGreaterEqual(val, 0.0)
            self.assertLess(val, 1.0)
    
    def test_thread_safety(self):
        """Test thread-local RNG functionality."""
        # Set global RNG
        set_global_rng(12345, 0)
        rng1 = get_global_rng()
        
        # Set different global RNG
        set_global_rng(12345, 1)
        rng2 = get_global_rng()
        
        # Should produce different values due to different rank
        val1 = rng1.random()
        val2 = rng2.random()
        self.assertNotEqual(val1, val2)


class TestQInitialization(unittest.TestCase):
    """Test Q-value initialization strategies."""
    
    def setUp(self):
        """Set up test parameters."""
        self.params = SimParams(
            n_agents=2,
            n_prices=11,
            state_depth=1,
            q_strategy="R"
        )
        self.rng = get_lecuyer_raw(12345)
    
    def test_qinit_shapes(self):
        """Test that all strategies produce correct Q matrix shapes."""
        expected_shape = (self.params.n_states, self.params.n_actions)
        
        for strategy in STRATEGY_TYPES:
            with self.subTest(strategy=strategy):
                if strategy == 'T':
                    # Skip pre-trained strategy (requires file)
                    continue
                
                Q = init_Q(strategy, self.params, self.rng)
                self.assertEqual(Q.shape, expected_shape)
                self.assertEqual(Q.dtype, np.float64)
    
    def test_fixed_strategy(self):
        """Test fixed strategy initialization."""
        Q = init_Q('F', self.params, self.rng, fixed_price=0.5)
        
        # Fixed action should be 5 (middle of 0-10 range)
        fixed_action = 5
        
        # All states should have finite values for fixed action
        self.assertTrue(np.isfinite(Q[:, fixed_action]).all())
        
        # Other actions should have -inf values
        for action in range(self.params.n_actions):
            if action != fixed_action:
                self.assertTrue(np.all(Q[:, action] == -np.inf))
    
    def test_random_strategy(self):
        """Test random strategy initialization."""
        Q = init_Q('R', self.params, self.rng, min_val=0.0, max_val=1.0)
        
        # All values should be finite and in range
        self.assertTrue(np.isfinite(Q).all())
        self.assertTrue(np.all(Q >= 0.0))
        self.assertTrue(np.all(Q <= 1.0))
    
    def test_uniform_strategy(self):
        """Test uniform strategy initialization."""
        constant_val = 0.5
        Q = init_Q('U', self.params, self.rng, constant_value=constant_val)
        
        # All values should equal the constant
        self.assertTrue(np.all(Q == constant_val))
    
    def test_grim_trigger_strategy(self):
        """Test Grim Trigger strategy initialization."""
        Q = init_Q('G', self.params, self.rng, 
                  coop_price=0.8, punish_price=0.0,
                  coop_value=1.0, punish_value=-1.0)
        
        # Should have finite values
        self.assertTrue(np.isfinite(Q).all())
        
        # Cooperation action should generally have higher values
        coop_action = int(0.8 * (self.params.n_actions - 1))
        punish_action = int(0.0 * (self.params.n_actions - 1))
        
        # Most states should prefer cooperation
        mean_coop = np.mean(Q[:, coop_action])
        mean_punish = np.mean(Q[:, punish_action])
        self.assertGreater(mean_coop, mean_punish)
    
    def test_opponent_random_strategy(self):
        """Test opponent random strategy initialization."""
        Q = init_Q('O', self.params, self.rng, random_scale=1.0)
        
        # Should have finite values
        self.assertTrue(np.isfinite(Q).all())
        
        # Should have some variation (not all zeros)
        self.assertGreater(np.std(Q), 0.1)
    
    def test_all_agents_initialization(self):
        """Test initialization for multiple agents."""
        strategies = ['R', 'F']
        Q_matrices = init_all_agents_Q(
            strategies, self.params, self.rng,
            fixed_price=[0.3, 0.7]
        )
        
        self.assertEqual(len(Q_matrices), 2)
        
        # Check shapes
        for Q in Q_matrices:
            validate_Q_matrix(Q, self.params)
    
    def test_validation(self):
        """Test Q matrix validation."""
        # Valid matrix
        Q_valid = np.random.rand(self.params.n_states, self.params.n_actions).astype(np.float64)
        self.assertTrue(validate_Q_matrix(Q_valid, self.params))
        
        # Invalid shape
        Q_invalid_shape = np.random.rand(5, 5)
        with self.assertRaises(ValueError):
            validate_Q_matrix(Q_invalid_shape, self.params)
        
        # Invalid dtype
        Q_invalid_dtype = np.random.rand(self.params.n_states, self.params.n_actions).astype(np.float32)
        with self.assertRaises(ValueError):
            validate_Q_matrix(Q_invalid_dtype, self.params)
        
        # Non-finite values
        Q_nonfinite = Q_valid.copy()
        Q_nonfinite[0, 0] = np.inf
        with self.assertRaises(ValueError):
            validate_Q_matrix(Q_nonfinite, self.params)


class TestConvergence(unittest.TestCase):
    """Test convergence detection logic."""
    
    def setUp(self):
        """Set up test parameters."""
        self.params = SimParams(n_agents=2, n_prices=11)
    
    def test_convergence_flag(self):
        """Test convergence detection with artificial data."""
        # Create convergent price history
        n_steps = 1500
        window = 1000
        tol = 1e-4
        
        # First part: changing prices
        changing_prices = np.random.rand(500, 2) * 0.5 + 0.25
        
        # Second part: stable prices
        stable_price = 0.5
        stable_prices = np.full((n_steps - 500, 2), stable_price)
        
        # Combine
        price_hist = np.vstack([changing_prices, stable_prices])
        
        # Should detect convergence
        self.assertTrue(has_converged(price_hist, window, tol))
        
        # Test non-convergent case
        non_convergent = np.random.rand(n_steps, 2)
        self.assertFalse(has_converged(non_convergent, window, tol))
    
    def test_nash_distance(self):
        """Test Nash equilibrium distance calculation."""
        # Test with exact Nash prices
        nash_price = 0.5  # Simplified
        nash_prices = np.array([nash_price, nash_price])
        distance = compute_nash_distance(nash_prices, self.params)
        self.assertAlmostEqual(distance, 0.0, places=6)
        
        # Test with non-Nash prices
        non_nash_prices = np.array([0.3, 0.7])
        distance = compute_nash_distance(non_nash_prices, self.params)
        self.assertGreater(distance, 0.0)
    
    def test_cooperative_distance(self):
        """Test cooperative equilibrium distance calculation."""
        # Test with exact cooperative prices
        coop_price = 0.8  # Simplified
        coop_prices = np.array([coop_price, coop_price])
        distance = compute_cooperative_distance(coop_prices, self.params)
        self.assertAlmostEqual(distance, 0.0, places=6)
        
        # Test with non-cooperative prices
        non_coop_prices = np.array([0.3, 0.4])
        distance = compute_cooperative_distance(non_coop_prices, self.params)
        self.assertGreater(distance, 0.0)
    
    def test_comprehensive_analysis(self):
        """Test comprehensive convergence analysis."""
        # Create artificial data
        price_hist = np.array([[0.5, 0.5]] * 1500)
        strategy_hist = [np.array([[1, 1], [2, 2]])] * 1500
        
        result = analyze_convergence(price_hist, strategy_hist, self.params)
        
        # Check result structure
        expected_keys = [
            'price_converged', 'strategy_converged', 'overall_converged',
            'final_prices', 'nash_distance', 'coop_distance',
            'final_strategy', 'convergence_time', 'final_volatility'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Should detect convergence
        self.assertTrue(result['overall_converged'])
        self.assertIsNotNone(result['convergence_time'])


class TestDtypePolicy(unittest.TestCase):
    """Test data type policy."""
    
    def test_dtype_constants(self):
        """Test that dtype constants are correct."""
        self.assertEqual(dp.DTYPE, np.float64)
        self.assertEqual(dp.INT_DTYPE, np.int64)
        self.assertEqual(dp.INT32_DTYPE, np.int32)
    
    def test_array_creation(self):
        """Test array creation functions."""
        shape = (10, 5)
        
        # Test zeros
        arr = dp.zeros(shape)
        self.assertEqual(arr.shape, shape)
        self.assertEqual(arr.dtype, dp.DTYPE)
        self.assertTrue(np.all(arr == 0))
        
        # Test ones
        arr = dp.ones(shape)
        self.assertEqual(arr.shape, shape)
        self.assertEqual(arr.dtype, dp.DTYPE)
        self.assertTrue(np.all(arr == 1))
        
        # Test full
        arr = dp.full(shape, 3.14)
        self.assertEqual(arr.shape, shape)
        self.assertEqual(arr.dtype, dp.DTYPE)
        self.assertTrue(np.all(arr == 3.14))
    
    def test_dtype_validation(self):
        """Test dtype validation."""
        # Valid array
        arr = np.array([1.0, 2.0, 3.0], dtype=dp.DTYPE)
        dp.validate_array_dtype(arr)  # Should not raise
        
        # Invalid array
        arr_invalid = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            dp.validate_array_dtype(arr_invalid)
    
    def test_safe_division(self):
        """Test safe division function."""
        # Normal division
        result = dp.safe_division(10.0, 2.0)
        self.assertEqual(result, 5.0)
        
        # Division by zero
        result = dp.safe_division(10.0, 0.0, default=999.0)
        self.assertEqual(result, 999.0)
        
        # Division by very small number
        result = dp.safe_division(10.0, 1e-16, default=999.0)
        self.assertEqual(result, 999.0)
    
    def test_equality_functions(self):
        """Test equality comparison functions."""
        # Close numbers
        self.assertTrue(dp.is_close(1.0, 1.0 + 1e-13))
        self.assertFalse(dp.is_close(1.0, 1.0 + 1e-10))
        
        self.assertTrue(dp.are_equal_reals(1.0, 1.0 + 1e-13))
        self.assertFalse(dp.are_equal_reals(1.0, 1.0 + 1e-10))


if __name__ == '__main__':
    # Create test suite
    test_classes = [
        TestLecuyerRNG,
        TestQInitialization,
        TestConvergence,
        TestDtypePolicy
    ]
    
    loader = unittest.TestLoader()
    suites = [loader.loadTestsFromTestCase(test_class) for test_class in test_classes]
    combined_suite = unittest.TestSuite(suites)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(combined_suite)
    
    # Exit with error code if tests failed
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code) 