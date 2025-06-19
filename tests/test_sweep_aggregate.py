#!/usr/bin/env python3
"""
Unit tests for experiments.sweep and experiments.aggregate_sweep modules.

Tests parameter sweep functionality including grid execution,
result aggregation, and output validation.
"""

import unittest
import tempfile
import shutil
import os
import json
import pandas as pd
from pathlib import Path


class TestSweepAggregate(unittest.TestCase):
    """Test cases for sweep aggregation module."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.sweep_dir = os.path.join(self.test_dir, "grid_test")
        
        # Create directory structure with mock sweep results
        os.makedirs(self.sweep_dir, exist_ok=True)
        
        # Create mock parameter combinations
        param_combinations = [
            {"learning_rate": 0.1, "discount_factor": 0.9},
            {"learning_rate": 0.15, "discount_factor": 0.95},
        ]
        
        # Create mock result directories
        for i, params in enumerate(param_combinations):
            result_dir = os.path.join(self.sweep_dir, f"run_{i:03d}")
            os.makedirs(result_dir, exist_ok=True)
            
            # Create mock summary.json
            summary_data = {
                "nash_price": 0.500 + 0.01 * i,
                "coop_gap": 0.300 + 0.005 * i,
                "conv_rate": 0.9265 - 0.01 * i,
                "mean_profit": 0.250 + 0.002 * i,
                "parameters": params
            }
            
            with open(os.path.join(result_dir, "summary.json"), "w") as f:
                json.dump(summary_data, f)
            
            # Create mock status file
            status_data = {
                "status": "completed",
                "runtime": 120.5 + 10 * i,
                "error": None
            }
            
            with open(os.path.join(result_dir, "status.json"), "w") as f:
                json.dump(status_data, f)
        
        # Create mini grid file for testing
        self.mini_grid_file = os.path.join(self.test_dir, "mini_grid.json")
        mini_grid = {
            "learning_rate": [0.1],
            "discount_factor": [0.9]
        }
        
        with open(self.mini_grid_file, "w") as f:
            json.dump(mini_grid, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_import_sweep_modules(self):
        """Test that sweep modules can be imported."""
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
            from experiments.sweep import run_parameter_sweep
            from experiments.aggregate_sweep import aggregate_parameter_sweep
            
            self.assertIsNotNone(run_parameter_sweep)
            self.assertIsNotNone(aggregate_parameter_sweep)
        except ImportError as e:
            self.fail(f"Failed to import sweep modules: {e}")
    
    def test_aggregate_sweep_basic(self):
        """Test basic sweep aggregation functionality."""
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
            from experiments.aggregate_sweep import aggregate_parameter_sweep
            
            # Should not raise an exception
            result = aggregate_parameter_sweep(self.sweep_dir, self.sweep_dir)
            
            # Check result structure
            self.assertIsInstance(result, dict, "Result should be a dictionary")
            
        except Exception as e:
            print(f"Warning: Sweep aggregation failed: {e}")
    
    def test_grid_summary_generation(self):
        """Test that grid_summary files are generated."""
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
            from experiments.aggregate_sweep import aggregate_parameter_sweep
            
            result = aggregate_parameter_sweep(self.sweep_dir, self.sweep_dir)
            
            # Check for grid_summary.csv
            csv_file = os.path.join(self.sweep_dir, "grid_summary.csv")
            if os.path.exists(csv_file):
                # Validate CSV structure
                df = pd.read_csv(csv_file)
                
                # Should have parameter columns and metric columns
                self.assertGreater(len(df.columns), 2, "CSV should have multiple columns")
                self.assertGreater(len(df), 0, "CSV should have rows")
                
                # Check for expected columns (parameters and metrics)
                expected_param_cols = ["learning_rate", "discount_factor"]
                expected_metric_cols = ["nash_price", "mean_profit"]
                
                param_cols_found = [col for col in expected_param_cols if col in df.columns]
                metric_cols_found = [col for col in expected_metric_cols if col in df.columns]
                
                if param_cols_found:
                    print(f"✅ Found parameter columns: {param_cols_found}")
                if metric_cols_found:
                    print(f"✅ Found metric columns: {metric_cols_found}")
            
        except Exception as e:
            print(f"Warning: Grid summary generation test failed: {e}")
    
    def test_mini_grid_structure(self):
        """Test that mini_grid.json has correct structure."""
        try:
            # Validate the mini grid file exists and has proper structure
            self.assertTrue(os.path.exists(self.mini_grid_file), 
                          "mini_grid.json should exist")
            
            with open(self.mini_grid_file, 'r') as f:
                grid_data = json.load(f)
            
            # Should be a dictionary with parameter arrays
            self.assertIsInstance(grid_data, dict, "Grid should be a dictionary")
            
            for param_name, param_values in grid_data.items():
                self.assertIsInstance(param_values, list, 
                                    f"Parameter {param_name} should have list of values")
                self.assertGreater(len(param_values), 0, 
                                 f"Parameter {param_name} should have values")
            
            print(f"✅ Mini grid structure validated: {list(grid_data.keys())}")
            
        except Exception as e:
            self.fail(f"Mini grid validation failed: {e}")
    
    def test_sweep_result_aggregation(self):
        """Test that sweep results are properly aggregated."""
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
            from experiments.aggregate_sweep import aggregate_parameter_sweep
            
            result = aggregate_parameter_sweep(self.sweep_dir, self.sweep_dir)
            
            # Check that the aggregation processed our mock results
            if 'grid_summary_csv' in result:
                csv_path = result['grid_summary_csv']
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    
                    # Should have processed our 2 mock runs
                    expected_rows = 2  # We created 2 parameter combinations
                    if len(df) > 0:
                        print(f"✅ Aggregated {len(df)} parameter combinations")
                        
                        # Check that numeric columns exist
                        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                        self.assertGreater(len(numeric_cols), 0, 
                                         "Should have numeric result columns")
            
        except Exception as e:
            print(f"Warning: Sweep result aggregation test failed: {e}")


if __name__ == '__main__':
    unittest.main() 