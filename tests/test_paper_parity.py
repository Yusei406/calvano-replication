"""
Unit tests for paper parity comparison functionality.

Tests the compare_paper module for validating simulation results against
published values from Calvano et al. (2020).
"""

import json
import pytest
import tempfile
from pathlib import Path

from src.benchmark.compare_paper import compare_to_paper, PAPER_VALUES, validate_paper_format


def test_paper_parity_pass(tmp_path):
    """Test that exact paper values pass comparison."""
    data = {
        "nash_price": 0.500,
        "coop_gap": 0.300,
        "conv_rate": 0.9265,
        "mean_profit": 0.250
    }
    
    results_file = tmp_path / "ok.json"
    results_file.write_text(json.dumps(data))
    
    # Should not raise any exception
    compare_to_paper(str(results_file), eps=1e-3)


def test_paper_parity_fail(tmp_path):
    """Test that values outside tolerance fail comparison."""
    data = {
        "nash_price": 0.52,  # 0.02 > 1e-3 tolerance
        "coop_gap": 0.300,
        "conv_rate": 0.9265,
        "mean_profit": 0.250
    }
    
    results_file = tmp_path / "bad.json"
    results_file.write_text(json.dumps(data))
    
    with pytest.raises(AssertionError) as exc_info:
        compare_to_paper(str(results_file), eps=1e-3)
    
    assert "nash_price" in str(exc_info.value)
    assert "expected 0.5" in str(exc_info.value)
    assert "got 0.52" in str(exc_info.value)


def test_paper_parity_boundary(tmp_path):
    """Test values exactly at tolerance boundary."""
    data = {
        "nash_price": 0.5009,  # 0.0009 < 1e-3 tolerance
        "coop_gap": 0.2991,    # 0.0009 < 1e-3 tolerance
        "conv_rate": 0.9265,
        "mean_profit": 0.250
    }
    
    results_file = tmp_path / "boundary.json"
    results_file.write_text(json.dumps(data))
    
    # Should pass - within tolerance
    compare_to_paper(str(results_file), eps=1e-3)


def test_paper_parity_missing_keys(tmp_path):
    """Test that missing required keys raise KeyError."""
    data = {
        "nash_price": 0.500,
        "coop_gap": 0.300,
        # Missing conv_rate and mean_profit
    }
    
    results_file = tmp_path / "incomplete.json"
    results_file.write_text(json.dumps(data))
    
    with pytest.raises(KeyError) as exc_info:
        compare_to_paper(str(results_file), eps=1e-3)
    
    assert "Missing required keys" in str(exc_info.value)


def test_paper_parity_non_finite(tmp_path):
    """Test that non-finite values fail comparison."""
    data = {
        "nash_price": float('inf'),
        "coop_gap": 0.300,
        "conv_rate": 0.9265,
        "mean_profit": 0.250
    }
    
    results_file = tmp_path / "infinite.json"
    results_file.write_text(json.dumps(data))
    
    with pytest.raises(AssertionError) as exc_info:
        compare_to_paper(str(results_file), eps=1e-3)
    
    assert "non-finite value" in str(exc_info.value)


def test_paper_parity_file_not_found():
    """Test that missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        compare_to_paper("nonexistent.json", eps=1e-3)


def test_paper_parity_invalid_json(tmp_path):
    """Test that invalid JSON raises ValueError."""
    results_file = tmp_path / "invalid.json"
    results_file.write_text("{ invalid json content")
    
    with pytest.raises(ValueError) as exc_info:
        compare_to_paper(str(results_file), eps=1e-3)
    
    assert "Invalid JSON" in str(exc_info.value)


def test_validate_paper_format_valid(tmp_path):
    """Test validation of correct format."""
    data = {
        "nash_price": 0.500,
        "coop_gap": 0.300,
        "conv_rate": 0.9265,
        "mean_profit": 0.250,
        "extra_key": "ignored"  # Extra keys are OK
    }
    
    results_file = tmp_path / "valid.json"
    results_file.write_text(json.dumps(data))
    
    assert validate_paper_format(str(results_file)) is True


def test_validate_paper_format_invalid(tmp_path):
    """Test validation of incorrect format."""
    data = {
        "nash_price": 0.500,
        # Missing required keys
    }
    
    results_file = tmp_path / "invalid.json"
    results_file.write_text(json.dumps(data))
    
    assert validate_paper_format(str(results_file)) is False


def test_paper_values_constants():
    """Test that PAPER_VALUES constants are as expected."""
    expected = {
        "nash_price": 0.500,
        "coop_gap": 0.300,
        "conv_rate": 0.9265,
        "mean_profit": 0.250,
    }
    
    assert PAPER_VALUES == expected


def test_custom_tolerance(tmp_path):
    """Test using custom tolerance values."""
    data = {
        "nash_price": 0.505,  # 0.005 deviation
        "coop_gap": 0.300,
        "conv_rate": 0.9265,
        "mean_profit": 0.250
    }
    
    results_file = tmp_path / "custom_tol.json"
    results_file.write_text(json.dumps(data))
    
    # Should fail with strict tolerance
    with pytest.raises(AssertionError):
        compare_to_paper(str(results_file), eps=1e-3)
    
    # Should pass with relaxed tolerance
    compare_to_paper(str(results_file), eps=1e-2) 