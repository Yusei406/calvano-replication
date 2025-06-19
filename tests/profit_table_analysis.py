#!/usr/bin/env python3
"""
Profit table analysis script for detecting anomalies.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from params import SimParams
from cooperative_benchmark import demand_function, profit_function


def create_profit_table():
    """Create profit table π_ij = π(p_i, p_j)."""
    print("=== CREATING PROFIT TABLE ===")
    
    params = SimParams({
        'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 0.1,
        'demand_model': 'logit'
    })
    
    # Price grid
    prices = np.arange(0.1, 2.1, 0.1)  # 0.1 to 2.0, step 0.1
    n = len(prices)
    
    profit1_table = np.zeros((n, n))
    
    for i, p1 in enumerate(prices):
        for j, p2 in enumerate(prices):
            profit1, profit2 = profit_function(p1, p2, params)
            profit1_table[i, j] = profit1
    
    # Create DataFrame
    price_labels = [f"{p:.1f}" for p in prices]
    df = pd.DataFrame(profit1_table, index=price_labels, columns=price_labels)
    df.index.name = "Price_1"
    df.columns.name = "Price_2"
    
    print(f"Created {n}x{n} profit table")
    
    # Basic statistics
    print(f"Max profit: {df.values.max():.4f}")
    print(f"Min profit: {df.values.min():.4f}")
    print(f"Mean profit: {df.values.mean():.4f}")
    
    # Find maximum
    max_idx = np.unravel_index(np.argmax(df.values), df.values.shape)
    max_p1, max_p2 = prices[max_idx[0]], prices[max_idx[1]]
    print(f"Maximum at p=({max_p1:.1f}, {max_p2:.1f}): {df.values[max_idx]:.4f}")
    
    # Check diagonal (equal prices)
    diagonal = np.diag(df.values)
    max_diag_idx = np.argmax(diagonal)
    print(f"Best equal price: p={prices[max_diag_idx]:.1f}, profit={diagonal[max_diag_idx]:.4f}")
    
    return df, prices


def detect_anomalies(df):
    """Detect anomalies in profit table."""
    print("\n=== ANOMALY DETECTION ===")
    
    values = df.values
    anomalies = []
    
    # Check for NaN/infinite
    if not np.all(np.isfinite(values)):
        anomalies.append("Non-finite values detected")
    
    # Check for negative profits where unexpected
    high_price_mask = np.array([[float(df.index[i]) > 0.5 and float(df.columns[j]) > 0.5 
                                for j in range(len(df.columns))] 
                               for i in range(len(df.index))])
    
    negative_high_prices = values < 0
    if np.any(negative_high_prices & high_price_mask):
        count = np.sum(negative_high_prices & high_price_mask)
        anomalies.append(f"{count} negative profits at high prices")
    
    # Check for large jumps
    threshold = 0.05
    row_jumps = np.max(np.abs(np.diff(values, axis=1)))
    col_jumps = np.max(np.abs(np.diff(values, axis=0)))
    
    if row_jumps > threshold:
        anomalies.append(f"Large row jump: {row_jumps:.4f}")
    if col_jumps > threshold:
        anomalies.append(f"Large column jump: {col_jumps:.4f}")
    
    print(f"Found {len(anomalies)} anomalies:")
    for anomaly in anomalies:
        print(f"  - {anomaly}")
    
    return anomalies


def main():
    """Main analysis."""
    df, prices = create_profit_table()
    anomalies = detect_anomalies(df)
    
    # Try to save to CSV (Excel alternative)
    try:
        df.to_csv("tests/profit_table.csv")
        print("\n✅ Profit table saved to tests/profit_table.csv")
    except Exception as e:
        print(f"\n❌ Could not save table: {e}")
    
    if not anomalies:
        print("\n✅ No significant anomalies detected")
    else:
        print(f"\n⚠️  {len(anomalies)} anomalies detected")
    
    return df, anomalies


if __name__ == "__main__":
    main() 