#!/usr/bin/env python3
"""
Precision μ scan to identify optimal μ value with ±0.01 accuracy.

Fast scan with 20k episodes × 5 runs for each μ value.
"""

import time
from datetime import datetime
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from params import SimParams
from qlearning import QLearningAgent
from environment import LogitEnvironment


def run_mu_precision_scan(mu_values, max_episodes=20000, n_runs=5):
    """Run precision μ scan with multiple values."""
    print(f"🔴 μ PRECISION SCAN")
    print(f"{'='*50}")
    print(f"μ values: {mu_values}")
    print(f"Episodes: {max_episodes}, Runs: {n_runs}")
    
    results = {}
    overall_start = time.time()
    
    # Common parameters
    price_grid = [round(x * 0.02, 2) for x in range(101)]  # 0.00 to 2.00, step 0.02
    n_actions = len(price_grid)
    
    base_params = {
        'price_grid': price_grid,
        'n_actions': n_actions,
        'n_states': n_actions ** 2,
        'a0': 2.0, 'a': 1.0, 'c': 0.5,
        'demand_model': 'logit'
    }
    
    for mu in mu_values:
        print(f"\n--- Testing μ = {mu:.3f} ---")
        mu_start = time.time()
        
        run_results = []
        
        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs}", end=' ')
            
            params = SimParams({**base_params, 'mu': mu})
            env = LogitEnvironment(params)
            agent1 = QLearningAgent(0, n_actions, n_actions**2, 0.20, 0.95, 0.05)
            agent2 = QLearningAgent(1, n_actions, n_actions**2, 0.20, 0.95, 0.05)
            
            profits = []
            prices = []
            state = 0
            
            for episode in range(max_episodes):
                action1 = agent1.select_action(state)
                action2 = agent2.select_action(state)
                
                next_state, rewards, done, info = env.step([action1, action2])
                profit1, profit2 = rewards[0], rewards[1]
                
                # Record final 20% of episodes (for faster scan)
                if episode >= int(max_episodes * 0.8):
                    profits.append((profit1, profit2))
                    prices.append(info['prices'])
                
                agent1.update_q_value(state, action1, profit1, next_state)
                agent2.update_q_value(state, action2, profit2, next_state)
                
                state = next_state
            
            # Calculate run statistics
            if profits:
                profit1_mean = np.mean([p[0] for p in profits])
                profit2_mean = np.mean([p[1] for p in profits])
                joint_mean = profit1_mean + profit2_mean
                
                price1_mean = np.mean([p[0] for p in prices]) if prices else 0
                price2_mean = np.mean([p[1] for p in prices]) if prices else 0
            else:
                profit1_mean = profit2_mean = joint_mean = 0.0
                price1_mean = price2_mean = 0.0
            
            run_results.append({
                'individual_profit': profit1_mean,
                'joint_profit': joint_mean,
                'mean_price1': price1_mean,
                'mean_price2': price2_mean
            })
            
            print(f"Profit: {profit1_mean:.4f}")
        
        mu_time = time.time() - mu_start
        
        # Aggregate μ statistics
        individual_profits = [r['individual_profit'] for r in run_results]
        joint_profits = [r['joint_profit'] for r in run_results]
        prices1 = [r['mean_price1'] for r in run_results]
        prices2 = [r['mean_price2'] for r in run_results]
        
        mu_stats = {
            'mu': mu,
            'n_runs': n_runs,
            'max_episodes': max_episodes,
            'mean_individual_profit': np.mean(individual_profits),
            'std_individual_profit': np.std(individual_profits),
            'mean_joint_profit': np.mean(joint_profits),
            'std_joint_profit': np.std(joint_profits),
            'mean_price1': np.mean(prices1),
            'mean_price2': np.mean(prices2),
            'execution_time_min': mu_time / 60,
            'individual_target_rate': np.mean([p >= 0.18 for p in individual_profits]),
            'joint_target_rate': np.mean([p >= 0.26 for p in joint_profits])
        }
        
        results[mu] = mu_stats
        
        print(f"μ={mu:.3f} completed in {mu_time/60:.1f} min")
        print(f"Individual: {mu_stats['mean_individual_profit']:.4f} ± {mu_stats['std_individual_profit']:.4f}")
        print(f"Joint: {mu_stats['mean_joint_profit']:.4f} ± {mu_stats['std_joint_profit']:.4f}")
        print(f"Prices: ({mu_stats['mean_price1']:.3f}, {mu_stats['mean_price2']:.3f})")
        print(f"Target rates: Ind={mu_stats['individual_target_rate']:.1%}, Joint={mu_stats['joint_target_rate']:.1%}")
    
    total_time = time.time() - overall_start
    
    # Find optimal μ
    optimal_mu_individual = max(results.keys(), key=lambda mu: results[mu]['mean_individual_profit'])
    optimal_mu_joint = max(results.keys(), key=lambda mu: results[mu]['mean_joint_profit'])
    
    print(f"\n{'='*50}")
    print(f"μ PRECISION SCAN RESULTS")
    print(f"{'='*50}")
    print(f"Optimal μ (Individual): {optimal_mu_individual:.3f} → {results[optimal_mu_individual]['mean_individual_profit']:.4f}")
    print(f"Optimal μ (Joint): {optimal_mu_joint:.3f} → {results[optimal_mu_joint]['mean_joint_profit']:.4f}")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)
    
    output_file = f"results/mu_precision_scan_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'scan_results': results,
            'optimal_mu_individual': optimal_mu_individual,
            'optimal_mu_joint': optimal_mu_joint,
            'scan_parameters': {
                'mu_values': mu_values,
                'max_episodes': max_episodes,
                'n_runs': n_runs,
                'price_grid_size': len(price_grid),
                'price_step': 0.02
            },
            'execution_time_min': total_time / 60
        }, f, indent=2)
    
    print(f"📄 Results saved: {output_file}")
    
    # Create visualization
    create_mu_precision_plot(results, timestamp)
    
    return results, optimal_mu_individual, optimal_mu_joint


def create_mu_precision_plot(results, timestamp):
    """Create μ precision scan visualization."""
    mu_values = sorted(results.keys())
    individual_profits = [results[mu]['mean_individual_profit'] for mu in mu_values]
    joint_profits = [results[mu]['mean_joint_profit'] for mu in mu_values]
    individual_stds = [results[mu]['std_individual_profit'] for mu in mu_values]
    joint_stds = [results[mu]['std_joint_profit'] for mu in mu_values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Individual profit plot
    ax1.errorbar(mu_values, individual_profits, yerr=individual_stds, 
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax1.axhline(y=0.18, color='red', linestyle='--', alpha=0.7, label='Target (0.18)')
    ax1.set_xlabel('μ (noise parameter)', fontsize=12)
    ax1.set_ylabel('Individual Profit', fontsize=12)
    ax1.set_title('Individual Profit vs μ (Precision Scan)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Find and mark optimal point
    optimal_idx = np.argmax(individual_profits)
    ax1.plot(mu_values[optimal_idx], individual_profits[optimal_idx], 
            'r*', markersize=15, label=f'Optimal μ={mu_values[optimal_idx]:.3f}')
    ax1.legend()
    
    # Joint profit plot
    ax2.errorbar(mu_values, joint_profits, yerr=joint_stds, 
                marker='s', capsize=5, capthick=2, linewidth=2, markersize=8, color='green')
    ax2.axhline(y=0.26, color='red', linestyle='--', alpha=0.7, label='Target (0.26)')
    ax2.set_xlabel('μ (noise parameter)', fontsize=12)
    ax2.set_ylabel('Joint Profit', fontsize=12)
    ax2.set_title('Joint Profit vs μ (Precision Scan)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Find and mark optimal point
    optimal_idx = np.argmax(joint_profits)
    ax2.plot(mu_values[optimal_idx], joint_profits[optimal_idx], 
            'r*', markersize=15, label=f'Optimal μ={mu_values[optimal_idx]:.3f}')
    ax2.legend()
    
    plt.tight_layout()
    
    plot_file = f"results/mu_precision_scan_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved: {plot_file}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='μ Precision Scan')
    parser.add_argument('--mu_values', nargs='+', type=float, 
                       default=[0.02, 0.03, 0.04, 0.05, 0.06], 
                       help='μ values to test')
    parser.add_argument('--episodes', type=int, default=20000, help='Episodes per run')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs')
    
    args = parser.parse_args()
    
    run_mu_precision_scan(args.mu_values, args.episodes, args.runs) 