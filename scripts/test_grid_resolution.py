#!/usr/bin/env python3
"""
Price grid resolution test.

Test the impact of price grid resolution on profit performance.
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


def test_grid_resolution(price_steps, max_episodes=10000, n_runs=5):
    """Test different price grid resolutions."""
    print(f"🟠 PRICE GRID RESOLUTION TEST")
    print(f"{'='*50}")
    print(f"Price steps: {price_steps}")
    print(f"Episodes: {max_episodes}, Runs: {n_runs}")
    
    results = {}
    overall_start = time.time()
    
    # Base parameters (using optimal μ=0.05)
    base_params = {
        'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 0.05,
        'demand_model': 'logit'
    }
    
    for step in price_steps:
        print(f"\n--- Testing price_step = {step:.3f} ---")
        step_start = time.time()
        
        # Create price grid based on step size
        max_price = 2.0
        n_prices = int(max_price / step) + 1
        price_grid = [round(i * step, 3) for i in range(n_prices)]
        n_actions = len(price_grid)
        
        print(f"Grid: {n_actions} actions (0.00 to {max_price:.2f}, step {step:.3f})")
        
        test_params = {
            **base_params,
            'price_grid': price_grid,
            'n_actions': n_actions,
            'n_states': n_actions ** 2
        }
        
        run_results = []
        
        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs}", end=' ')
            
            params = SimParams(test_params)
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
                
                # Record final 20% of episodes
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
        
        step_time = time.time() - step_start
        
        # Aggregate step statistics
        individual_profits = [r['individual_profit'] for r in run_results]
        joint_profits = [r['joint_profit'] for r in run_results]
        prices1 = [r['mean_price1'] for r in run_results]
        prices2 = [r['mean_price2'] for r in run_results]
        
        step_stats = {
            'price_step': step,
            'n_actions': n_actions,
            'n_runs': n_runs,
            'max_episodes': max_episodes,
            'mean_individual_profit': np.mean(individual_profits),
            'std_individual_profit': np.std(individual_profits),
            'mean_joint_profit': np.mean(joint_profits),
            'std_joint_profit': np.std(joint_profits),
            'mean_price1': np.mean(prices1),
            'mean_price2': np.mean(prices2),
            'execution_time_min': step_time / 60,
            'individual_target_rate': np.mean([p >= 0.18 for p in individual_profits]),
            'joint_target_rate': np.mean([p >= 0.26 for p in joint_profits])
        }
        
        results[step] = step_stats
        
        print(f"step={step:.3f} completed in {step_time/60:.1f} min")
        print(f"Individual: {step_stats['mean_individual_profit']:.4f} ± {step_stats['std_individual_profit']:.4f}")
        print(f"Joint: {step_stats['mean_joint_profit']:.4f} ± {step_stats['std_joint_profit']:.4f}")
        print(f"Prices: ({step_stats['mean_price1']:.3f}, {step_stats['mean_price2']:.3f})")
        print(f"Grid size: {n_actions} actions")
    
    total_time = time.time() - overall_start
    
    # Find optimal step
    optimal_individual = max(results.keys(), key=lambda step: results[step]['mean_individual_profit'])
    optimal_joint = max(results.keys(), key=lambda step: results[step]['mean_joint_profit'])
    
    print(f"\n{'='*50}")
    print(f"GRID RESOLUTION TEST RESULTS")
    print(f"{'='*50}")
    print(f"Optimal step (Individual): {optimal_individual:.3f} → {results[optimal_individual]['mean_individual_profit']:.4f}")
    print(f"Optimal step (Joint): {optimal_joint:.3f} → {results[optimal_joint]['mean_joint_profit']:.4f}")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    # Check resolution impact (profit difference between finest and coarsest)
    finest_step = min(price_steps)
    coarsest_step = max(price_steps)
    profit_diff = abs(results[finest_step]['mean_individual_profit'] - results[coarsest_step]['mean_individual_profit'])
    percent_diff = (profit_diff / max(results[finest_step]['mean_individual_profit'], results[coarsest_step]['mean_individual_profit'])) * 100
    
    print(f"Resolution impact: {profit_diff:.4f} profit difference ({percent_diff:.1f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)
    
    output_file = f"results/grid_resolution_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'resolution_results': results,
            'optimal_individual_step': optimal_individual,
            'optimal_joint_step': optimal_joint,
            'resolution_impact_percent': percent_diff,
            'test_parameters': {
                'price_steps': price_steps,
                'max_episodes': max_episodes,
                'n_runs': n_runs,
                'optimal_mu': 0.05
            },
            'execution_time_min': total_time / 60
        }, f, indent=2)
    
    print(f"📄 Results saved: {output_file}")
    
    # Create visualization
    create_resolution_plot(results, timestamp)
    
    return results


def create_resolution_plot(results, timestamp):
    """Create grid resolution visualization."""
    steps = sorted(results.keys())
    individual_profits = [results[step]['mean_individual_profit'] for step in steps]
    joint_profits = [results[step]['mean_joint_profit'] for step in steps]
    individual_stds = [results[step]['std_individual_profit'] for step in steps]
    joint_stds = [results[step]['std_joint_profit'] for step in steps]
    grid_sizes = [results[step]['n_actions'] for step in steps]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Individual profit vs step size
    ax1.errorbar(steps, individual_profits, yerr=individual_stds, 
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax1.axhline(y=0.18, color='red', linestyle='--', alpha=0.7, label='Target (0.18)')
    ax1.set_xlabel('Price Step Size', fontsize=12)
    ax1.set_ylabel('Individual Profit', fontsize=12)
    ax1.set_title('Individual Profit vs Price Step', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Joint profit vs step size
    ax2.errorbar(steps, joint_profits, yerr=joint_stds, 
                marker='s', capsize=5, capthick=2, linewidth=2, markersize=8, color='green')
    ax2.axhline(y=0.26, color='red', linestyle='--', alpha=0.7, label='Target (0.26)')
    ax2.set_xlabel('Price Step Size', fontsize=12)
    ax2.set_ylabel('Joint Profit', fontsize=12)
    ax2.set_title('Joint Profit vs Price Step', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Individual profit vs grid size
    ax3.errorbar(grid_sizes, individual_profits, yerr=individual_stds, 
                marker='^', capsize=5, capthick=2, linewidth=2, markersize=8, color='orange')
    ax3.axhline(y=0.18, color='red', linestyle='--', alpha=0.7, label='Target (0.18)')
    ax3.set_xlabel('Grid Size (Number of Actions)', fontsize=12)
    ax3.set_ylabel('Individual Profit', fontsize=12)
    ax3.set_title('Individual Profit vs Grid Size', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Joint profit vs grid size
    ax4.errorbar(grid_sizes, joint_profits, yerr=joint_stds, 
                marker='d', capsize=5, capthick=2, linewidth=2, markersize=8, color='purple')
    ax4.axhline(y=0.26, color='red', linestyle='--', alpha=0.7, label='Target (0.26)')
    ax4.set_xlabel('Grid Size (Number of Actions)', fontsize=12)
    ax4.set_ylabel('Joint Profit', fontsize=12)
    ax4.set_title('Joint Profit vs Grid Size', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    plot_file = f"results/grid_resolution_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved: {plot_file}")
    
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Grid Resolution Test')
    parser.add_argument('--steps', nargs='+', type=float, 
                       default=[0.02, 0.01, 0.005], 
                       help='Price step sizes to test')
    parser.add_argument('--episodes', type=int, default=10000, help='Episodes per run')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs')
    
    args = parser.parse_args()
    
    test_grid_resolution(args.steps, args.episodes, args.runs) 