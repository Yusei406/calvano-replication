#!/usr/bin/env python3
"""
Cost (c) and demand level (a0) sensitivity analysis.

Fast scan to understand model robustness across different economic parameters.
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


def run_parameter_sensitivity(param_name, param_values, max_episodes=15000, n_runs=3):
    """Run sensitivity analysis for a given parameter."""
    print(f"🟠 {param_name.upper()} SENSITIVITY SCAN")
    print(f"{'='*50}")
    print(f"{param_name} values: {param_values}")
    print(f"Episodes: {max_episodes}, Runs: {n_runs}")
    
    results = {}
    overall_start = time.time()
    
    # Common parameters (using optimal μ=0.05)
    price_grid = [round(x * 0.02, 2) for x in range(101)]  # 0.00 to 2.00, step 0.02
    n_actions = len(price_grid)
    
    base_params = {
        'price_grid': price_grid,
        'n_actions': n_actions,
        'n_states': n_actions ** 2,
        'a0': 2.0, 'a': 1.0, 'c': 0.5, 'mu': 0.05,  # Optimal μ
        'demand_model': 'logit'
    }
    
    for param_value in param_values:
        print(f"\n--- Testing {param_name} = {param_value:.2f} ---")
        param_start = time.time()
        
        run_results = []
        
        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs}", end=' ')
            
            # Update the specific parameter
            test_params = base_params.copy()
            test_params[param_name] = param_value
            
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
        
        param_time = time.time() - param_start
        
        # Aggregate parameter statistics
        individual_profits = [r['individual_profit'] for r in run_results]
        joint_profits = [r['joint_profit'] for r in run_results]
        prices1 = [r['mean_price1'] for r in run_results]
        prices2 = [r['mean_price2'] for r in run_results]
        
        param_stats = {
            param_name: param_value,
            'n_runs': n_runs,
            'max_episodes': max_episodes,
            'mean_individual_profit': np.mean(individual_profits),
            'std_individual_profit': np.std(individual_profits),
            'mean_joint_profit': np.mean(joint_profits),
            'std_joint_profit': np.std(joint_profits),
            'mean_price1': np.mean(prices1),
            'mean_price2': np.mean(prices2),
            'execution_time_min': param_time / 60,
            'individual_target_rate': np.mean([p >= 0.18 for p in individual_profits]),
            'joint_target_rate': np.mean([p >= 0.26 for p in joint_profits])
        }
        
        results[param_value] = param_stats
        
        print(f"{param_name}={param_value:.2f} completed in {param_time/60:.1f} min")
        print(f"Individual: {param_stats['mean_individual_profit']:.4f} ± {param_stats['std_individual_profit']:.4f}")
        print(f"Joint: {param_stats['mean_joint_profit']:.4f} ± {param_stats['std_joint_profit']:.4f}")
        print(f"Prices: ({param_stats['mean_price1']:.3f}, {param_stats['mean_price2']:.3f})")
        print(f"Target rates: Ind={param_stats['individual_target_rate']:.1%}, Joint={param_stats['joint_target_rate']:.1%}")
    
    total_time = time.time() - overall_start
    
    # Find optimal parameter value
    optimal_individual = max(results.keys(), key=lambda val: results[val]['mean_individual_profit'])
    optimal_joint = max(results.keys(), key=lambda val: results[val]['mean_joint_profit'])
    
    print(f"\n{'='*50}")
    print(f"{param_name.upper()} SENSITIVITY RESULTS")
    print(f"{'='*50}")
    print(f"Optimal {param_name} (Individual): {optimal_individual:.2f} → {results[optimal_individual]['mean_individual_profit']:.4f}")
    print(f"Optimal {param_name} (Joint): {optimal_joint:.2f} → {results[optimal_joint]['mean_joint_profit']:.4f}")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    # Check robustness (how many values achieve target)
    individual_success_rate = np.mean([results[val]['individual_target_rate'] > 0.5 for val in results.keys()])
    joint_success_rate = np.mean([results[val]['joint_target_rate'] > 0.5 for val in results.keys()])
    
    print(f"Robustness: {individual_success_rate:.1%} values achieve individual target")
    print(f"Robustness: {joint_success_rate:.1%} values achieve joint target")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)
    
    output_file = f"results/{param_name}_sensitivity_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'sensitivity_results': results,
            'optimal_individual': optimal_individual,
            'optimal_joint': optimal_joint,
            'robustness_individual': individual_success_rate,
            'robustness_joint': joint_success_rate,
            'scan_parameters': {
                'param_name': param_name,
                'param_values': param_values,
                'max_episodes': max_episodes,
                'n_runs': n_runs,
                'optimal_mu': 0.05
            },
            'execution_time_min': total_time / 60
        }, f, indent=2)
    
    print(f"📄 Results saved: {output_file}")
    
    # Create visualization
    create_sensitivity_plot(results, param_name, timestamp)
    
    return results


def create_sensitivity_plot(results, param_name, timestamp):
    """Create parameter sensitivity visualization."""
    param_values = sorted(results.keys())
    individual_profits = [results[val]['mean_individual_profit'] for val in param_values]
    joint_profits = [results[val]['mean_joint_profit'] for val in param_values]
    individual_stds = [results[val]['std_individual_profit'] for val in param_values]
    joint_stds = [results[val]['std_joint_profit'] for val in param_values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Individual profit plot
    ax1.errorbar(param_values, individual_profits, yerr=individual_stds, 
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax1.axhline(y=0.18, color='red', linestyle='--', alpha=0.7, label='Target (0.18)')
    ax1.set_xlabel(f'{param_name} parameter', fontsize=12)
    ax1.set_ylabel('Individual Profit', fontsize=12)
    ax1.set_title(f'Individual Profit vs {param_name} (Sensitivity)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Mark optimal point
    optimal_idx = np.argmax(individual_profits)
    ax1.plot(param_values[optimal_idx], individual_profits[optimal_idx], 
            'r*', markersize=15, label=f'Optimal {param_name}={param_values[optimal_idx]:.2f}')
    ax1.legend()
    
    # Joint profit plot
    ax2.errorbar(param_values, joint_profits, yerr=joint_stds, 
                marker='s', capsize=5, capthick=2, linewidth=2, markersize=8, color='green')
    ax2.axhline(y=0.26, color='red', linestyle='--', alpha=0.7, label='Target (0.26)')
    ax2.set_xlabel(f'{param_name} parameter', fontsize=12)
    ax2.set_ylabel('Joint Profit', fontsize=12)
    ax2.set_title(f'Joint Profit vs {param_name} (Sensitivity)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Mark optimal point
    optimal_idx = np.argmax(joint_profits)
    ax2.plot(param_values[optimal_idx], joint_profits[optimal_idx], 
            'r*', markersize=15, label=f'Optimal {param_name}={param_values[optimal_idx]:.2f}')
    ax2.legend()
    
    plt.tight_layout()
    
    plot_file = f"results/{param_name}_sensitivity_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved: {plot_file}")
    
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Parameter Sensitivity Analysis')
    parser.add_argument('--param', choices=['c', 'a0'], required=True,
                       help='Parameter to analyze (c=cost, a0=demand_level)')
    parser.add_argument('--values', nargs='+', type=float, required=True,
                       help='Parameter values to test')
    parser.add_argument('--episodes', type=int, default=15000, help='Episodes per run')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs')
    
    args = parser.parse_args()
    
    run_parameter_sensitivity(args.param, args.values, args.episodes, args.runs) 