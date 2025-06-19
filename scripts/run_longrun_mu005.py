#!/usr/bin/env python3
"""
μ=0.05 Long-run Verification Test.

Dedicated long-run test to verify if μ=0.05 can reach paper targets.
"""

import time
from datetime import datetime
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from params import SimParams
from qlearning import QLearningAgent
from environment import LogitEnvironment


def run_longrun_mu005(max_episodes=100000, n_runs=25):
    """Run long-run test with μ=0.05."""
    print(f"🔴 μ=0.05 LONG-RUN VERIFICATION")
    print(f"{'='*50}")
    print(f"Episodes: {max_episodes}, Runs: {n_runs}")
    print(f"Target: Individual ≥ 0.18, Joint ≥ 0.26")
    
    # Optimized parameters
    price_grid = [round(x * 0.02, 2) for x in range(101)]  # 0.00 to 2.00, step 0.02
    n_actions = len(price_grid)
    
    print(f"Grid: {n_actions} actions (0.00 to 2.00, step 0.02)")
    
    base_params = {
        'price_grid': price_grid,
        'n_actions': n_actions,
        'n_states': n_actions ** 2,
        'a0': 2.0, 'a': 1.0, 'c': 0.5,
        'mu': 0.05,  # Optimal μ value
        'demand_model': 'logit'
    }
    
    results = []
    overall_start = time.time()
    
    # Log start
    log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting μ=0.05 long-run: {max_episodes}×{n_runs}\n"
    with open('longrun_mu005.log', 'w') as f:
        f.write(log_entry)
    
    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")
        run_start = time.time()
        
        params = SimParams(base_params)
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
            
            # Record final 10% of episodes
            if episode >= int(max_episodes * 0.9):
                profits.append((profit1, profit2))
                prices.append(info['prices'])
            
            agent1.update_q_value(state, action1, profit1, next_state)
            agent2.update_q_value(state, action2, profit2, next_state)
            
            state = next_state
            
            # Progress logging
            if episode % 10000 == 0 and episode > 0:
                recent_profits = profits[-100:] if len(profits) >= 100 else profits
                if recent_profits:
                    recent_mean = np.mean([p[0] + p[1] for p in recent_profits])
                    print(f"Episode {episode}: Recent joint profit = {recent_mean:.4f}")
        
        run_time = time.time() - run_start
        
        # Calculate run statistics
        if profits:
            profit1_mean = np.mean([p[0] for p in profits])
            profit2_mean = np.mean([p[1] for p in profits])
            joint_mean = profit1_mean + profit2_mean
            profit_std = np.std([p[0] for p in profits])
            
            price1_mean = np.mean([p[0] for p in prices]) if prices else 0
            price2_mean = np.mean([p[1] for p in prices]) if prices else 0
        else:
            profit1_mean = profit2_mean = joint_mean = profit_std = 0.0
            price1_mean = price2_mean = 0.0
        
        result = {
            'run': run + 1,
            'individual_profit': profit1_mean,
            'joint_profit': joint_mean,
            'profit_std': profit_std,
            'mean_price1': price1_mean,
            'mean_price2': price2_mean,
            'run_time_min': run_time / 60,
            'episodes': max_episodes
        }
        results.append(result)
        
        print(f"Run {run + 1} completed in {run_time/60:.1f} min")
        print(f"Individual: {profit1_mean:.4f}, Joint: {joint_mean:.4f}")
        print(f"Prices: ({price1_mean:.3f}, {price2_mean:.3f})")
        
        # Intermediate logging every 5 runs
        if (run + 1) % 5 == 0:
            current_results = results[:run+1]
            joint_profits = [r['joint_profit'] for r in current_results]
            individual_profits = [r['individual_profit'] for r in current_results]
            
            mean_joint = np.mean(joint_profits)
            mean_individual = np.mean(individual_profits)
            std_joint = np.std(joint_profits)
            
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Run {run+1}/{n_runs}: Joint={mean_joint:.4f}±{std_joint:.4f}, Individual={mean_individual:.4f}\n"
            with open('longrun_mu005.log', 'a') as f:
                f.write(log_entry)
            
            print(f"Current average: Joint={mean_joint:.4f}±{std_joint:.4f}")
    
    total_time = time.time() - overall_start
    
    # Final analysis
    joint_profits = [r['joint_profit'] for r in results]
    individual_profits = [r['individual_profit'] for r in results]
    
    final_stats = {
        'mu': 0.05,
        'n_runs': n_runs,
        'max_episodes': max_episodes,
        'mean_individual_profit': np.mean(individual_profits),
        'std_individual_profit': np.std(individual_profits),
        'mean_joint_profit': np.mean(joint_profits),
        'std_joint_profit': np.std(joint_profits),
        'individual_target_rate': np.mean([p >= 0.18 for p in individual_profits]),
        'joint_target_rate': np.mean([p >= 0.26 for p in joint_profits]),
        'total_time_hours': total_time / 3600,
        'grid_size': n_actions,
        'price_step': 0.02
    }
    
    print(f"\n{'='*50}")
    print(f"μ=0.05 LONG-RUN FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Individual profit: {final_stats['mean_individual_profit']:.4f} ± {final_stats['std_individual_profit']:.4f}")
    print(f"Joint profit: {final_stats['mean_joint_profit']:.4f} ± {final_stats['std_joint_profit']:.4f}")
    print(f"Individual ≥ 0.18: {final_stats['individual_target_rate']:.1%}")
    print(f"Joint ≥ 0.26: {final_stats['joint_target_rate']:.1%}")
    print(f"Total time: {final_stats['total_time_hours']:.2f} hours")
    
    # Target evaluation
    individual_target = final_stats['mean_individual_profit'] >= 0.18
    joint_target = final_stats['mean_joint_profit'] >= 0.26
    
    print(f"\n--- TARGET ACHIEVEMENT ---")
    print(f"Individual ≥ 0.18: {'✅' if individual_target else '❌'}")
    print(f"Joint ≥ 0.26: {'✅' if joint_target else '❌'}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)
    
    output_file = f"results/longrun_mu005_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'final_stats': final_stats,
            'individual_results': results,
            'parameters': base_params
        }, f, indent=2)
    
    print(f"\n📄 Results saved: {output_file}")
    
    # Final log
    log_final = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] μ=0.05 COMPLETE: Individual={final_stats['mean_individual_profit']:.4f}, Joint={final_stats['mean_joint_profit']:.4f}\n"
    with open('longrun_mu005.log', 'a') as f:
        f.write(log_final)
    
    return final_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='μ=0.05 Long-run Verification')
    parser.add_argument('--episodes', type=int, default=100000, help='Episodes per run')
    parser.add_argument('--runs', type=int, default=25, help='Number of runs')
    
    args = parser.parse_args()
    
    run_longrun_mu005(args.episodes, args.runs) 