"""
Q-learning implementation for Calvano et al. (2020) replication
"""

import numpy as np
from typing import Dict, List, Optional, Any
from params import SimParams, load_config

# Handle imports for both package and standalone usage
try:
    from .environment import CalvanoEnvironment
    from .environment_logit import LogitEnvironment
    from .demand import DemandCalculator
except ImportError:
    from environment import CalvanoEnvironment
    from environment_logit import LogitEnvironment  


class QLearningAgent:
    """Q-learning agent implementation."""
    
    def __init__(self, agent_id: int, n_actions: int, n_states: int, 
                 learning_rate: float = 0.1, discount_factor: float = 0.95,
                 epsilon: float = 0.1, q_init_strategy: str = "R"):
        """
        Initialize Q-learning agent.
        
        Args:
            agent_id: Unique identifier for this agent
            n_actions: Number of possible actions
            n_states: Number of possible states
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma/delta)
            epsilon: Exploration rate for epsilon-greedy
            q_init_strategy: Q-table initialization strategy
        """
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.n_states = n_states
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table based on strategy
        self.q_table = self._initialize_q_table(q_init_strategy)
        
        # Track statistics
        self.episode_rewards = []
        self.episode_actions = []
        
    def _initialize_q_table(self, strategy: str) -> np.ndarray:
        """Initialize Q-table using specified strategy."""
        if strategy == "R":
            # Random initialization 
            return np.random.random((self.n_states, self.n_actions))
        elif strategy == "Z":
            # Zero initialization
            return np.zeros((self.n_states, self.n_actions))
        elif strategy == "O":
            # Ones initialization
            return np.ones((self.n_states, self.n_actions))
        else:
            raise ValueError(f"Unknown Q-table initialization strategy: {strategy}")
    
    def select_action(self, state: int, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            explore: Whether to use exploration (epsilon-greedy)
            
        Returns:
            Selected action
        """
        if explore and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploitation: best action based on Q-values
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state: int, action: int, reward: float, 
                      next_state: int, done: bool = False) -> None:
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Q-learning update
        current_q = self.q_table[state, action]
        self.q_table[state, action] = current_q + self.learning_rate * (target - current_q)
    
    def get_policy(self) -> np.ndarray:
        """Get deterministic policy based on current Q-values."""
        return np.argmax(self.q_table, axis=1)
    
    def get_action_probabilities(self, state: int) -> np.ndarray:
        """Get action probabilities for given state (for analysis)."""
        q_values = self.q_table[state]
        # Softmax for probability distribution
        exp_q = np.exp(q_values - np.max(q_values))  # Numerical stability
        return exp_q / np.sum(exp_q)


class QLearningSimulation:
    """Main simulation class for Q-learning experiments."""
    
    def __init__(self, params: Optional[SimParams] = None, config_path: str = "config.json"):
        """
        Initialize simulation with parameters.
        
        Args:
            params: SimParams object (optional)
            config_path: Path to config file (used if params is None)
        """
        if params is None:
            config = load_config(config_path)
            self.params = SimParams(config)
        else:
            self.params = params
            
        # Create environment
        if self.params.demand_model == "logit":
            self.env = LogitEnvironment(self.params)
        else:
            self.env = CalvanoEnvironment(self.params)
            
        # Create agents
        self.agents = []
        for i in range(self.params.n_agents):
            agent = QLearningAgent(
                agent_id=i,
                n_actions=self.params.n_actions,
                n_states=self.params.n_states,
                learning_rate=self.params.alpha,
                discount_factor=self.params.delta,
                epsilon=self.params.epsilon,
                q_init_strategy=self.params.q_init_strategy
            )
            self.agents.append(agent)
            
        # Initialize logging
        self.episode_logs = []
        self.convergence_logs = []
        
    def run_episode(self) -> Dict[str, Any]:
        """
        Run a single episode of the game.
        
        Returns:
            Episode statistics dictionary
        """
        # Reset environment
        state = self.env.reset()
        
        # Agent actions
        actions = []
        for agent in self.agents:
            action = agent.select_action(state)
            actions.append(action)
            
        # Environment step
        next_state, rewards, done, info = self.env.step(actions)
        
        # Update Q-values for all agents
        for i, agent in enumerate(self.agents):
            agent.update_q_value(state, actions[i], rewards[i], next_state, done)
            
        # Collect episode statistics
        episode_stats = {
            'state': state,
            'actions': actions,
            'rewards': rewards,
            'next_state': next_state,
            'prices': info.get('prices', []),
            'demands': info.get('demands', []),
            'profits': rewards  # Profits are the rewards
        }
        
        return episode_stats
    
    def run_simulation(self, n_episodes: int) -> Dict[str, Any]:
        """
        Run complete simulation for specified number of episodes.
        
        Args:
            n_episodes: Number of episodes to run
            
        Returns:
            Simulation results dictionary
        """
        print(f"Running simulation for {n_episodes} episodes...")
        
        all_episode_stats = []
        
        for episode in range(n_episodes):
            episode_stats = self.run_episode()
            all_episode_stats.append(episode_stats)
            
            # Log progress
            if (episode + 1) % 10000 == 0:
                avg_profit = np.mean([stats['profits'][0] for stats in all_episode_stats[-1000:]])
                print(f"Episode {episode + 1}: Average profit = {avg_profit:.4f}")
                
        # Compile results
        results = {
            'params': self.params.to_dict(),
            'n_episodes': n_episodes,
            'episode_stats': all_episode_stats,
            'final_q_tables': [agent.q_table.copy() for agent in self.agents],
            'final_policies': [agent.get_policy() for agent in self.agents],
            'convergence_analysis': self._analyze_convergence(all_episode_stats)
        }
        
        return results
    
    def _analyze_convergence(self, episode_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze convergence properties of the simulation."""
        if len(episode_stats) < self.params.conv_window:
            return {
                'converged': False, 
                'convergence_rate': 0.0,
                'message': 'Insufficient episodes for convergence analysis'
            }
            
        # Extract profits from last window (without double discounting)
        window_profits = []
        for stats in episode_stats[-self.params.conv_window:]:
            # Use raw profits without additional discounting
            profits = stats['profits']
            if isinstance(profits, list) and len(profits) > 0:
                # Take average profit across agents for this episode
                avg_profit = sum(profits) / len(profits)
                window_profits.append(avg_profit)
            elif isinstance(profits, (int, float)):
                window_profits.append(profits)
        
        if not window_profits:
            return {
                'converged': False,
                'convergence_rate': 0.0,
                'message': 'No valid profit data'
            }
            
        # Check for convergence (low variance in recent episodes)
        profit_std = np.std(window_profits)
        profit_mean = np.mean(window_profits)
        
        # Convergence criterion: standard deviation below tolerance
        converged = profit_std < self.params.conv_tolerance
        
        # Convergence rate: fraction of episodes that would be considered converged
        # Check convergence in sliding windows of size conv_window
        convergence_checks = []
        min_window_size = max(100, self.params.conv_window // 10)  # Smaller windows for rate calculation
        
        if len(episode_stats) >= min_window_size:
            # Check multiple windows to compute convergence rate
            for i in range(min_window_size, len(episode_stats), min_window_size):
                window_end = i
                window_start = max(0, i - min_window_size)
                
                window_profits_check = []
                for stats in episode_stats[window_start:window_end]:
                    profits = stats['profits']
                    if isinstance(profits, list) and len(profits) > 0:
                        avg_profit = sum(profits) / len(profits)
                        window_profits_check.append(avg_profit)
                    elif isinstance(profits, (int, float)):
                        window_profits_check.append(profits)
                
                if window_profits_check:
                    window_std = np.std(window_profits_check)
                    window_converged = window_std < self.params.conv_tolerance
                    convergence_checks.append(1.0 if window_converged else 0.0)
        
        # Convergence rate: proportion of windows that converged
        convergence_rate = np.mean(convergence_checks) if convergence_checks else 0.0
        
        return {
            'converged': converged,
            'convergence_rate': float(convergence_rate),
            'final_profit_mean': float(profit_mean),
            'final_profit_std': float(profit_std),
            'convergence_window': self.params.conv_window,
            'convergence_tolerance': self.params.conv_tolerance,
            'num_convergence_checks': len(convergence_checks)
        }


def run_qlearning_experiment(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Run a complete Q-learning experiment with configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Experiment results dictionary
    """
    # Load configuration
    config = load_config(config_path)
    params = SimParams(config)
    
    # Create and run simulation
    sim = QLearningSimulation(params)
    results = sim.run_simulation(params.max_episodes)
    
    # Add benchmarks for comparison
    try:
        from cooperative_benchmark import (
            compute_cooperative_benchmark,
            compute_nash_benchmark,
            compute_monopoly_benchmark
        )
        
        coop = compute_cooperative_benchmark(params)
        nash = compute_nash_benchmark(params)
        monopoly = compute_monopoly_benchmark(params)
        
        results['benchmarks'] = {
            'cooperative': coop,
            'nash': nash,
            'monopoly': monopoly
        }
    except ImportError:
        print("Warning: Could not import benchmark functions")
        results['benchmarks'] = None
    
    return results


if __name__ == "__main__":
    # Run experiment with default config
    results = run_qlearning_experiment()
    
    # Print summary
    conv_analysis = results['convergence_analysis']
    print(f"\nSimulation completed!")
    print(f"Converged: {conv_analysis['converged']}")
    print(f"Final profit mean: {conv_analysis['final_profit_mean']:.4f}")
    print(f"Final profit std: {conv_analysis['final_profit_std']:.4f}")
    
    if results['benchmarks']:
        coop = results['benchmarks']['cooperative']
        print(f"Cooperative benchmark: {coop['joint_profit']:.4f}") 