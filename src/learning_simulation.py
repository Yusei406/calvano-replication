import numpy as np
from typing import List, Tuple, Dict, Any
from .params import SimParams, SimResults
from .QL_routines import (
    init_state,
    generate_u_ini_price,
    generate_u_exploration,
    compute_state_number,
    compute_action_number,
    max_loc_break_ties,
    convert_number_base
)
from .init.QInit import init_Q, validate_Q_matrix
from .rng.Lecuyer import get_global_raw_rng, get_global_rng
from .worker import get_worker_rng, get_worker_params
from .convergence import has_converged, analyze_convergence
from .dtype_policy import DTYPE, zeros, array
import multiprocessing as mp
from functools import partial

def compute_p_prime(
    current_state: int,
    agent_index: int,
    q_matrix: np.ndarray,
    exploration_parameter: float,
    params: SimParams
) -> int:
    """
    Compute the next price (action) for an agent using epsilon-greedy or Boltzmann exploration.
    
    Args:
        current_state: Current state index
        agent_index: Index of the agent
        q_matrix: Q-value matrix for the agent
        exploration_parameter: Exploration parameter
        params: Simulation parameters
        
    Returns:
        Next action (price index)
    """
    rng = get_worker_rng()
    
    if params.exploration_type == 1:  # Greedy exploration
        if rng.ran2() < exploration_parameter:
            # Explore: choose random action
            return int(rng.ran2() * params.n_actions)
        else:
            # Exploit: choose best action
            max_val, max_loc = max_loc_break_ties(q_matrix[current_state, :], params)
            return max_loc
    else:  # Boltzmann exploration
        # Compute Boltzmann probabilities
        q_values = q_matrix[current_state, :]
        exp_values = np.exp(q_values / exploration_parameter)
        probabilities = exp_values / np.sum(exp_values)
        
        # Sample action according to probabilities
        cumulative_prob = 0.0
        random_value = rng.ran2()
        for action in range(params.n_actions):
            cumulative_prob += probabilities[action]
            if random_value <= cumulative_prob:
                return action
        return params.n_actions - 1  # Fallback

def compute_experiment(
    i_experiment: int,
    cod_experiment: int,
    alpha: float,
    exploration_parameters: np.ndarray,
    delta: float
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Compute a single Q-learning experiment.
    
    Args:
        i_experiment: Experiment index
        cod_experiment: Experiment code
        alpha: Learning rate
        exploration_parameters: Array of exploration parameters
        delta: Discount factor
        
    Returns:
        Tuple containing:
        - Q matrices for each agent
        - Strategies for each agent
        - States at convergence
        - Prices at convergence
    """
    # Get worker's parameters and RNG
    params = get_worker_params()
    rng = get_worker_rng()
    
    # Initialize Q matrices using new system
    q_matrices = []
    for agent in range(params.n_agents):
        q_matrix = init_Q(
            strategy=params.q_strategy,
            params=params,
            rng=rng,
            agent_idx=agent,
            fixed_price=params.fixed_price,
            q_init_path=params.q_init_path,
            coop_price=params.coop_price,
            punish_price=params.punish_price,
            coop_value=params.coop_value,
            punish_value=params.punish_value,
            random_scale=params.random_scale,
            min_val=params.min_val,
            max_val=params.max_val,
            constant_value=params.constant_value
        )
        validate_Q_matrix(q_matrix, params)
        q_matrices.append(q_matrix)
    
    # Initialize strategies
    strategies = [zeros((params.n_states,), dtype=np.int32) for _ in range(params.n_agents)]
    
    # Initialize state
    current_state = init_state(params)
    
    # Track price history for convergence
    price_history = []
    strategy_history = []
    
    # Main learning loop
    for iteration in range(params.max_iterations):
        # Generate actions for each agent
        actions = []
        for agent in range(params.n_agents):
            action = compute_p_prime(
                current_state,
                agent,
                q_matrices[agent],
                exploration_parameters[agent],
                params
            )
            actions.append(action)
        
        # Compute next state
        next_state = compute_state_number(actions, params)
        
        # Compute rewards
        rewards = compute_rewards(actions, params)
        
        # Update Q values
        for agent in range(params.n_agents):
            old_q = q_matrices[agent][current_state, actions[agent]]
            max_next_q = np.max(q_matrices[agent][next_state, :])
            new_q = (1 - alpha) * old_q + alpha * (rewards[agent] + delta * max_next_q)
            q_matrices[agent][current_state, actions[agent]] = new_q
        
        # Update state
        current_state = next_state
        
        # Update strategies and check convergence periodically
        if iteration % params.n_perfect_measurement == 0:
            # Update strategies
            for agent in range(params.n_agents):
                for state in range(params.n_states):
                    max_val, max_loc = max_loc_break_ties(q_matrices[agent][state, :], params)
                    strategies[agent][state] = max_loc
            
            # Record prices and strategies for convergence check
            current_prices = array([actions[agent] / (params.n_actions - 1) for agent in range(params.n_agents)])
            price_history.append(current_prices)
            strategy_history.append([s.copy() for s in strategies])
            
            # Check convergence
            if len(price_history) >= params.convergence_window // params.n_perfect_measurement:
                price_hist_array = array(price_history)
                if has_converged(price_hist_array, params.convergence_window // params.n_perfect_measurement, params.convergence_tolerance):
                    break
    
    # Get final states and prices
    states = get_states_at_convergence(strategies, params)
    prices = get_prices_at_convergence(states, params)
    
    return q_matrices, strategies, states, prices

def run_simulation(job_args: Dict[str, Any]) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Run a single simulation with given parameters.
    Worker function for multiprocessing.
    
    Args:
        job_args: Job-specific arguments
        
    Returns:
        Tuple containing:
        - Q matrices for each agent
        - Strategies for each agent
        - States at convergence
        - Prices at convergence
    """
    # Extract job arguments
    i_experiment = job_args['i_experiment']
    cod_experiment = job_args['cod_experiment']
    alpha = job_args['alpha']
    exploration_parameters = job_args['exploration_parameters']
    delta = job_args['delta']
    
    # Run experiment (params are available via get_worker_params())
    q_matrices, strategies, states, prices = compute_experiment(
        i_experiment,
        cod_experiment,
        alpha,
        exploration_parameters,
        delta
    )
    
    return q_matrices, strategies, states, prices

def run_parallel_simulations(
    params: SimParams,
    job_list: List[Dict[str, Any]]
) -> List[Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]]:
    """
    Run multiple simulations in parallel.
    
    Args:
        params: Simulation parameters
        job_list: List of job arguments for each simulation
        
    Returns:
        List of results from each simulation
    """
    from .worker import init_worker
    
    # Create process pool with worker initialization
    with mp.Pool(
        processes=params.n_cores,
        initializer=init_worker,
        initargs=(0, params.rng_seed, params)
    ) as pool:
        results = pool.map(run_simulation, job_list)
    
    return results

def compute_rewards(actions: List[int], params: SimParams) -> List[float]:
    """Compute rewards for each agent based on actions."""
    # Convert actions to prices
    prices = [action / (params.n_actions - 1) for action in actions]
    
    # Compute demands using logit model
    utilities = [params.a0 - params.a * p + params.c for p in prices]
    exp_utilities = np.exp(np.array(utilities) / params.mu)
    total_exp = np.sum(exp_utilities)
    demands = exp_utilities / total_exp
    
    # Compute profits
    profits = [p * d for p, d in zip(prices, demands)]
    
    return profits

def get_states_at_convergence(strategies: List[np.ndarray], params: SimParams) -> np.ndarray:
    """Get states at convergence from strategies."""
    states = np.zeros(params.n_states, dtype=int)
    
    for state in range(params.n_states):
        actions = [strategy[state] for strategy in strategies]
        states[state] = compute_state_number(actions, params)
    
    return states

def get_prices_at_convergence(states: np.ndarray, params: SimParams) -> np.ndarray:
    """Get prices at convergence from states."""
    n_states = len(states)
    prices = np.zeros((n_states, params.n_agents))
    
    for state in range(n_states):
        actions = convert_number_base(state, params.n_actions, params.n_agents)
        prices[state] = [action / (params.n_actions - 1) for action in actions]
    
    return prices

def write_experiment_results(
    results: SimResults,
    params: SimParams,
    output_file: str
) -> None:
    """Write experiment results to file."""
    with open(output_file, 'w') as f:
        # Write header
        f.write(f"Experiment {params.experiment_number}\n")
        f.write(f"Sessions: {params.n_sessions}\n")
        f.write(f"Agents: {params.n_agents}\n")
        f.write(f"States: {params.n_states}\n")
        f.write(f"Actions: {params.n_actions}\n")
        f.write("\n")
        
        # Write convergence results
        f.write("Convergence Results:\n")
        convergence_rate = np.mean(results.converged) if results.converged is not None else 0.0
        f.write(f"Convergence Rate: {convergence_rate:.4f}\n")
        
        if results.time_to_convergence is not None:
            avg_time = np.mean(results.time_to_convergence[results.converged])
            f.write(f"Average Time to Convergence: {avg_time:.2f}\n")
        
        f.write("\n")
        
        # Write profit results
        if results.pi is not None:
            f.write("Profit Results:\n")
            avg_profits = np.mean(results.pi, axis=0)
            for agent in range(params.n_agents):
                f.write(f"Agent {agent}: {avg_profits[agent]:.4f}\n") 