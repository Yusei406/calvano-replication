import numpy as np
from typing import List, Tuple, Dict
from globals import GlobalVars
from QL_routines import compute_q_cell, init_q_matrices, compute_p_prime
from generic_routines import are_equal_reals, convert_number_base, compute_state_number, compute_action_number
from impulse_response import compute_static_best_response

def compute_learning_trajectory(
    i_experiment: int,
    cod_experiment: int,
    alpha: np.ndarray,
    exploration_parameters: np.ndarray,
    delta: float,
    globals: GlobalVars
) -> None:
    """
    Computes Monte Carlo Q-Learning Trajectories
    
    Parameters:
    - i_experiment: model index
    - cod_experiment: experiment code
    - alpha: learning rates for each agent
    - exploration_parameters: exploration parameters
    - delta: discount factor
    """
    print('Computing learning trajectories')
    
    # Initialize variables
    num_thres_path_cycle_length = 10
    thres_path_cycle_length = np.arange(1, num_thres_path_cycle_length + 1)
    
    # Initialize arrays for results
    pg_mat = np.zeros((globals.params_learning_trajectory[0], globals.num_sessions))
    ic_mat = np.zeros((globals.params_learning_trajectory[0], globals.num_sessions))
    ir_mat = np.zeros((globals.params_learning_trajectory[0], globals.num_sessions))
    
    # Read strategies and states at convergence from file
    with open(globals.file_name_info_experiment, 'r') as f:
        for i_session in range(globals.num_sessions):
            print(f'Session = {i_session + 1} started')
            
            r_session = int(f.readline().strip())
            globals.converged[r_session] = int(f.readline().strip())
            globals.time_to_convergence[r_session] = float(f.readline().strip())
            globals.index_last_state[:, r_session] = np.array(
                [int(x) for x in f.readline().strip().split()]
            )
            
            for i_state in range(globals.num_states):
                globals.index_strategies[
                    i_state::globals.num_states, r_session
                ] = np.array([int(x) for x in f.readline().strip().split()])
    
    # Loop over sessions
    for i_session in range(globals.num_sessions):
        cycle_length_session = globals.cycle_length[i_session]
        cycle_states_session = globals.cycle_states[:cycle_length_session, i_session]
        
        # Initialize Q matrices
        q = np.zeros((globals.num_states, globals.num_prices, globals.num_agents))
        max_val_q = np.zeros((globals.num_states, globals.num_agents))
        strategy_prime = np.zeros((globals.num_states, globals.num_agents), dtype=np.int32)
        
        init_q_matrices(
            i_session,
            q,
            max_val_q,
            strategy_prime,
            globals.pi,
            delta,
            globals
        )
        strategy = strategy_prime.copy()
        
        # Randomly initialize prices and state
        p = np.zeros((globals.depth_state, globals.num_agents), dtype=np.int32)
        state_prime = 0
        action_prime = 0
        
        # Initialize state
        u_ini_price = np.random.rand(globals.depth_state, globals.num_agents)
        for i_agent in range(globals.num_agents):
            for i_depth in range(globals.depth_state):
                p[i_depth, i_agent] = int(u_ini_price[i_depth, i_agent] * globals.num_prices)
        state_prime = compute_state_number(p, globals)
        action_prime = compute_action_number(p, globals)
        state = state_prime
        
        # Loop
        if globals.type_exploration_mechanism == 1:
            eps = np.ones(globals.num_agents)
        elif globals.type_exploration_mechanism == 2:
            eps = np.ones(globals.num_agents) * 1e3
        
        pg_sum = 0.0
        ic_sum = 0.0
        ir_sum = 0.0
        
        for i_iters in range(1, globals.params_learning_trajectory[0] * globals.params_learning_trajectory[1] + 1):
            # Generate exploration random numbers
            u_exploration = np.random.rand(2, globals.num_agents)
            
            # Compute p_prime by balancing exploration vs. exploitation
            p_prime = compute_p_prime(
                exploration_parameters,
                u_exploration,
                strategy_prime,
                state,
                i_iters,
                q,
                eps,
                globals
            )
            
            # Define the new state
            if globals.depth_state > 1:
                p[1:globals.depth_state, :] = p[:globals.depth_state-1, :]
            p[0, :] = p_prime
            state_prime = compute_state_number(p, globals)
            action_prime = compute_action_number(p, globals)
            
            # Store trajectories
            # Profit Gain
            profit_gain = globals.pg[action_prime, :]
            pg_sum += np.sum(profit_gain) / globals.num_agents
            
            if i_iters % globals.params_learning_trajectory[1] == 0:
                # Profit Gain
                pg_mat[i_iters//globals.params_learning_trajectory[1]-1, i_session] = (
                    pg_mat[i_iters//globals.params_learning_trajectory[1]-1, i_session] +
                    pg_sum / globals.params_learning_trajectory[1]
                )
                pg_sum = 0.0
                
                # Incentive Compatibility
                for i_cycle in range(cycle_length_session):
                    ss1 = cycle_states_session[i_cycle]
                    for i_agent in range(globals.num_agents):
                        q_row_values = np.zeros(globals.num_prices)
                        for i_price in range(globals.num_prices):
                            # Compute row of true Q matrix
                            q_row_values[i_price], _, _, _ = compute_q_cell(
                                strategy,
                                ss1,
                                i_price,
                                i_agent,
                                delta,
                                globals
                            )
                        
                        if are_equal_reals(
                            np.max(q_row_values),
                            q_row_values[strategy[ss1, i_agent]]
                        ):
                            ic_sum += 1.0 / (globals.num_agents * cycle_length_session)
                
                ic_mat[i_iters//globals.params_learning_trajectory[1]-1, i_session] = ic_sum
                ic_sum = 0.0
                
                # Punishment from nondeviating agent in period t+2
                for i_cycle in range(cycle_length_session):
                    for i_agent in range(globals.num_agents):
                        # Period t: Initial state
                        ss0 = cycle_states_session[i_cycle]
                        p_lt1 = convert_number_base(ss0-1, globals.num_prices, globals.length_states)
                        p_lt1 = p_lt1.reshape((globals.depth_state, globals.num_agents))
                        
                        # Period t+1: Shock to static best response
                        p_prime_lt1 = strategy[ss0, :].copy()
                        if globals.depth_state > 1:
                            p_lt1[1:globals.depth_state, :] = p_lt1[:globals.depth_state-1, :]
                        p_lt1[0, :] = p_prime_lt1
                        nu_pi_static_br = compute_static_best_response(
                            strategy,
                            ss0,
                            i_agent,
                            p_lt1[0, i_agent],
                            globals
                        )
                        ss1 = compute_state_number(p_lt1, globals)
                        
                        # Period t+2
                        p_prime_lt2 = strategy[ss1, :].copy()
                        avg_p_ratio = 0.0
                        for j_agent in range(globals.num_agents):
                            if j_agent == i_agent:
                                continue
                            avg_p_ratio += (
                                globals.prices_grids[p_prime_lt2[j_agent], j_agent] /
                                globals.prices_grids[p_prime_lt1[j_agent], j_agent]
                            )
                        avg_p_ratio /= (globals.num_agents - 1)
                        ir_sum += avg_p_ratio / (globals.num_agents * cycle_length_session)
                
                ir_mat[i_iters//globals.params_learning_trajectory[1]-1, i_session] = ir_sum
                ir_sum = 0.0
            
            # Each agent collects his payoff and updates
            for i_agent in range(globals.num_agents):
                # Q matrices and strategies update
                old_q = q[state, p_prime[i_agent], i_agent]
                new_q = old_q + alpha[i_agent] * (
                    globals.pi[action_prime, i_agent] +
                    delta * max_val_q[state_prime, i_agent] -
                    old_q
                )
                q[state, p_prime[i_agent], i_agent] = new_q
                
                if new_q > max_val_q[state, i_agent]:
                    max_val_q[state, i_agent] = new_q
                    if strategy_prime[state, i_agent] != p_prime[i_agent]:
                        strategy_prime[state, i_agent] = p_prime[i_agent]
                
                if (new_q < max_val_q[state, i_agent] and
                    strategy_prime[state, i_agent] == p_prime[i_agent]):
                    # Find maximum with tie breaking
                    max_val = np.max(q[state, :, i_agent])
                    max_indices = np.where(q[state, :, i_agent] == max_val)[0]
                    strategy_prime[state, i_agent] = np.random.choice(max_indices)
            
            # Update and iterate
            strategy[state, :] = strategy_prime[state, :]
            state = state_prime
    
    # Compute statistics on PG and IC trajectories
    pg_ss = compute_row_summary_statistics(
        globals.params_learning_trajectory[0],
        globals.num_sessions,
        pg_mat
    )
    ic_ss = compute_row_summary_statistics(
        globals.params_learning_trajectory[0],
        globals.num_sessions,
        ic_mat
    )
    ir_ss = compute_row_summary_statistics(
        globals.params_learning_trajectory[0],
        globals.num_sessions,
        ir_mat
    )
    
    # Write results to file
    l_trajectory_file_name = f'LTrajectories_{globals.experiment_number}'
    with open(l_trajectory_file_name, 'w') as f:
        # Write header
        header = '         iter '
        header += '          AvgPG            SdPG           MinPG         q0025PG          q025PG           q05PG          q075PG         q0975PG           MaxPG '
        header += '          AvgIC            SdIC           MinIC         q0025IC          q025IC           q05IC          q075IC         q0975IC           MaxIC '
        header += '          AvgIR            SdIR           MinIR         q0025IR          q025IR           q05IR          q075IR         q0975IR           MaxIR '
        f.write(header + '\n')
        
        # Write results
        for i in range(globals.params_learning_trajectory[0]):
            result = f'{(i+1)*globals.params_learning_trajectory[1]:12d} '
            result += ' '.join(f'{x:15.5f}' for x in pg_ss[i, :])
            result += ' '
            result += ' '.join(f'{x:15.5f}' for x in ic_ss[i, :])
            result += ' '
            result += ' '.join(f'{x:15.5f}' for x in ir_ss[i, :])
            f.write(result + '\n')

def compute_row_summary_statistics(
    num_rows: int,
    num_sessions: int,
    mat: np.ndarray
) -> np.ndarray:
    """
    Computes summary statistics for each row of a matrix
    
    Parameters:
    - num_rows: number of rows
    - num_sessions: number of sessions
    - mat: matrix of values
    
    Returns:
    - Array of summary statistics for each row
    """
    ss = np.zeros((num_rows, 9))
    
    for i in range(num_rows):
        # Mean
        ss[i, 0] = np.mean(mat[i, :])
        # Standard deviation
        ss[i, 1] = np.std(mat[i, :])
        # Minimum
        ss[i, 2] = np.min(mat[i, :])
        # 0.025 quantile
        ss[i, 3] = np.quantile(mat[i, :], 0.025)
        # 0.25 quantile
        ss[i, 4] = np.quantile(mat[i, :], 0.25)
        # 0.5 quantile (median)
        ss[i, 5] = np.quantile(mat[i, :], 0.5)
        # 0.75 quantile
        ss[i, 6] = np.quantile(mat[i, :], 0.75)
        # 0.975 quantile
        ss[i, 7] = np.quantile(mat[i, :], 0.975)
        # Maximum
        ss[i, 8] = np.max(mat[i, :])
    
    return ss 