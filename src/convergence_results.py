import numpy as np
from typing import List, Tuple, Dict
from globals import GlobalVars
from QL_routines import compute_state_number, compute_action_number

def compute_conv_results(i_experiment: int, globals: GlobalVars) -> None:
    """
    Computes statistics for one model
    """
    print('Computing convergence results (average profits and frequency of prices)')
    
    # Initialize variables
    profits = np.zeros((globals.num_sessions, globals.num_agents))
    freq_states = np.zeros((globals.num_sessions, globals.num_states))
    
    # Read strategies and states at convergence from file
    with open(globals.file_name_info_experiment, 'r') as f:
        for i_session in range(globals.num_sessions):
            if i_session % 100 == 0:
                print(f'Read {i_session} strategies')
            
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
    
    # Write to InfoExperiment file
    with open(globals.file_name_info_experiment, 'w') as f:
        # Loop over sessions
        for i_session in range(globals.num_sessions):
            print(f'iSession = {i_session + 1}')
            
            optimal_strategy_vec = globals.index_strategies[:, i_session]
            last_state_vec = globals.index_last_state[:, i_session]
            
            # Reshape strategy and state vectors
            optimal_strategy = optimal_strategy_vec.reshape(
                (globals.num_states, globals.num_agents)
            )
            
            if globals.depth_state0 == 0:
                last_observed_prices = optimal_strategy
            else:
                last_observed_prices = last_state_vec.reshape(
                    (globals.depth_state, globals.num_agents)
                )
            
            # Convergence analysis
            visited_states = np.zeros(globals.num_periods, dtype=np.int32)
            visited_profits = np.zeros((globals.num_periods, globals.num_agents))
            p_hist = np.zeros((globals.num_periods, globals.num_agents), dtype=np.int32)
            
            p = last_observed_prices.copy()
            p_prime = optimal_strategy[compute_state_number(p, globals), :]
            
            # Simulate price dynamics
            for i_period in range(globals.num_periods):
                if globals.depth_state > 1:
                    p[1:, :] = p[:-1, :]
                p[0, :] = p_prime
                p_hist[i_period, :] = p_prime
                
                visited_states[i_period] = compute_state_number(p, globals)
                
                for i_agent in range(globals.num_agents):
                    visited_profits[i_period, i_agent] = globals.pi[
                        compute_action_number(p_prime, globals),
                        i_agent
                    ]
                
                # Check if the state has already been visited
                if i_period >= 1 and np.any(
                    visited_states[:i_period] == visited_states[i_period]
                ):
                    break
                
                # Update pPrime and iterate
                p_prime = optimal_strategy[visited_states[i_period], :]
            
            # Find cycle length
            cycle_start = np.argmin(
                (visited_states[:i_period] - visited_states[i_period]) ** 2
            )
            cycle_length = i_period - cycle_start
            
            # Compute profits and state frequencies
            profits[i_session, :] = np.mean(
                visited_profits[cycle_start:i_period + 1, :],
                axis=0
            )
            
            freq_states[i_session, visited_states[cycle_start:i_period + 1]] = (
                1.0 / cycle_length
            )
            
            # Prepare cycle data for output
            p_hist[:cycle_length, :] = p_hist[cycle_start:i_period + 1, :]
            p_hist[cycle_length:, :] = 0
            
            visited_states[:cycle_length] = visited_states[cycle_start:i_period + 1]
            visited_states[cycle_length:] = 0
            
            visited_profits[:cycle_length, :] = visited_profits[cycle_start:i_period + 1, :]
            visited_profits[cycle_length:, :] = 0.0
            
            # Write session info to file
            f.write(f' {i_session + 1:8d}\n')
            f.write(f' {globals.converged[i_session]:1d}\n')
            f.write(f' {globals.time_to_convergence[i_session]:9.2f}\n')
            f.write(f' {cycle_length:8d}\n')
            
            # Write visited states
            f.write(' ' + ' '.join(
                f'{x:{globals.length_format_states_print}d}'
                for x in visited_states[:cycle_length]
            ) + '\n')
            
            # Write price history
            for i_agent in range(globals.num_agents):
                f.write(' ' + ' '.join(
                    f'{x:{globals.length_format_action_print}d}'
                    for x in p_hist[:cycle_length, i_agent]
                ) + '\n')
            
            # Write visited profits
            for i_agent in range(globals.num_agents):
                f.write(' ' + ' '.join(
                    f'{x:8.5f}'
                    for x in visited_profits[:cycle_length, i_agent]
                ) + '\n')
            
            # Write optimal strategy
            for i_state in range(globals.num_states):
                f.write(' ' + ' '.join(
                    f'{x:{globals.length_format_action_print}d}'
                    for x in optimal_strategy[i_state, :]
                ) + '\n')
    
    # Compute averages and descriptive statistics
    # Profits
    mean_profit = np.mean(profits, axis=0)
    se_profit = np.sqrt(np.abs(
        np.mean(profits ** 2, axis=0) - mean_profit ** 2
    ))
    
    avg_profits = np.mean(profits, axis=1)
    mean_avg_profit = np.mean(avg_profits)
    se_avg_profit = np.sqrt(np.abs(
        np.mean(avg_profits ** 2) - mean_avg_profit ** 2
    ))
    
    mean_profit_gain = (mean_profit - globals.nash_profits) / (
        globals.coop_profits - globals.nash_profits
    )
    se_profit_gain = se_profit / (globals.coop_profits - globals.nash_profits)
    
    mean_nash_profit = np.mean(globals.nash_profits)
    mean_coop_profit = np.mean(globals.coop_profits)
    
    mean_avg_profit_gain = (mean_avg_profit - mean_nash_profit) / (
        mean_coop_profit - mean_nash_profit
    )
    se_avg_profit_gain = se_avg_profit / (mean_coop_profit - mean_nash_profit)
    
    # States
    mean_freq_states = np.mean(freq_states, axis=0)
    
    # Write results to file
    with open('A_convResults.txt', 'a') as f:
        # Write header for first experiment
        if i_experiment == 0:
            header = 'Experiment '
            header += ' '.join(f'    alpha{i+1} ' for i in range(globals.num_agents))
            header += ' '.join(f'     beta{i+1} ' for i in range(globals.num_exploration_parameters))
            header += '     delta '
            header += ' '.join(
                f'typeQini{i+1} ' + ' '.join(f'par{j+1}Qini{i+1} ' for j in range(globals.num_agents))
                for i in range(globals.num_agents)
            )
            header += ' '.join(f'  DemPar{i+1:02d} ' for i in range(globals.num_demand_parameters))
            header += ' '.join(f'NashPrice{i+1} ' for i in range(globals.num_agents))
            header += ' '.join(f'CoopPrice{i+1} ' for i in range(globals.num_agents))
            header += ' '.join(f'NashProft{i+1} ' for i in range(globals.num_agents))
            header += ' '.join(f'CoopProft{i+1} ' for i in range(globals.num_agents))
            header += ' '.join(f'NashMktSh{i+1} ' for i in range(globals.num_agents))
            header += ' '.join(f'CoopMktSh{i+1} ' for i in range(globals.num_agents))
            header += ' '.join(
                f'Ag{i+1}Price{j+1:02d} ' for i in range(globals.num_agents)
                for j in range(globals.num_prices)
            )
            header += ' '.join(
                f'  avgProf{i+1}    seProf{i+1} ' for i in range(globals.num_agents)
            )
            header += '   avgProf     seProf '
            header += ' '.join(
                f'avgPrGain{i+1}  sePrGain{i+1} ' for i in range(globals.num_agents)
            )
            header += ' avgPrGain   sePrGain '
            header += ' '.join(
                f'{globals.label_states[i]:{max(10, 3+globals.length_format_states_print)}s} '
                for i in range(globals.num_states)
            )
            f.write(header + '\n')
        
        # Write experiment results
        result = f'{globals.cod_experiment:10d} '
        result += ' '.join(f'{globals.alpha[i]:10.5f}' for i in range(globals.num_agents))
        result += ' '.join(f'{globals.exploration_parameters[i]:10.5f}' for i in range(globals.num_exploration_parameters))
        result += f' {globals.delta:10.5f} '
        result += ' '.join(
            f'{globals.type_q_initialization[i]:9s} ' +
            ' '.join(f'{globals.par_q_initialization[i,j]:9.2f}' for j in range(globals.num_agents))
            for i in range(globals.num_agents)
        )
        result += ' '.join(f'{globals.demand_parameters[i]:10.5f}' for i in range(globals.num_demand_parameters))
        result += ' '.join(f'{globals.nash_prices[i]:10.5f}' for i in range(globals.num_agents))
        result += ' '.join(f'{globals.coop_prices[i]:10.5f}' for i in range(globals.num_agents))
        result += ' '.join(f'{globals.nash_profits[i]:10.5f}' for i in range(globals.num_agents))
        result += ' '.join(f'{globals.coop_profits[i]:10.5f}' for i in range(globals.num_agents))
        result += ' '.join(f'{globals.nash_market_shares[i]:10.5f}' for i in range(globals.num_agents))
        result += ' '.join(f'{globals.coop_market_shares[i]:10.5f}' for i in range(globals.num_agents))
        result += ' '.join(
            f'{globals.prices_grids[j,i]:10.5f}' for i in range(globals.num_agents)
            for j in range(globals.num_prices)
        )
        result += ' '.join(
            f'{mean_profit[i]:10.5f} {se_profit[i]:10.5f}'
            for i in range(globals.num_agents)
        )
        result += f' {mean_avg_profit:10.5f} {se_avg_profit:10.5f} '
        result += ' '.join(
            f'{mean_profit_gain[i]:10.5f} {se_profit_gain[i]:10.5f}'
            for i in range(globals.num_agents)
        )
        result += f' {mean_avg_profit_gain:10.5f} {se_avg_profit_gain:10.5f} '
        result += ' '.join(
            f'{mean_freq_states[i]:{max(10, 3+globals.length_format_states_print)}.6f}'
            for i in range(globals.num_states)
        )
        f.write(result + '\n') 