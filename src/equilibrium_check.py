import numpy as np
from typing import List, Tuple, Dict
from globals import GlobalVars
from QL_routines import compute_q_cell

def compute_eq_check(i_experiment: int, globals: GlobalVars) -> None:
    """
    Computes statistics for one model
    
    Parameters:
    - i_experiment: model index
    """
    print('Computing equilibrium checks')
    
    # Initialize variables
    thres_cycle_length = np.arange(1, globals.num_thres_cycle_length + 1)
    
    # Read strategies and states at convergence from file
    with open(globals.file_name_info_experiment, 'r') as f:
        for i_session in range(globals.num_sessions):
            print(f'iSession = {i_session + 1}')
            
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
    
    # Initialize arrays for results
    freq_br_all = np.zeros((globals.num_agents, globals.num_sessions))
    freq_br_on_path = np.zeros((globals.num_agents, globals.num_sessions))
    freq_br_off_path = np.zeros((globals.num_agents, globals.num_sessions))
    freq_eq_all = np.zeros(globals.num_sessions)
    freq_eq_on_path = np.zeros(globals.num_sessions)
    freq_eq_off_path = np.zeros(globals.num_sessions)
    flag_br_all = np.zeros((globals.num_agents, globals.num_sessions), dtype=np.int32)
    flag_br_on_path = np.zeros((globals.num_agents, globals.num_sessions), dtype=np.int32)
    flag_br_off_path = np.zeros((globals.num_agents, globals.num_sessions), dtype=np.int32)
    flag_eq_all = np.zeros(globals.num_sessions, dtype=np.int32)
    flag_eq_on_path = np.zeros(globals.num_sessions, dtype=np.int32)
    flag_eq_off_path = np.zeros(globals.num_sessions, dtype=np.int32)
    
    # Loop over sessions
    for i_session in range(globals.num_sessions):
        optimal_strategy_vec = globals.index_strategies[:, i_session]
        optimal_strategy = optimal_strategy_vec.reshape(
            (globals.num_states, globals.num_agents)
        )
        cycle_length_session = globals.cycle_length[i_session]
        cycle_states_session = globals.cycle_states[:cycle_length_session, i_session]
        
        # Compute equilibrium check for this session
        (
            freq_br_all[:, i_session],
            freq_br_on_path[:, i_session],
            freq_br_off_path[:, i_session],
            freq_eq_all[i_session],
            freq_eq_on_path[i_session],
            freq_eq_off_path[i_session],
            flag_br_all[:, i_session],
            flag_br_on_path[:, i_session],
            flag_br_off_path[:, i_session],
            flag_eq_all[i_session],
            flag_eq_on_path[i_session],
            flag_eq_off_path[i_session]
        ) = compute_eq_check_session(
            optimal_strategy,
            cycle_length_session,
            cycle_states_session,
            globals
        )
    
    # Compute averages and descriptive statistics
    avg_freq_br_all = np.zeros((globals.num_agents + 1, globals.num_thres_cycle_length + 1))
    avg_freq_br_on_path = np.zeros((globals.num_agents + 1, globals.num_thres_cycle_length + 1))
    avg_freq_br_off_path = np.zeros((globals.num_agents + 1, globals.num_thres_cycle_length + 1))
    avg_freq_eq_all = np.zeros(globals.num_thres_cycle_length + 1)
    avg_freq_eq_on_path = np.zeros(globals.num_thres_cycle_length + 1)
    avg_freq_eq_off_path = np.zeros(globals.num_thres_cycle_length + 1)
    avg_flag_br_all = np.zeros((globals.num_agents + 1, globals.num_thres_cycle_length + 1))
    avg_flag_br_on_path = np.zeros((globals.num_agents + 1, globals.num_thres_cycle_length + 1))
    avg_flag_br_off_path = np.zeros((globals.num_agents + 1, globals.num_thres_cycle_length + 1))
    avg_flag_eq_all = np.zeros(globals.num_thres_cycle_length + 1)
    avg_flag_eq_on_path = np.zeros(globals.num_thres_cycle_length + 1)
    avg_flag_eq_off_path = np.zeros(globals.num_thres_cycle_length + 1)
    
    # Total averages
    num_cycle_length = np.zeros(globals.num_thres_cycle_length + 1, dtype=np.int32)
    num_cycle_length[0] = globals.num_sessions
    
    r_num = float(globals.num_agents * num_cycle_length[0])
    avg_freq_br_all[0, 0] = np.sum(freq_br_all) / r_num
    avg_freq_br_on_path[0, 0] = np.sum(freq_br_on_path) / r_num
    avg_freq_br_off_path[0, 0] = np.sum(freq_br_off_path) / r_num
    avg_flag_br_all[0, 0] = np.sum(flag_br_all) / r_num
    avg_flag_br_on_path[0, 0] = np.sum(flag_br_on_path) / r_num
    avg_flag_br_off_path[0, 0] = np.sum(flag_br_off_path) / r_num
    
    r_num = float(num_cycle_length[0])
    avg_freq_eq_all[0] = np.sum(freq_eq_all) / r_num
    avg_freq_eq_on_path[0] = np.sum(freq_eq_on_path) / r_num
    avg_freq_eq_off_path[0] = np.sum(freq_eq_off_path) / r_num
    avg_flag_eq_all[0] = np.sum(flag_eq_all) / r_num
    avg_flag_eq_on_path[0] = np.sum(flag_eq_on_path) / r_num
    avg_flag_eq_off_path[0] = np.sum(flag_eq_off_path) / r_num
    
    for i_agent in range(globals.num_agents):
        r_num = float(num_cycle_length[0])
        avg_freq_br_all[i_agent + 1, 0] = np.sum(freq_br_all[i_agent, :]) / r_num
        avg_freq_br_on_path[i_agent + 1, 0] = np.sum(freq_br_on_path[i_agent, :]) / r_num
        avg_freq_br_off_path[i_agent + 1, 0] = np.sum(freq_br_off_path[i_agent, :]) / r_num
        avg_flag_br_all[i_agent + 1, 0] = np.sum(flag_br_all[i_agent, :]) / r_num
        avg_flag_br_on_path[i_agent + 1, 0] = np.sum(flag_br_on_path[i_agent, :]) / r_num
        avg_flag_br_off_path[i_agent + 1, 0] = np.sum(flag_br_off_path[i_agent, :]) / r_num
    
    # Averages by cycle length
    for i_thres in range(globals.num_thres_cycle_length):
        if i_thres < globals.num_thres_cycle_length - 1:
            cond = globals.cycle_length == thres_cycle_length[i_thres]
        else:
            cond = globals.cycle_length >= thres_cycle_length[i_thres]
        
        num_cycle_length[i_thres + 1] = np.sum(cond)
        
        if num_cycle_length[i_thres + 1] > 0:
            r_num = float(globals.num_agents * num_cycle_length[i_thres + 1])
            matcond = np.tile(cond, (globals.num_agents, 1))
            
            avg_freq_br_all[0, i_thres + 1] = np.sum(freq_br_all[matcond]) / r_num
            avg_freq_br_on_path[0, i_thres + 1] = np.sum(freq_br_on_path[matcond]) / r_num
            avg_freq_br_off_path[0, i_thres + 1] = np.sum(freq_br_off_path[matcond]) / r_num
            avg_flag_br_all[0, i_thres + 1] = np.sum(flag_br_all[matcond]) / r_num
            avg_flag_br_on_path[0, i_thres + 1] = np.sum(flag_br_on_path[matcond]) / r_num
            avg_flag_br_off_path[0, i_thres + 1] = np.sum(flag_br_off_path[matcond]) / r_num
            
            r_num = float(num_cycle_length[i_thres + 1])
            avg_freq_eq_all[i_thres + 1] = np.sum(freq_eq_all[cond]) / r_num
            avg_freq_eq_on_path[i_thres + 1] = np.sum(freq_eq_on_path[cond]) / r_num
            avg_freq_eq_off_path[i_thres + 1] = np.sum(freq_eq_off_path[cond]) / r_num
            avg_flag_eq_all[i_thres + 1] = np.sum(flag_eq_all[cond]) / r_num
            avg_flag_eq_on_path[i_thres + 1] = np.sum(flag_eq_on_path[cond]) / r_num
            avg_flag_eq_off_path[i_thres + 1] = np.sum(flag_eq_off_path[cond]) / r_num
            
            for i_agent in range(globals.num_agents):
                r_num = float(num_cycle_length[i_thres + 1])
                avg_freq_br_all[i_agent + 1, i_thres + 1] = np.sum(freq_br_all[i_agent, cond]) / r_num
                avg_freq_br_on_path[i_agent + 1, i_thres + 1] = np.sum(freq_br_on_path[i_agent, cond]) / r_num
                avg_freq_br_off_path[i_agent + 1, i_thres + 1] = np.sum(freq_br_off_path[i_agent, cond]) / r_num
                avg_flag_br_all[i_agent + 1, i_thres + 1] = np.sum(flag_br_all[i_agent, cond]) / r_num
                avg_flag_br_on_path[i_agent + 1, i_thres + 1] = np.sum(flag_br_on_path[i_agent, cond]) / r_num
                avg_flag_br_off_path[i_agent + 1, i_thres + 1] = np.sum(flag_br_off_path[i_agent, cond]) / r_num
    
    # Write results to file
    with open('A_equilibriumCheck.txt', 'a') as f:
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
            header += ' '.join(f'num_Len{i+1:02d} ' for i in range(globals.num_thres_cycle_length + 1))
            header += ' '.join(
                f'FlagEQAll_Len{i+1:02d} FlagEQOnPath_Len{i+1:02d} FlagEQOffPath_Len{i+1:02d} '
                f'FreqEQAll_Len{i+1:02d} FreqEQOnPath_Len{i+1:02d} FreqEQOffPath_Len{i+1:02d} '
                f'FlagBRAll_Ag0_Len{i+1:02d} FlagBROnPath_Ag0_Len{i+1:02d} FlagBROffPath_Ag0_Len{i+1:02d} '
                f'FreqBRAll_Ag0_Len{i+1:02d} FreqBROnPath_Ag0_Len{i+1:02d} FreqBROffPath_Ag0_Len{i+1:02d}'
                for i in range(globals.num_thres_cycle_length + 1)
            )
            header += ' '.join(
                ' '.join(
                    f'FlagBRAll_Ag{i+1}_Len{j+1:02d} FlagBROnPath_Ag{i+1}_Len{j+1:02d} '
                    f'FlagBROffPath_Ag{i+1}_Len{j+1:02d} FreqBRAll_Ag{i+1}_Len{j+1:02d} '
                    f'FreqBROnPath_Ag{i+1}_Len{j+1:02d} FreqBROffPath_Ag{i+1}_Len{j+1:02d}'
                    for j in range(globals.num_thres_cycle_length + 1)
                )
                for i in range(globals.num_agents)
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
        result += ' '.join(f'{num_cycle_length[i]:9d}' for i in range(globals.num_thres_cycle_length + 1))
        result += ' '.join(
            f'{avg_flag_eq_all[i]:15.7f} {avg_flag_eq_on_path[i]:18.7f} {avg_flag_eq_off_path[i]:19.7f} '
            f'{avg_freq_eq_all[i]:19.7f} {avg_freq_eq_on_path[i]:22.7f} {avg_freq_eq_off_path[i]:23.7f} '
            f'{avg_flag_br_all[0,i]:19.7f} {avg_flag_br_on_path[0,i]:22.7f} {avg_flag_br_off_path[0,i]:23.7f} '
            f'{avg_freq_br_all[0,i]:19.7f} {avg_freq_br_on_path[0,i]:22.7f} {avg_freq_br_off_path[0,i]:23.7f}'
            for i in range(globals.num_thres_cycle_length + 1)
        )
        result += ' '.join(
            ' '.join(
                f'{avg_flag_br_all[i+1,j]:19.7f} {avg_flag_br_on_path[i+1,j]:22.7f} '
                f'{avg_flag_br_off_path[i+1,j]:23.7f} {avg_freq_br_all[i+1,j]:19.7f} '
                f'{avg_freq_br_on_path[i+1,j]:22.7f} {avg_freq_br_off_path[i+1,j]:23.7f}'
                for j in range(globals.num_thres_cycle_length + 1)
            )
            for i in range(globals.num_agents)
        )
        f.write(result + '\n')

def compute_eq_check_session(
    optimal_strategy: np.ndarray,
    cycle_length_session: int,
    cycle_states_session: np.ndarray,
    globals: GlobalVars
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    float, float, float,
    np.ndarray, np.ndarray, np.ndarray,
    int, int, int
]:
    """
    Computes equilibrium check for an individual replication
    
    Parameters:
    - optimal_strategy: strategy for all agents
    - cycle_length_session: length of the replication's path (i.e., state cycle)
    - cycle_states_session: replication's path (i.e., state cycle)
    
    Returns:
    - freq_br_all: % of all states in which at least one agent is best responding
    - freq_br_on_path: % of on path states in which at least one agent is best responding
    - freq_br_off_path: % of off path states in which at least one agent is best responding
    - freq_eq_all: % of all states in which at all agents are best responding
    - freq_eq_on_path: % of on path states in which at all agents are best responding
    - freq_eq_off_path: % of off path states in which at all agents are best responding
    - flag_br_all: = 1: in all states at least one agent is best responding
    - flag_br_on_path: = 1: in all on path states at least one agent is best responding
    - flag_br_off_path: = 1: in all off path states at least one agent is best responding
    - flag_eq_all: = 1: in all states both agents are best responding
    - flag_eq_on_path: = 1: in all on path states both agents are best responding
    - flag_eq_off_path: = 1: in all off path states both agents are best responding
    """
    # 1. For each agent A and each state S, check whether A is best responding in state S
    is_best_reply = np.zeros((globals.num_states, globals.num_agents), dtype=np.int32)
    
    for i_state in range(globals.num_states):
        # Compute state value function for OptimalStrategy in i_state, for all prices and agents
        for i_agent in range(globals.num_agents):
            state_value_function = np.zeros(globals.num_prices)
            
            for i_price in range(globals.num_prices):
                state_value_function[i_price], _, _ = compute_q_cell(
                    optimal_strategy, i_state, i_price, i_agent, globals.delta, globals
                )
            
            max_state_value_function = np.max(state_value_function)
            strategy_price = optimal_strategy[i_state, i_agent]
            improved_prices = np.where(
                np.abs(state_value_function - max_state_value_function) <= 0.0
            )[0]
            
            if strategy_price in improved_prices:
                is_best_reply[i_state, i_agent] = 1
    
    # 2. For each agent A, compute statistics
    num_states_br_all = np.zeros(globals.num_agents, dtype=np.int32)
    num_states_br_on_path = np.zeros(globals.num_agents, dtype=np.int32)
    num_states_br_off_path = np.zeros(globals.num_agents, dtype=np.int32)
    flag_br_all = np.zeros(globals.num_agents, dtype=np.int32)
    flag_br_on_path = np.zeros(globals.num_agents, dtype=np.int32)
    flag_br_off_path = np.zeros(globals.num_agents, dtype=np.int32)
    
    for i_agent in range(globals.num_agents):
        num_states_br_all[i_agent] = np.sum(is_best_reply[:, i_agent])
        num_states_br_on_path[i_agent] = np.sum(is_best_reply[cycle_states_session, i_agent])
        num_states_br_off_path[i_agent] = num_states_br_all[i_agent] - num_states_br_on_path[i_agent]
        
        if num_states_br_all[i_agent] == globals.num_states:
            flag_br_all[i_agent] = 1
        if num_states_br_on_path[i_agent] == cycle_length_session:
            flag_br_on_path[i_agent] = 1
        if num_states_br_off_path[i_agent] == (globals.num_states - cycle_length_session):
            flag_br_off_path[i_agent] = 1
    
    # 3. Simultaneously for all agents, compute statistics
    num_states_eq_all = 0
    num_states_eq_on_path = 0
    num_states_eq_off_path = 0
    
    for i_state in range(globals.num_states):
        if np.all(is_best_reply[i_state, :] == 1):
            num_states_eq_all += 1
            
            if np.any(cycle_states_session == i_state):
                num_states_eq_on_path += 1
            else:
                num_states_eq_off_path += 1
    
    flag_eq_all = 0
    flag_eq_on_path = 0
    flag_eq_off_path = 0
    
    if num_states_eq_all == globals.num_states:
        flag_eq_all = 1
    if num_states_eq_on_path == cycle_length_session:
        flag_eq_on_path = 1
    if num_states_eq_off_path == (globals.num_states - cycle_length_session):
        flag_eq_off_path = 1
    
    # 4. Convert number of states into frequencies
    freq_br_all = num_states_br_all.astype(float) / globals.num_states
    freq_br_on_path = num_states_br_on_path.astype(float) / cycle_length_session
    freq_br_off_path = num_states_br_off_path.astype(float) / (globals.num_states - cycle_length_session)
    freq_eq_all = float(num_states_eq_all) / globals.num_states
    freq_eq_on_path = float(num_states_eq_on_path) / cycle_length_session
    freq_eq_off_path = float(num_states_eq_off_path) / (globals.num_states - cycle_length_session)
    
    return (
        freq_br_all, freq_br_on_path, freq_br_off_path,
        freq_eq_all, freq_eq_on_path, freq_eq_off_path,
        flag_br_all, flag_br_on_path, flag_br_off_path,
        flag_eq_all, flag_eq_on_path, flag_eq_off_path
    ) 