import numpy as np
from typing import List, Tuple, Dict
from globals import GlobalVars
from QL_routines import compute_q_cell
from generic_routines import are_equal_reals

def compute_q_gap_to_max(i_experiment: int, globals: GlobalVars) -> None:
    """
    Computes statistics for one model
    
    Parameters:
    - i_experiment: model index
    """
    print('Computing Q gaps')
    
    # Initialize variables
    num_thres_path_cycle_length = 10
    thres_path_cycle_length = np.arange(1, num_thres_path_cycle_length + 1)
    
    # Initialize arrays for results
    sum_q_gap_tot = np.zeros((num_thres_path_cycle_length + 1, globals.num_agents + 1))
    sum_q_gap_on_path = np.zeros((num_thres_path_cycle_length + 1, globals.num_agents + 1))
    sum_q_gap_not_on_path = np.zeros((num_thres_path_cycle_length + 1, globals.num_agents + 1))
    sum_q_gap_not_br_all_states = np.zeros((num_thres_path_cycle_length + 1, globals.num_agents + 1))
    sum_q_gap_not_br_on_path = np.zeros((num_thres_path_cycle_length + 1, globals.num_agents + 1))
    sum_q_gap_not_eq_all_states = np.zeros((num_thres_path_cycle_length + 1, globals.num_agents + 1))
    sum_q_gap_not_eq_on_path = np.zeros((num_thres_path_cycle_length + 1, globals.num_agents + 1))
    
    num_q_gap_tot = np.zeros((num_thres_path_cycle_length + 1, globals.num_agents + 1), dtype=np.int32)
    num_q_gap_on_path = np.zeros((num_thres_path_cycle_length + 1, globals.num_agents + 1), dtype=np.int32)
    num_q_gap_not_on_path = np.zeros((num_thres_path_cycle_length + 1, globals.num_agents + 1), dtype=np.int32)
    num_q_gap_not_br_all_states = np.zeros((num_thres_path_cycle_length + 1, globals.num_agents + 1), dtype=np.int32)
    num_q_gap_not_br_on_path = np.zeros((num_thres_path_cycle_length + 1, globals.num_agents + 1), dtype=np.int32)
    num_q_gap_not_eq_all_states = np.zeros((num_thres_path_cycle_length + 1, globals.num_agents + 1), dtype=np.int32)
    num_q_gap_not_eq_on_path = np.zeros((num_thres_path_cycle_length + 1, globals.num_agents + 1), dtype=np.int32)
    
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
    
    # Loop over sessions
    for i_session in range(globals.num_sessions):
        optimal_strategy_vec = globals.index_strategies[:, i_session]
        optimal_strategy = optimal_strategy_vec.reshape(
            (globals.num_states, globals.num_agents)
        )
        cycle_length_session = globals.cycle_length[i_session]
        cycle_states_session = globals.cycle_states[:cycle_length_session, i_session]
        
        # Compute Q gap for the optimal strategy for all agents, in all states and actions
        (
            q_gap_tot,
            q_gap_on_path,
            q_gap_not_on_path,
            q_gap_not_br_all_states,
            q_gap_not_br_on_path,
            q_gap_not_eq_all_states,
            q_gap_not_eq_on_path
        ) = compute_q_gap_to_max_session(
            optimal_strategy,
            cycle_length_session,
            cycle_states_session,
            globals
        )
        
        # Summing by agent and threshold
        i_thres = min(globals.cycle_length[i_session], thres_path_cycle_length[-1])
        
        if not np.any(q_gap_tot <= 0.0):
            sum_q_gap_tot[0, :] += q_gap_tot
            sum_q_gap_tot[i_thres, :] += q_gap_tot
            num_q_gap_tot[0, :] += 1
            num_q_gap_tot[i_thres, :] += 1
        
        if not np.any(q_gap_on_path <= 0.0):
            sum_q_gap_on_path[0, :] += q_gap_on_path
            sum_q_gap_on_path[i_thres, :] += q_gap_on_path
            num_q_gap_on_path[0, :] += 1
            num_q_gap_on_path[i_thres, :] += 1
        
        if not np.any(q_gap_not_on_path <= 0.0):
            sum_q_gap_not_on_path[0, :] += q_gap_not_on_path
            sum_q_gap_not_on_path[i_thres, :] += q_gap_not_on_path
            num_q_gap_not_on_path[0, :] += 1
            num_q_gap_not_on_path[i_thres, :] += 1
        
        if not np.any(q_gap_not_br_all_states <= 0.0):
            sum_q_gap_not_br_all_states[0, :] += q_gap_not_br_all_states
            sum_q_gap_not_br_all_states[i_thres, :] += q_gap_not_br_all_states
            num_q_gap_not_br_all_states[0, :] += 1
            num_q_gap_not_br_all_states[i_thres, :] += 1
        
        if not np.any(q_gap_not_br_on_path <= 0.0):
            sum_q_gap_not_br_on_path[0, :] += q_gap_not_br_on_path
            sum_q_gap_not_br_on_path[i_thres, :] += q_gap_not_br_on_path
            num_q_gap_not_br_on_path[0, :] += 1
            num_q_gap_not_br_on_path[i_thres, :] += 1
        
        if not np.any(q_gap_not_eq_all_states <= 0.0):
            sum_q_gap_not_eq_all_states[0, :] += q_gap_not_eq_all_states
            sum_q_gap_not_eq_all_states[i_thres, :] += q_gap_not_eq_all_states
            num_q_gap_not_eq_all_states[0, :] += 1
            num_q_gap_not_eq_all_states[i_thres, :] += 1
        
        if not np.any(q_gap_not_eq_on_path <= 0.0):
            sum_q_gap_not_eq_on_path[0, :] += q_gap_not_eq_on_path
            sum_q_gap_not_eq_on_path[i_thres, :] += q_gap_not_eq_on_path
            num_q_gap_not_eq_on_path[0, :] += 1
            num_q_gap_not_eq_on_path[i_thres, :] += 1
    
    # Averaging
    sum_q_gap_tot = sum_q_gap_tot / num_q_gap_tot
    sum_q_gap_tot[np.isnan(sum_q_gap_tot)] = -999.999
    sum_q_gap_on_path = sum_q_gap_on_path / num_q_gap_on_path
    sum_q_gap_on_path[np.isnan(sum_q_gap_on_path)] = -999.999
    sum_q_gap_not_on_path = sum_q_gap_not_on_path / num_q_gap_not_on_path
    sum_q_gap_not_on_path[np.isnan(sum_q_gap_not_on_path)] = -999.999
    sum_q_gap_not_br_all_states = sum_q_gap_not_br_all_states / num_q_gap_not_br_all_states
    sum_q_gap_not_br_all_states[np.isnan(sum_q_gap_not_br_all_states)] = -999.999
    sum_q_gap_not_br_on_path = sum_q_gap_not_br_on_path / num_q_gap_not_br_on_path
    sum_q_gap_not_br_on_path[np.isnan(sum_q_gap_not_br_on_path)] = -999.999
    sum_q_gap_not_eq_all_states = sum_q_gap_not_eq_all_states / num_q_gap_not_eq_all_states
    sum_q_gap_not_eq_all_states[np.isnan(sum_q_gap_not_eq_all_states)] = -999.999
    sum_q_gap_not_eq_on_path = sum_q_gap_not_eq_on_path / num_q_gap_not_eq_on_path
    sum_q_gap_not_eq_on_path[np.isnan(sum_q_gap_not_eq_on_path)] = -999.999
    
    # Write results to file
    with open('A_qGapToMaximum.txt', 'a') as f:
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
            header += '   QGapTot QGaponPath QGapNotOnPath '
            header += 'QGapNotBRAllSt QGapNotBROnPath QGapNotEqAllSt QGapNotEqOnPath '
            header += ' '.join(
                f'QGapTotAg{i+1} QGaponPathAg{i+1} QGapNotOnPathAg{i+1} '
                f'QGapNotBRAllStAg{i+1} QGapNotBROnPathAg{i+1} QGapNotEqAllStAg{i+1} QGapNotEqOnPathAg{i+1} '
                for i in range(globals.num_agents)
            )
            header += ' '.join(
                f'   PL{i+1:02d}QGapTot PL{i+1:02d}QGaponPath PL{i+1:02d}QGapNotOnPath '
                f'PL{i+1:02d}QGapNotBRAllSt PL{i+1:02d}QGapNotBROnPath PL{i+1:02d}QGapNotEqAllSt PL{i+1:02d}QGapNotEqOnPath '
                + ' '.join(
                    f'PL{i+1:02d}QGapTotAg{j+1} PL{i+1:02d}QGaponPathAg{j+1} PL{i+1:02d}QGapNotOnPathAg{j+1} '
                    f'PL{i+1:02d}QGapNotBRAllStAg{j+1} PL{i+1:02d}QGapNotBROnPathAg{j+1} '
                    f'PL{i+1:02d}QGapNotEqAllStAg{j+1} PL{i+1:02d}QGapNotEqOnPathAg{j+1} '
                    for j in range(globals.num_agents)
                )
                for i in range(num_thres_path_cycle_length)
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
        result += f' {sum_q_gap_tot[0,0]:10.7f} {sum_q_gap_on_path[0,0]:10.7f} {sum_q_gap_not_on_path[0,0]:13.7f} '
        result += f'{sum_q_gap_not_br_all_states[0,0]:14.7f} {sum_q_gap_not_br_on_path[0,0]:15.7f} '
        result += f'{sum_q_gap_not_eq_all_states[0,0]:14.7f} {sum_q_gap_not_eq_on_path[0,0]:15.7f} '
        result += ' '.join(
            f'{sum_q_gap_tot[0,i+1]:10.7f} {sum_q_gap_on_path[0,i+1]:13.7f} {sum_q_gap_not_on_path[0,i+1]:16.7f} '
            f'{sum_q_gap_not_br_all_states[0,i+1]:17.7f} {sum_q_gap_not_br_on_path[0,i+1]:18.7f} '
            f'{sum_q_gap_not_eq_all_states[0,i+1]:17.7f} {sum_q_gap_not_eq_on_path[0,i+1]:18.7f}'
            for i in range(globals.num_agents)
        )
        result += ' '.join(
            f'{sum_q_gap_tot[i+1,0]:14.7f} {sum_q_gap_on_path[i+1,0]:14.7f} {sum_q_gap_not_on_path[i+1,0]:17.7f} '
            f'{sum_q_gap_not_br_all_states[i+1,0]:18.7f} {sum_q_gap_not_br_on_path[i+1,0]:19.7f} '
            f'{sum_q_gap_not_eq_all_states[i+1,0]:18.7f} {sum_q_gap_not_eq_on_path[i+1,0]:19.7f} '
            + ' '.join(
                f'{sum_q_gap_tot[i+1,j+1]:14.7f} {sum_q_gap_on_path[i+1,j+1]:17.7f} '
                f'{sum_q_gap_not_on_path[i+1,j+1]:20.7f} {sum_q_gap_not_br_all_states[i+1,j+1]:21.7f} '
                f'{sum_q_gap_not_br_on_path[i+1,j+1]:22.7f} {sum_q_gap_not_eq_all_states[i+1,j+1]:21.7f} '
                f'{sum_q_gap_not_eq_on_path[i+1,j+1]:22.7f}'
                for j in range(globals.num_agents)
            )
            for i in range(num_thres_path_cycle_length)
        )
        f.write(result + '\n')

def compute_q_gap_to_max_session(
    optimal_strategy: np.ndarray,
    cycle_length: int,
    cycle_states: np.ndarray,
    globals: GlobalVars
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Computes Q gap w.r.t. maximum by state for an individual replication
    
    Parameters:
    - optimal_strategy: strategy for all agents
    - cycle_length: length of the replication's equilibrium path (i.e., state cycle)
    - cycle_states: replication's equilibrium path (i.e., state cycle)
    
    Returns:
    - q_gap_tot: Average Q gap over all states
    - q_gap_on_path: Average Q gap over cycle states
    - q_gap_not_on_path: Average Q gap over non-cycle states
    - q_gap_not_br_all_states: Average Q gap over non-best responding states
    - q_gap_not_br_on_path: Average Q gap over non-best responding, non-cycle states
    - q_gap_not_eq_all_states: Average Q gap over non-equilibrium states
    - q_gap_not_eq_on_path: Average Q gap over non-equilibrium cycle states
    """
    # 1. Compute true Q for the optimal strategy for all agents, in all states and actions
    q_true = np.zeros((globals.num_states, globals.num_prices, globals.num_agents))
    max_q_true = np.zeros((globals.num_states, globals.num_agents))
    q_gap = np.zeros((globals.num_states, globals.num_agents))
    
    for i_state in range(globals.num_states):
        for i_agent in range(globals.num_agents):
            for i_price in range(globals.num_prices):
                # Compute state value function of agent i_agent for the optimal strategy in (i_state,i_price)
                q_true[i_state, i_price, i_agent], _, _ = compute_q_cell(
                    optimal_strategy, i_state, i_price, i_agent, globals.delta, globals
                )
            
            # Compute gap in Q function values w.r.t. maximum
            max_q_true[i_state, i_agent] = np.max(q_true[i_state, :, i_agent])
            q_gap[i_state, i_agent] = (
                max_q_true[i_state, i_agent] - q_true[i_state, optimal_strategy[i_state, i_agent], i_agent]
            ) / abs(max_q_true[i_state, i_agent])
    
    # 2. Compute mask matrices
    is_on_path = np.zeros((globals.num_states, globals.num_agents), dtype=bool)
    is_not_on_path = np.zeros((globals.num_states, globals.num_agents), dtype=bool)
    is_not_br_all_states = np.zeros((globals.num_states, globals.num_agents), dtype=bool)
    is_not_br_on_path = np.zeros((globals.num_states, globals.num_agents), dtype=bool)
    is_not_eq_all_states = np.zeros((globals.num_states, globals.num_agents), dtype=bool)
    is_not_eq_on_path = np.zeros((globals.num_states, globals.num_agents), dtype=bool)
    
    for i_state in range(globals.num_states):
        if np.any(cycle_states == i_state):
            is_on_path[i_state, :] = True
        if np.all(cycle_states != i_state):
            is_not_on_path[i_state, :] = True
        
        is_br = np.zeros(globals.num_agents, dtype=bool)
        for i_agent in range(globals.num_agents):
            if are_equal_reals(
                q_true[i_state, optimal_strategy[i_state, i_agent], i_agent],
                max_q_true[i_state, i_agent]
            ):
                is_br[i_agent] = True
            else:
                is_not_br_all_states[i_state, i_agent] = True
                if np.any(cycle_states == i_state):
                    is_not_br_on_path[i_state, i_agent] = True
        
        for i_agent in range(globals.num_agents):
            if not np.all(is_br):
                is_not_eq_all_states[i_state, i_agent] = True
                if np.any(cycle_states == i_state):
                    is_not_eq_on_path[i_state, i_agent] = True
    
    # 3. Compute Q gap averages over subsets of states
    q_gap_tot = np.zeros(globals.num_agents + 1)
    q_gap_on_path = np.zeros(globals.num_agents + 1)
    q_gap_not_on_path = np.zeros(globals.num_agents + 1)
    q_gap_not_br_all_states = np.zeros(globals.num_agents + 1)
    q_gap_not_br_on_path = np.zeros(globals.num_agents + 1)
    q_gap_not_eq_all_states = np.zeros(globals.num_agents + 1)
    q_gap_not_eq_on_path = np.zeros(globals.num_agents + 1)
    
    q_gap_tot[0] = np.sum(q_gap) / (globals.num_agents * globals.num_states)
    q_gap_on_path[0] = np.sum(q_gap[is_on_path]) / np.sum(is_on_path)
    q_gap_not_on_path[0] = np.sum(q_gap[is_not_on_path]) / np.sum(is_not_on_path)
    q_gap_not_br_all_states[0] = np.sum(q_gap[is_not_br_all_states]) / np.sum(is_not_br_all_states)
    q_gap_not_br_on_path[0] = np.sum(q_gap[is_not_br_on_path]) / np.sum(is_not_br_on_path)
    q_gap_not_eq_all_states[0] = np.sum(q_gap[is_not_eq_all_states]) / np.sum(is_not_eq_all_states)
    q_gap_not_eq_on_path[0] = np.sum(q_gap[is_not_eq_on_path]) / np.sum(is_not_eq_on_path)
    
    for i_agent in range(globals.num_agents):
        q_gap_tot[i_agent + 1] = np.sum(q_gap[:, i_agent]) / globals.num_states
        q_gap_on_path[i_agent + 1] = np.sum(q_gap[:, i_agent][is_on_path[:, i_agent]]) / np.sum(is_on_path[:, i_agent])
        q_gap_not_on_path[i_agent + 1] = np.sum(q_gap[:, i_agent][is_not_on_path[:, i_agent]]) / np.sum(is_not_on_path[:, i_agent])
        q_gap_not_br_all_states[i_agent + 1] = np.sum(q_gap[:, i_agent][is_not_br_all_states[:, i_agent]]) / np.sum(is_not_br_all_states[:, i_agent])
        q_gap_not_br_on_path[i_agent + 1] = np.sum(q_gap[:, i_agent][is_not_br_on_path[:, i_agent]]) / np.sum(is_not_br_on_path[:, i_agent])
        q_gap_not_eq_all_states[i_agent + 1] = np.sum(q_gap[:, i_agent][is_not_eq_all_states[:, i_agent]]) / np.sum(is_not_eq_all_states[:, i_agent])
        q_gap_not_eq_on_path[i_agent + 1] = np.sum(q_gap[:, i_agent][is_not_eq_on_path[:, i_agent]]) / np.sum(is_not_eq_on_path[:, i_agent])
    
    q_gap_tot[np.isnan(q_gap_tot)] = -999.999
    q_gap_on_path[np.isnan(q_gap_on_path)] = -999.999
    q_gap_not_on_path[np.isnan(q_gap_not_on_path)] = -999.999
    q_gap_not_br_all_states[np.isnan(q_gap_not_br_all_states)] = -999.999
    q_gap_not_br_on_path[np.isnan(q_gap_not_br_on_path)] = -999.999
    q_gap_not_eq_all_states[np.isnan(q_gap_not_eq_all_states)] = -999.999
    q_gap_not_eq_on_path[np.isnan(q_gap_not_eq_on_path)] = -999.999
    
    return (
        q_gap_tot, q_gap_on_path, q_gap_not_on_path,
        q_gap_not_br_all_states, q_gap_not_br_on_path,
        q_gap_not_eq_all_states, q_gap_not_eq_on_path
    ) 