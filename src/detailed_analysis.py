import numpy as np
from typing import List, Tuple, Dict
from globals import GlobalVars
from QL_routines import compute_q_cell, compute_dynamic_best_response
from generic_routines import are_equal_reals, convert_number_base, compute_state_number
from impulse_response import compute_static_best_response, compute_individual_ir
from equilibrium_check import compute_eq_check_session
from q_gap_to_maximum import compute_q_gap_to_max_session

def compute_detailed_analysis(i_experiment: int, globals: GlobalVars) -> None:
    """
    Computes disaggregated analysis
    
    Parameters:
    - i_experiment: model index
    """
    print('Computing Detailed Analysis')
    
    # Open output file
    file_name = f"A_det_{globals.experiment_number}"
    with open(file_name, 'w') as f:
        # Write header
        header = ' Session  DevToPrice '
        header += ' '.join(f'NashProfit{i+1} ' for i in range(globals.num_agents))
        header += ' '.join(f'CoopProfit{i+1} ' for i in range(globals.num_agents))
        header += ' PreShockCycleLength '
        header += ' '.join(f'  AvgPrePrice{i+1} ' for i in range(globals.num_agents))
        header += ' '.join(f'  AvgPreProfit{i+1} ' for i in range(globals.num_agents))
        header += ' '.join(f'ProfitGain{i+1} ' for i in range(globals.num_agents))
        header += 'Converged TimeToConvergence  PreShockNumInCycle '
        header += 'flagEQAll flagEQOnPath flagEQOffPath '
        header += 'freqEQAll freqEQOnPath freqEQOffPath '
        header += ' '.join(f'flagBRAll{i+1} ' for i in range(globals.num_agents))
        header += ' '.join(f'flagBROnPath{i+1} ' for i in range(globals.num_agents))
        header += ' '.join(f'flagBROffPath{i+1} ' for i in range(globals.num_agents))
        header += ' '.join(f'freqBRAll{i+1} ' for i in range(globals.num_agents))
        header += ' '.join(f'freqBROnPath{i+1} ' for i in range(globals.num_agents))
        header += ' '.join(f'freqBROffPath{i+1} ' for i in range(globals.num_agents))
        header += '   QGapTot QGapOnPath QGapNotOnPath QGapNotBRAllStates '
        header += 'QGapNotBRonPath QGapNotEqAllStates QGapNotEqonPath '
        header += ' '.join(f'   QGapTot{i+1} ' for i in range(globals.num_agents))
        header += ' '.join(f'QGapOnPath{i+1} ' for i in range(globals.num_agents))
        header += ' '.join(f'QGapNotOnPath{i+1} ' for i in range(globals.num_agents))
        header += ' '.join(f'QGapNotBRAllStates{i+1} ' for i in range(globals.num_agents))
        header += ' '.join(f'QGapNotBRonPath{i+1} ' for i in range(globals.num_agents))
        header += ' '.join(f'QGapNotEqAllStates{i+1} ' for i in range(globals.num_agents))
        header += ' '.join(f'QGapNotEqonPath{i+1} ' for i in range(globals.num_agents))
        header += ' '.join(f' PreShockPrice{i+1} ' for i in range(globals.num_agents))
        header += ' '.join(f' PreShockProfit{i+1} ' for i in range(globals.num_agents))
        header += ' ShockAgent  ObsAgent   DeviationQ ShockLength SameCyclePrePost DevAgStaticBR001 '
        header += ' '.join(f' ShockPrice{i+1:03d} ' for i in range(globals.num_shock_periods_print))
        header += ' '.join(f' ShockProfit{i+1:03d} ' for i in range(globals.num_shock_periods_print))
        header += ' '.join(f' StaticBRPrice{i+1:03d} ' for i in range(globals.num_shock_periods_print))
        header += ' '.join(f' DynamicBRPrice{i+1:03d} ' for i in range(globals.num_shock_periods_print))
        header += ' '.join(f' OptStratQ{i+1:03d} ' for i in range(globals.num_shock_periods_print))
        header += ' '.join(f' DynamicBRQ{i+1:03d} ' for i in range(globals.num_shock_periods_print))
        header += ' PostShockCycleLength '
        header += ' '.join(f'  AvgPostPrice{i+1} ' for i in range(globals.num_agents))
        header += ' '.join(f'  AvgPostProfit{i+1} ' for i in range(globals.num_agents))
        f.write(header + '\n')
        
        # Read strategies and states at convergence from file
        with open(globals.file_name_info_experiment, 'r') as info_file:
            for i_session in range(globals.num_sessions):
                print(f'Session = {i_session + 1} started')
                
                r_session = int(info_file.readline().strip())
                globals.converged[r_session] = int(info_file.readline().strip())
                globals.time_to_convergence[r_session] = float(info_file.readline().strip())
                globals.index_last_state[:, r_session] = np.array(
                    [int(x) for x in info_file.readline().strip().split()]
                )
                
                for i_state in range(globals.num_states):
                    globals.index_strategies[
                        i_state::globals.num_states, r_session
                    ] = np.array([int(x) for x in info_file.readline().strip().split()])
                
                optimal_strategy_vec = globals.index_strategies[:, i_session]
                optimal_strategy = optimal_strategy_vec.reshape(
                    (globals.num_states, globals.num_agents)
                )
                
                # Pre-shock period analysis
                periods_length_pre = globals.cycle_length[i_session]
                visited_states_pre = np.zeros(globals.num_periods, dtype=np.int32)
                pre_prices = np.zeros((globals.num_periods, globals.num_agents))
                pre_profits = np.zeros((globals.num_periods, globals.num_agents))
                ind_pre_prices = np.zeros((globals.num_periods, globals.num_agents), dtype=np.int32)
                avg_pre_prices = np.zeros(globals.num_agents)
                avg_pre_profits = np.zeros(globals.num_agents)
                
                for i_period in range(periods_length_pre):
                    visited_states_pre[i_period] = globals.cycle_states[i_period, i_session]
                    for i_agent in range(globals.num_agents):
                        ind_pre_prices[i_period, i_agent] = globals.cycle_prices[i_agent, i_period, i_session]
                        pre_prices[i_period, i_agent] = globals.prices_grids[ind_pre_prices[i_period, i_agent], i_agent]
                        pre_profits[i_period, i_agent] = globals.cycle_profits[i_agent, i_period, i_session]
                
                avg_pre_prices = np.sum(pre_prices[:periods_length_pre, :], axis=0) / periods_length_pre
                avg_pre_profits = np.sum(pre_profits[:periods_length_pre, :], axis=0) / periods_length_pre
                
                # Compute indicators that depend on the strategy only
                profit_gains = (avg_pre_profits - globals.nash_profits) / (globals.coop_profits - globals.nash_profits)
                
                # Compute equilibrium check
                (
                    freq_br_all, freq_br_on_path, freq_br_off_path,
                    freq_eq_all, freq_eq_on_path, freq_eq_off_path,
                    flag_br_all, flag_br_on_path, flag_br_off_path,
                    flag_eq_all, flag_eq_on_path, flag_eq_off_path
                ) = compute_eq_check_session(
                    optimal_strategy,
                    periods_length_pre,
                    visited_states_pre[:periods_length_pre],
                    globals
                )
                
                # Compute Q gap
                (
                    q_gap_tot_session, q_gap_on_path_session, q_gap_not_on_path_session,
                    q_gap_not_br_all_states_session, q_gap_not_br_on_path_session,
                    q_gap_not_eq_all_states_session, q_gap_not_eq_on_path_session
                ) = compute_q_gap_to_max_session(
                    optimal_strategy,
                    periods_length_pre,
                    globals.cycle_states[:periods_length_pre, i_session],
                    globals
                )
                
                # IR analysis with deviation to i_price
                for i_price in range(globals.num_prices):
                    for i_state_pre in range(periods_length_pre):
                        for i_agent in range(globals.num_agents):
                            shock_prices = np.zeros((globals.num_shock_periods_print, globals.num_agents), dtype=np.int32)
                            shock_real_prices = np.zeros((globals.num_shock_periods_print, globals.num_agents))
                            shock_profits = np.zeros((globals.num_shock_periods_print, globals.num_agents))
                            static_br_prices = np.zeros((globals.num_shock_periods_print, globals.num_agents), dtype=np.int32)
                            dynamic_br_prices = np.zeros((globals.num_shock_periods_print, globals.num_agents), dtype=np.int32)
                            opt_strat_q = np.zeros((globals.num_shock_periods_print, globals.num_agents))
                            dynamic_br_q = np.zeros((globals.num_shock_periods_print, globals.num_agents))
                            deviation_q = np.zeros(globals.num_agents)
                            avg_post_prices = np.zeros(globals.num_agents)
                            avg_post_profits = np.zeros(globals.num_agents)
                            visited_states = np.zeros(globals.num_periods, dtype=np.int32)
                            
                            # Find prices and Qs in deviation period n. 1
                            p_prime = optimal_strategy[visited_states_pre[i_state_pre], :].copy()
                            p_prime[i_agent] = i_price
                            for j_agent in range(globals.num_agents):
                                deviation_q[j_agent], _, _, _ = compute_q_cell(
                                    optimal_strategy,
                                    visited_states_pre[i_state_pre],
                                    p_prime[j_agent],
                                    j_agent,
                                    globals.delta,
                                    globals
                                )
                            
                            # Computing individual IRs
                            (
                                shock_states, shock_prices, shock_real_prices,
                                shock_profits, avg_post_prices, avg_post_profits,
                                shock_length, same_cycle_pre_post, post_length
                            ) = compute_individual_ir(
                                optimal_strategy,
                                visited_states_pre[i_state_pre],
                                i_agent,
                                i_price,
                                1,
                                globals.num_shock_periods_print,
                                periods_length_pre,
                                visited_states_pre[:periods_length_pre],
                                globals
                            )
                            
                            # Computing additional information
                            for i_period in range(globals.num_shock_periods_print):
                                p = convert_number_base(
                                    shock_states[i_period]-1,
                                    globals.num_prices,
                                    globals.num_agents * globals.depth_state
                                )
                                p = p.reshape((globals.depth_state, globals.num_agents))
                                shock_prices[i_period, :] = p[0, :]
                                
                                if i_period == 0:
                                    i_period_state = visited_states_pre[i_state_pre]
                                else:
                                    i_period_state = shock_states[i_period-1]
                                
                                for j_agent in range(globals.num_agents):
                                    # Find DynamicBR prices and Qs
                                    dynamic_br_prices[i_period, j_agent], dynamic_br_q[i_period, j_agent] = (
                                        compute_dynamic_best_response(
                                            optimal_strategy,
                                            i_period_state,
                                            j_agent,
                                            globals.delta,
                                            globals
                                        )
                                    )
                                    
                                    # Find prices and Qs according to the strategy at convergence
                                    opt_strat_q[i_period, j_agent], _, _, _ = compute_q_cell(
                                        optimal_strategy,
                                        i_period_state,
                                        optimal_strategy[i_period_state, j_agent],
                                        j_agent,
                                        globals.delta,
                                        globals
                                    )
                                    
                                    # Find StaticBR prices and PIs
                                    static_br_prices[i_period, j_agent], _ = compute_static_best_response(
                                        optimal_strategy,
                                        visited_states_pre[i_state_pre],
                                        j_agent,
                                        static_br_prices[i_period, j_agent],
                                        globals
                                    )
                            
                            # Write results to output file
                            for j_agent in range(globals.num_agents):
                                result = f'{i_session+1:8d} {i_price+1:11d} '
                                result += ' '.join(f'{globals.nash_profits[i]:12.5f}' for i in range(globals.num_agents))
                                result += ' '
                                result += ' '.join(f'{globals.coop_profits[i]:12.5f}' for i in range(globals.num_agents))
                                result += f' {periods_length_pre:20d} '
                                result += ' '.join(f'{avg_pre_prices[i]:14.5f}' for i in range(globals.num_agents))
                                result += ' '
                                result += ' '.join(f'{avg_pre_profits[i]:15.5f}' for i in range(globals.num_agents))
                                result += ' '
                                result += ' '.join(f'{profit_gains[i]:11.5f}' for i in range(globals.num_agents))
                                result += f' {globals.converged[i_session]:9d} {globals.time_to_convergence[i_session]:17.5f} {i_state_pre+1:19d} '
                                result += f'{flag_eq_all:9d} {flag_eq_on_path:12d} {flag_eq_off_path:13d} '
                                result += f'{freq_eq_all:9.5f} {freq_eq_on_path:12.5f} {freq_eq_off_path:13.5f} '
                                result += ' '.join(f'{flag_br_all[i]:10d}' for i in range(globals.num_agents))
                                result += ' '
                                result += ' '.join(f'{flag_br_on_path[i]:13d}' for i in range(globals.num_agents))
                                result += ' '
                                result += ' '.join(f'{flag_br_off_path[i]:14d}' for i in range(globals.num_agents))
                                result += ' '
                                result += ' '.join(f'{freq_br_all[i]:10.5f}' for i in range(globals.num_agents))
                                result += ' '
                                result += ' '.join(f'{freq_br_on_path[i]:13.5f}' for i in range(globals.num_agents))
                                result += ' '
                                result += ' '.join(f'{freq_br_off_path[i]:14.5f}' for i in range(globals.num_agents))
                                result += f' {q_gap_tot_session[0]:10.5f} {q_gap_on_path_session[0]:10.5f} {q_gap_not_on_path_session[0]:13.5f} '
                                result += f'{q_gap_not_br_all_states_session[0]:18.5f} {q_gap_not_br_on_path_session[0]:15.5f} '
                                result += f'{q_gap_not_eq_all_states_session[0]:18.5f} {q_gap_not_eq_on_path_session[0]:15.5f} '
                                result += ' '.join(f'{q_gap_tot_session[i+1]:11.5f}' for i in range(globals.num_agents))
                                result += ' '
                                result += ' '.join(f'{q_gap_on_path_session[i+1]:11.5f}' for i in range(globals.num_agents))
                                result += ' '
                                result += ' '.join(f'{q_gap_not_on_path_session[i+1]:14.5f}' for i in range(globals.num_agents))
                                result += ' '
                                result += ' '.join(f'{q_gap_not_br_all_states_session[i+1]:19.5f}' for i in range(globals.num_agents))
                                result += ' '
                                result += ' '.join(f'{q_gap_not_br_on_path_session[i+1]:16.5f}' for i in range(globals.num_agents))
                                result += ' '
                                result += ' '.join(f'{q_gap_not_eq_all_states_session[i+1]:19.5f}' for i in range(globals.num_agents))
                                result += ' '
                                result += ' '.join(f'{q_gap_not_eq_on_path_session[i+1]:16.5f}' for i in range(globals.num_agents))
                                result += ' '
                                result += ' '.join(f'{ind_pre_prices[i_state_pre, i]:15d}' for i in range(globals.num_agents))
                                result += ' '
                                result += ' '.join(f'{pre_profits[i_state_pre, i]:16.5f}' for i in range(globals.num_agents))
                                result += f' {i_agent+1:11d} {j_agent+1:9d} {deviation_q[j_agent]:12.5f} {shock_length:11d} '
                                result += f'{same_cycle_pre_post:16d} {static_br_prices[0, i_agent]:16d} '
                                result += ' '.join(f'{shock_prices[i, j_agent]:14d}' for i in range(globals.num_shock_periods_print))
                                result += ' '
                                result += ' '.join(f'{shock_profits[i, j_agent]:15.5f}' for i in range(globals.num_shock_periods_print))
                                result += ' '
                                result += ' '.join(f'{static_br_prices[i, j_agent]:17d}' for i in range(globals.num_shock_periods_print))
                                result += ' '
                                result += ' '.join(f'{dynamic_br_prices[i, j_agent]:18d}' for i in range(globals.num_shock_periods_print))
                                result += ' '
                                result += ' '.join(f'{opt_strat_q[i, j_agent]:13.5f}' for i in range(globals.num_shock_periods_print))
                                result += ' '
                                result += ' '.join(f'{dynamic_br_q[i, j_agent]:14.5f}' for i in range(globals.num_shock_periods_print))
                                result += f' {post_length:21d} '
                                result += ' '.join(f'{avg_post_prices[i]:15.5f}' for i in range(globals.num_agents))
                                result += ' '
                                result += ' '.join(f'{avg_post_profits[i]:16.5f}' for i in range(globals.num_agents))
                                f.write(result + '\n')
                
                print(f'Session = {i_session + 1} completed') 