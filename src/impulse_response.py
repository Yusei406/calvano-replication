import numpy as np
from typing import List, Tuple, Dict
from globals import GlobalVars
from QL_routines import compute_state_number, compute_action_number, compute_q_cell
from generic_routines import convert_number_base

def compute_ir_analysis(i_experiment: int, unit_number: int, ir_type: int, globals: GlobalVars) -> None:
    """
    Computes and prints to file the Impulse Response summary statistics for a price deviation
    
    Parameters:
    - i_experiment: model index
    - unit_number: output unit number
    - ir_type: index to select the deviation type:
        ir_type <= -1: one-period deviation to the ir_type-th price
        ir_type = 0: one-period deviation to static BR
        1 <= ir_type <= 999: ir_type-period deviation to Nash
        ir_type = 1000: permanent deviation to Nash
    """
    print('Computing Impulse Responses')
    
    # Initialize variables
    num_thres_periods_length = 10
    num_thres_periods_length0 = num_thres_periods_length + 1
    num_shock_periods_print = 25
    
    # Initialize arrays
    freq_pre_length = np.zeros(num_thres_periods_length, dtype=np.int32)
    avg_pre_prices = np.zeros(globals.num_agents)
    avg_pre_profits = np.zeros(globals.num_agents)
    avg_pre_prices_q = np.zeros(globals.num_agents)
    avg_pre_profits_q = np.zeros(globals.num_agents)
    
    freq_shock_length = np.zeros((globals.num_agents, num_thres_periods_length), dtype=np.int32)
    freq_punishment_strategy = np.zeros((globals.num_agents, num_thres_periods_length0), dtype=np.int32)
    avg_shock_prices = np.zeros((num_shock_periods_print, globals.num_agents, globals.num_agents))
    avg_shock_prices_q = np.zeros((num_shock_periods_print, globals.num_agents, globals.num_agents))
    avg_shock_profits = np.zeros((num_shock_periods_print, globals.num_agents, globals.num_agents))
    avg_shock_profits_q = np.zeros((num_shock_periods_print, globals.num_agents, globals.num_agents))
    
    freq_post_length = np.zeros((globals.num_agents, num_thres_periods_length), dtype=np.int32)
    avg_post_prices = np.zeros((globals.num_agents, globals.num_agents))
    avg_post_prices_q = np.zeros((globals.num_agents, globals.num_agents))
    avg_post_profits = np.zeros((globals.num_agents, globals.num_agents))
    avg_post_profits_q = np.zeros((globals.num_agents, globals.num_agents))
    
    thres_periods_length = np.arange(1, num_thres_periods_length + 1)
    thres_periods_length0 = np.concatenate(([0], thres_periods_length))
    
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
        
        # Pre-shock period analysis
        periods_length_pre = globals.cycle_length[i_session]
        pos_thres = min(num_thres_periods_length, periods_length_pre)
        freq_pre_length[pos_thres - 1] += 1
        
        visited_states_pre = np.zeros(globals.num_periods, dtype=np.int32)
        pre_prices = np.zeros((globals.num_periods, globals.num_agents))
        pre_profits = np.zeros((globals.num_periods, globals.num_agents))
        
        for i_period in range(periods_length_pre):
            visited_states_pre[i_period] = globals.cycle_states[i_period, i_session]
            for i_agent in range(globals.num_agents):
                pre_prices[i_period, i_agent] = globals.prices_grids[
                    globals.cycle_prices[i_agent, i_period, i_session], i_agent
                ]
                pre_profits[i_period, i_agent] = globals.cycle_profits[
                    i_agent, i_period, i_session
                ]
        
        avg_pre_prices += np.mean(pre_prices[:periods_length_pre, :], axis=0)
        avg_pre_prices_q += np.mean(pre_prices[:periods_length_pre, :], axis=0) ** 2
        avg_pre_profits += np.mean(pre_profits[:periods_length_pre, :], axis=0)
        avg_pre_profits_q += np.mean(pre_profits[:periods_length_pre, :], axis=0) ** 2
        
        # Shock period analysis
        for i_agent in range(globals.num_agents):
            avg_shock_prices_tmp = np.zeros((num_shock_periods_print, globals.num_agents))
            avg_shock_profits_tmp = np.zeros((num_shock_periods_print, globals.num_agents))
            avg_post_prices_tmp = np.zeros(globals.num_agents)
            avg_post_profits_tmp = np.zeros(globals.num_agents)
            
            for i_state_pre in range(periods_length_pre):
                # Select deviation type
                if ir_type <= -1:
                    # One-period deviation to the ir_type-th price
                    dev_price = -ir_type
                    dev_length = 1
                elif ir_type == 0:
                    # One-period deviation to static BR
                    dev_price, _ = compute_static_best_response(
                        optimal_strategy, visited_states_pre[i_state_pre], i_agent, globals
                    )
                    dev_length = 1
                else:
                    # dev_length-period deviation to Nash
                    dev_price = np.argmin(
                        (globals.prices_grids[:, i_agent] - globals.nash_prices[i_agent]) ** 2
                    )
                    dev_length = ir_type
                
                # Compute individual IRs
                shock_states, shock_ind_prices, shock_prices, shock_profits, \
                post_prices, post_profits, shock_length, punishment_strategy, post_length = \
                    compute_individual_ir(
                        optimal_strategy, visited_states_pre[i_state_pre], i_agent,
                        dev_price, dev_length, num_shock_periods_print,
                        periods_length_pre, visited_states_pre[:periods_length_pre],
                        globals
                    )
                
                # Compute running averages
                nn = float(i_state_pre + 1)
                avg_shock_prices_tmp = (nn - 1.0) / nn * avg_shock_prices_tmp + shock_prices / nn
                avg_shock_profits_tmp = (nn - 1.0) / nn * avg_shock_profits_tmp + shock_profits / nn
                avg_post_prices_tmp = (nn - 1.0) / nn * avg_post_prices_tmp + post_prices / nn
                avg_post_profits_tmp = (nn - 1.0) / nn * avg_post_profits_tmp + post_profits / nn
                
                freq_shock_length[i_agent, min(num_thres_periods_length, shock_length) - 1] += 1
                freq_punishment_strategy[i_agent, min(num_thres_periods_length, punishment_strategy)] += 1
                freq_post_length[i_agent, min(num_thres_periods_length, post_length) - 1] += 1
            
            # Compute average prices and profits over pre-shock cycle states
            avg_shock_prices[:, i_agent, :] += avg_shock_prices_tmp
            avg_shock_prices_q[:, i_agent, :] += avg_shock_prices_tmp ** 2
            avg_shock_profits[:, i_agent, :] += avg_shock_profits_tmp
            avg_shock_profits_q[:, i_agent, :] += avg_shock_profits_tmp ** 2
            avg_post_prices[i_agent, :] += avg_post_prices_tmp
            avg_post_prices_q[i_agent, :] += avg_post_prices_tmp ** 2
            avg_post_profits[i_agent, :] += avg_post_profits_tmp
            avg_post_profits_q[i_agent, :] += avg_post_profits_tmp ** 2
    
    # Compute averages and descriptive statistics
    # Averages of prices and profits
    avg_pre_prices /= globals.num_sessions
    avg_pre_profits /= globals.num_sessions
    avg_shock_prices /= globals.num_sessions
    avg_shock_profits /= globals.num_sessions
    avg_post_prices /= globals.num_sessions
    avg_post_profits /= globals.num_sessions
    avg_pre_prices_q /= globals.num_sessions
    avg_pre_profits_q /= globals.num_sessions
    avg_shock_prices_q /= globals.num_sessions
    avg_shock_profits_q /= globals.num_sessions
    avg_post_prices_q /= globals.num_sessions
    avg_post_profits_q /= globals.num_sessions
    
    # Compute aggregate (deviating and non-deviating) averages of prices and profits
    aggr_pre_prices = np.mean(avg_pre_prices)
    aggr_pre_profits = np.mean(avg_pre_profits)
    aggr_pre_prices_q = np.mean(avg_pre_prices_q)
    aggr_pre_profits_q = np.mean(avg_pre_profits_q)
    
    aggr_dev_shock_prices = np.zeros(num_shock_periods_print)
    aggr_dev_shock_profits = np.zeros(num_shock_periods_print)
    aggr_dev_shock_prices_q = np.zeros(num_shock_periods_print)
    aggr_dev_shock_profits_q = np.zeros(num_shock_periods_print)
    
    for i_period in range(num_shock_periods_print):
        for i_agent in range(globals.num_agents):
            aggr_dev_shock_prices[i_period] += avg_shock_prices[i_period, i_agent, i_agent]
            aggr_dev_shock_profits[i_period] += avg_shock_profits[i_period, i_agent, i_agent]
            aggr_dev_shock_prices_q[i_period] += avg_shock_prices_q[i_period, i_agent, i_agent]
            aggr_dev_shock_profits_q[i_period] += avg_shock_profits_q[i_period, i_agent, i_agent]
        
        aggr_non_dev_shock_prices = (
            np.sum(avg_shock_prices[i_period, :, :]) - aggr_dev_shock_prices[i_period]
        ) / (globals.num_agents * (globals.num_agents - 1))
        aggr_dev_shock_prices[i_period] /= globals.num_agents
        aggr_non_dev_shock_profits = (
            np.sum(avg_shock_profits[i_period, :, :]) - aggr_dev_shock_profits[i_period]
        ) / (globals.num_agents * (globals.num_agents - 1))
        aggr_dev_shock_profits[i_period] /= globals.num_agents
        aggr_non_dev_shock_prices_q = (
            np.sum(avg_shock_prices_q[i_period, :, :]) - aggr_dev_shock_prices_q[i_period]
        ) / (globals.num_agents * (globals.num_agents - 1))
        aggr_dev_shock_prices_q[i_period] /= globals.num_agents
        aggr_non_dev_shock_profits_q = (
            np.sum(avg_shock_profits_q[i_period, :, :]) - aggr_dev_shock_profits_q[i_period]
        ) / (globals.num_agents * (globals.num_agents - 1))
        aggr_dev_shock_profits_q[i_period] /= globals.num_agents
    
    aggr_dev_post_prices = 0.0
    aggr_dev_post_profits = 0.0
    aggr_dev_post_prices_q = 0.0
    aggr_dev_post_profits_q = 0.0
    
    for i_agent in range(globals.num_agents):
        aggr_dev_post_prices += avg_post_prices[i_agent, i_agent]
        aggr_dev_post_profits += avg_post_profits[i_agent, i_agent]
        aggr_dev_post_prices_q += avg_post_prices_q[i_agent, i_agent]
        aggr_dev_post_profits_q += avg_post_profits_q[i_agent, i_agent]
    
    aggr_non_dev_post_prices = (
        np.sum(avg_post_prices) - aggr_dev_post_prices
    ) / (globals.num_agents * (globals.num_agents - 1))
    aggr_dev_post_prices /= globals.num_agents
    aggr_non_dev_post_profits = (
        np.sum(avg_post_profits) - aggr_dev_post_profits
    ) / (globals.num_agents * (globals.num_agents - 1))
    aggr_dev_post_profits /= globals.num_agents
    aggr_non_dev_post_prices_q = (
        np.sum(avg_post_prices_q) - aggr_dev_post_prices_q
    ) / (globals.num_agents * (globals.num_agents - 1))
    aggr_dev_post_prices_q /= globals.num_agents
    aggr_non_dev_post_profits_q = (
        np.sum(avg_post_profits_q) - aggr_dev_post_profits_q
    ) / (globals.num_agents * (globals.num_agents - 1))
    aggr_dev_post_profits_q /= globals.num_agents
    
    # Compute standard errors
    avg_pre_prices_q = np.sqrt(np.abs(avg_pre_prices_q - avg_pre_prices ** 2))
    avg_pre_profits_q = np.sqrt(np.abs(avg_pre_profits_q - avg_pre_profits ** 2))
    avg_shock_prices_q = np.sqrt(np.abs(avg_shock_prices_q - avg_shock_prices ** 2))
    avg_shock_profits_q = np.sqrt(np.abs(avg_shock_profits_q - avg_shock_profits ** 2))
    avg_post_prices_q = np.sqrt(np.abs(avg_post_prices_q - avg_post_prices ** 2))
    avg_post_profits_q = np.sqrt(np.abs(avg_post_profits_q - avg_post_profits ** 2))
    
    aggr_pre_prices_q = np.sqrt(np.abs(aggr_pre_prices_q - aggr_pre_prices ** 2))
    aggr_pre_profits_q = np.sqrt(np.abs(aggr_pre_profits_q - aggr_pre_profits ** 2))
    
    for i_period in range(num_shock_periods_print):
        aggr_non_dev_shock_prices_q[i_period] = np.sqrt(
            np.abs(aggr_non_dev_shock_prices_q[i_period] - aggr_non_dev_shock_prices[i_period] ** 2)
        )
        aggr_dev_shock_prices_q[i_period] = np.sqrt(
            np.abs(aggr_dev_shock_prices_q[i_period] - aggr_dev_shock_prices[i_period] ** 2)
        )
        aggr_non_dev_shock_profits_q[i_period] = np.sqrt(
            np.abs(aggr_non_dev_shock_profits_q[i_period] - aggr_non_dev_shock_profits[i_period] ** 2)
        )
        aggr_dev_shock_profits_q[i_period] = np.sqrt(
            np.abs(aggr_dev_shock_profits_q[i_period] - aggr_dev_shock_profits[i_period] ** 2)
        )
    
    aggr_non_dev_post_prices_q = np.sqrt(
        np.abs(aggr_non_dev_post_prices_q - aggr_non_dev_post_prices ** 2)
    )
    aggr_dev_post_prices_q = np.sqrt(
        np.abs(aggr_dev_post_prices_q - aggr_dev_post_prices ** 2)
    )
    aggr_non_dev_post_profits_q = np.sqrt(
        np.abs(aggr_non_dev_post_profits_q - aggr_non_dev_post_profits ** 2)
    )
    aggr_dev_post_profits_q = np.sqrt(
        np.abs(aggr_dev_post_profits_q - aggr_dev_post_profits ** 2)
    )
    
    # Write results to file
    with open('A_impulseResponse.txt', 'a') as f:
        # Write header for first experiment
        if i_experiment == 0:
            header = 'Experiment IRType '
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
            header += 'AggrPricePre '
            header += ' '.join(f'AggrDevPriceShockPer{i+1:03d} ' for i in range(num_shock_periods_print))
            header += 'AggrDevPricePost '
            header += ' '.join(f'AggrNonDevPriceShockPer{i+1:03d} ' for i in range(num_shock_periods_print))
            header += 'AggrNonDevPricePost '
            header += 'seAggrPricePre '
            header += ' '.join(f'seAggrDevPriceShockPer{i+1:03d} ' for i in range(num_shock_periods_print))
            header += 'seAggrDevPricePost '
            header += ' '.join(f'seAggrNonDevPriceShockPer{i+1:03d} ' for i in range(num_shock_periods_print))
            header += 'seAggrNonDevPricePost '
            header += 'AggrProfitPre '
            header += ' '.join(f'AggrDevProfitShockPer{i+1:03d} ' for i in range(num_shock_periods_print))
            header += 'AggrDevProfitPost '
            header += ' '.join(f'AggrNonDevProfitShockPer{i+1:03d} ' for i in range(num_shock_periods_print))
            header += 'AggrNonDevProfitPost '
            header += 'seAggrProfitPre '
            header += ' '.join(f'seAggrDevProfitShockPer{i+1:03d} ' for i in range(num_shock_periods_print))
            header += 'seAggrDevProfitPost '
            header += ' '.join(f'seAggrNonDevProfitShockPer{i+1:03d} ' for i in range(num_shock_periods_print))
            header += 'seAggrNonDevProfitPost '
            header += ' '.join(f'#PerLenPre={i+1:02d} ' for i in range(num_thres_periods_length))
            header += ' '.join(
                ' '.join(f'#PerLenPostAg{i+1}={j+1:02d} ' for j in range(num_thres_periods_length))
                for i in range(globals.num_agents)
            )
            header += ' '.join(
                ' '.join(f'#PerLenShockAg{i+1}={j+1:02d} ' for j in range(num_thres_periods_length))
                for i in range(globals.num_agents)
            )
            header += ' '.join(
                ' '.join(f'#PunishStratAg{i+1}={j:02d} ' for j in range(num_thres_periods_length0))
                for i in range(globals.num_agents)
            )
            header += ' '.join(
                ' '.join(
                    f'Ag{i+1}AvgPricePre ' +
                    ' '.join(f'Ag{i+1}AvgPriceShockAg{j+1}Per{k+1:03d} ' for k in range(num_shock_periods_print)) +
                    f'Ag{i+1}AvgPricePostAg{j+1} '
                    for j in range(globals.num_agents)
                )
                for i in range(globals.num_agents)
            )
            header += ' '.join(
                ' '.join(
                    f'seAg{i+1}AvgPricePre ' +
                    ' '.join(f'seAg{i+1}AvgPriceShockAg{j+1}Per{k+1:03d} ' for k in range(num_shock_periods_print)) +
                    f'seAg{i+1}AvgPricePostAg{j+1} '
                    for j in range(globals.num_agents)
                )
                for i in range(globals.num_agents)
            )
            header += ' '.join(
                ' '.join(
                    f'Ag{i+1}AvgProfitPre ' +
                    ' '.join(f'Ag{i+1}AvgProfitShockAg{j+1}Per{k+1:03d} ' for k in range(num_shock_periods_print)) +
                    f'Ag{i+1}AvgProfitPostAg{j+1} '
                    for j in range(globals.num_agents)
                )
                for i in range(globals.num_agents)
            )
            header += ' '.join(
                ' '.join(
                    f'seAg{i+1}AvgProfitPre ' +
                    ' '.join(f'seAg{i+1}AvgProfitShockAg{j+1}Per{k+1:03d} ' for k in range(num_shock_periods_print)) +
                    f'seAg{i+1}AvgProfitPostAg{j+1} '
                    for j in range(globals.num_agents)
                )
                for i in range(globals.num_agents)
            )
            f.write(header + '\n')
        
        # Write experiment results
        result = f'{globals.cod_experiment:10d} {ir_type:6d} '
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
        result += f' {aggr_pre_prices:12.7f} '
        result += ' '.join(f'{aggr_dev_shock_prices[i]:23.7f}' for i in range(num_shock_periods_print))
        result += f' {aggr_dev_post_prices:16.7f} '
        result += ' '.join(f'{aggr_non_dev_shock_prices[i]:26.7f}' for i in range(num_shock_periods_print))
        result += f' {aggr_non_dev_post_prices:19.7f} '
        result += f' {aggr_pre_prices_q:14.7f} '
        result += ' '.join(f'{aggr_dev_shock_prices_q[i]:25.7f}' for i in range(num_shock_periods_print))
        result += f' {aggr_dev_post_prices_q:18.7f} '
        result += ' '.join(f'{aggr_non_dev_shock_prices_q[i]:28.7f}' for i in range(num_shock_periods_print))
        result += f' {aggr_non_dev_post_prices_q:21.7f} '
        result += f' {aggr_pre_profits:13.7f} '
        result += ' '.join(f'{aggr_dev_shock_profits[i]:24.7f}' for i in range(num_shock_periods_print))
        result += f' {aggr_dev_post_profits:17.7f} '
        result += ' '.join(f'{aggr_non_dev_shock_profits[i]:27.7f}' for i in range(num_shock_periods_print))
        result += f' {aggr_non_dev_post_profits:20.7f} '
        result += f' {aggr_pre_profits_q:15.7f} '
        result += ' '.join(f'{aggr_dev_shock_profits_q[i]:26.7f}' for i in range(num_shock_periods_print))
        result += f' {aggr_dev_post_profits_q:19.7f} '
        result += ' '.join(f'{aggr_non_dev_shock_profits_q[i]:29.7f}' for i in range(num_shock_periods_print))
        result += f' {aggr_non_dev_post_profits_q:22.7f} '
        result += ' '.join(f'{freq_pre_length[i]:13d}' for i in range(num_thres_periods_length))
        result += ' '.join(
            ' '.join(f'{freq_post_length[i,j]:17d}' for j in range(num_thres_periods_length))
            for i in range(globals.num_agents)
        )
        result += ' '.join(
            ' '.join(f'{freq_shock_length[i,j]:18d}' for j in range(num_thres_periods_length))
            for i in range(globals.num_agents)
        )
        result += ' '.join(
            ' '.join(f'{freq_punishment_strategy[i,j]:18d}' for j in range(num_thres_periods_length0))
            for i in range(globals.num_agents)
        )
        result += ' '.join(
            ' '.join(
                f'{avg_pre_prices[j]:14.7f} ' +
                ' '.join(f'{avg_shock_prices[k,i,j]:25.7f}' for k in range(num_shock_periods_print)) +
                f'{avg_post_prices[i,j]:18.7f}'
                for j in range(globals.num_agents)
            )
            for i in range(globals.num_agents)
        )
        result += ' '.join(
            ' '.join(
                f'{avg_pre_prices_q[j]:16.7f} ' +
                ' '.join(f'{avg_shock_prices_q[k,i,j]:27.7f}' for k in range(num_shock_periods_print)) +
                f'{avg_post_prices_q[i,j]:20.7f}'
                for j in range(globals.num_agents)
            )
            for i in range(globals.num_agents)
        )
        result += ' '.join(
            ' '.join(
                f'{avg_pre_profits[j]:15.7f} ' +
                ' '.join(f'{avg_shock_profits[k,i,j]:26.7f}' for k in range(num_shock_periods_print)) +
                f'{avg_post_profits[i,j]:19.7f}'
                for j in range(globals.num_agents)
            )
            for i in range(globals.num_agents)
        )
        result += ' '.join(
            ' '.join(
                f'{avg_pre_profits_q[j]:17.7f} ' +
                ' '.join(f'{avg_shock_profits_q[k,i,j]:28.7f}' for k in range(num_shock_periods_print)) +
                f'{avg_post_profits_q[i,j]:21.7f}'
                for j in range(globals.num_agents)
            )
            for i in range(globals.num_agents)
        )
        f.write(result + '\n')

def compute_static_best_response(
    optimal_strategy: np.ndarray,
    i_state: int,
    i_agent: int,
    globals: GlobalVars
) -> Tuple[int, float]:
    """
    Computes static best response of i_agent given all agents' strategies
    'Best' means that the selected price maximizes i_agent's profits assuming
    that rivals play according to their strategies
    
    Parameters:
    - optimal_strategy: strategy for all agents
    - i_state: current state
    - i_agent: agent index
    
    Returns:
    - index_static_br: static BR price index
    - pi_static_br: i_agent's one-period profit when playing index_static_br
    """
    p_prime = optimal_strategy[i_state, :].copy()
    sel_profits = np.zeros(globals.num_prices)
    
    for i_price in range(globals.num_prices):
        p_prime[i_agent] = i_price
        sel_profits[i_price] = globals.pi[compute_action_number(p_prime, globals), i_agent]
    
    index_static_br = np.argmax(sel_profits)
    pi_static_br = np.max(sel_profits)
    
    return index_static_br, pi_static_br

def compute_dynamic_best_response(
    optimal_strategy: np.ndarray,
    i_state: int,
    i_agent: int,
    delta: float,
    globals: GlobalVars
) -> Tuple[int, float]:
    """
    Computes dynamic best response of one agent given all agents' strategies
    'Best' means that the selected price maximizes Q given the state and assuming
    that opponents play according to their strategies
    
    Parameters:
    - optimal_strategy: strategy for all agents
    - i_state: current state
    - i_agent: agent index
    - delta: discount factor
    
    Returns:
    - index_dynamic_br: dynamic BR price index
    - q_dynamic_br: Q(i_state, index_dynamic_br, i_agent)
    """
    sel_q = np.zeros(globals.num_prices)
    
    for i_price in range(globals.num_prices):
        sel_q[i_price], _, _ = compute_q_cell(
            optimal_strategy, i_state, i_price, i_agent, delta, globals
        )
    
    index_dynamic_br = np.argmax(sel_q)
    q_dynamic_br = np.max(sel_q)
    
    return index_dynamic_br, q_dynamic_br

def compute_individual_ir(
    optimal_strategy: np.ndarray,
    initial_state: int,
    dev_agent: int,
    dev_price: int,
    dev_length: int,
    dev_obs_length: int,
    pre_cycle_length: int,
    pre_cycle_states: np.ndarray,
    globals: GlobalVars
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, int, int, int
]:
    """
    Computes the Impulse Response for a price deviation on a single replication
    
    Parameters:
    - optimal_strategy: strategy for all agents
    - initial_state: initial state
    - dev_agent: deviating agent index
    - dev_price: deviation price index
    - dev_length: deviation period length
    - dev_obs_length: length of the observation interval of the deviation period
    - pre_cycle_length: length of the pre-deviation cycle
    - pre_cycle_states: pre-deviation cycle states
    
    Returns:
    - shock_states: trajectory of states in the deviation interval
    - shock_ind_prices: trajectory of all agents' price indexes in the deviation interval
    - shock_prices: trajectory of all agents' prices in the deviation interval
    - shock_profits: trajectory of all agents' profits in the deviation interval
    - avg_post_prices: average of all agents' prices in the post-deviation cycle
    - avg_post_profits: average of all agents' profits in the post-deviation cycle
    - shock_length: length of the non-cyclic deviation interval
    - punishment_strategy: indicator. After the deviation:
        = 0: the system returns to a cycle different from the pre-deviation cycle
        > 0: the system returns to the pre-deviation cycle after punishment_strategy periods
    - post_length: length of the post-deviation cycle
    """
    # Initialize arrays
    p = convert_number_base(initial_state - 1, globals.num_prices, globals.num_agents * globals.depth_state)
    p = p.reshape((globals.depth_state, globals.num_agents))
    p_prime = optimal_strategy[initial_state, :].copy()
    
    # Agent "dev_agent" selects the best deviation price,
    # the other agents stick to the strategy at convergence
    p_prime[dev_agent] = dev_price
    
    # Loop over deviation period
    visited_states = np.zeros(max(dev_obs_length, globals.num_periods), dtype=np.int32)
    shock_states = np.zeros(dev_obs_length, dtype=np.int32)
    shock_ind_prices = np.zeros((dev_obs_length, globals.num_agents), dtype=np.int32)
    shock_prices = np.zeros((dev_obs_length, globals.num_agents))
    shock_profits = np.zeros((dev_obs_length, globals.num_agents))
    flag_returned_to_state = False
    
    for i_period in range(max(dev_obs_length, globals.num_periods)):
        if globals.depth_state > 1:
            p[1:, :] = p[:-1, :]
        p[0, :] = p_prime
        visited_states[i_period] = compute_state_number(p, globals)
        
        for j_agent in range(globals.num_agents):
            if i_period < dev_obs_length:
                shock_states[i_period] = visited_states[i_period]
                shock_ind_prices[i_period, j_agent] = p_prime[j_agent]
                shock_prices[i_period, j_agent] = globals.prices_grids[p_prime[j_agent], j_agent]
                shock_profits[i_period, j_agent] = globals.pi[
                    compute_action_number(p_prime, globals), j_agent
                ]
        
        # Check if the state has already been visited
        # Case 1: the state returns to one of the states in the pre-shock cycle
        if (not flag_returned_to_state) and np.any(pre_cycle_states == visited_states[i_period]):
            shock_length = i_period + 1
            punishment_strategy = i_period + 1
            index_shock_state = p.flatten()
            flag_returned_to_state = True
        
        # Case 2: after some time, the state starts cycling among a new set of states
        if (i_period >= 1) and (not flag_returned_to_state) and np.any(
            visited_states[:i_period] == visited_states[i_period]
        ):
            shock_length = np.argmin(
                (visited_states[:i_period] - visited_states[i_period]) ** 2
            ) + 1
            punishment_strategy = 0
            index_shock_state = p.flatten()
            flag_returned_to_state = True
        
        # Update p_prime according to the deviation length
        p_prime = optimal_strategy[visited_states[i_period], :].copy()
        if dev_length == 1000:  # Permanent deviation
            p_prime[dev_agent] = dev_price
        elif dev_length > i_period:  # Temporary deviation
            p_prime[dev_agent] = dev_price
    
    # Post-shock period
    visited_states = np.zeros(globals.num_periods, dtype=np.int32)
    visited_prices = np.zeros((globals.num_periods, globals.num_agents))
    visited_profits = np.zeros((globals.num_periods, globals.num_agents))
    
    p = index_shock_state.reshape((globals.depth_state, globals.num_agents))
    p_prime = optimal_strategy[compute_state_number(p, globals), :].copy()
    
    for i_period in range(globals.num_periods):
        if globals.depth_state > 1:
            p[1:, :] = p[:-1, :]
        p[0, :] = p_prime
        visited_states[i_period] = compute_state_number(p, globals)
        
        for j_agent in range(globals.num_agents):
            visited_prices[i_period, j_agent] = globals.prices_grids[p_prime[j_agent], j_agent]
            visited_profits[i_period, j_agent] = globals.pi[
                compute_action_number(p_prime, globals), j_agent
            ]
        
        # Check if the state has already been visited
        if (i_period >= 1) and np.any(visited_states[:i_period] == visited_states[i_period]):
            break
        
        # Update p_prime and iterate
        p_prime = optimal_strategy[visited_states[i_period], :].copy()
    
    post_length = i_period - np.argmin(
        (visited_states[:i_period] - visited_states[i_period]) ** 2
    )
    
    avg_post_prices = np.mean(
        visited_prices[i_period - post_length + 1:i_period + 1, :],
        axis=0
    )
    avg_post_profits = np.mean(
        visited_profits[i_period - post_length + 1:i_period + 1, :],
        axis=0
    )
    
    return (
        shock_states, shock_ind_prices, shock_prices, shock_profits,
        avg_post_prices, avg_post_profits, shock_length, punishment_strategy, post_length
    ) 