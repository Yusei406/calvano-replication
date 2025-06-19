import numpy as np
from typing import List, Tuple, Dict, Optional

# Constants
NUM_SHOCK_PERIODS_PRINT = 10
NUM_THRES_CYCLE_LENGTH = 10

# Global variables
class GlobalVars:
    def __init__(self):
        # Number of agents
        self.n_agents = 2
        
        # Number of prices in the grid
        self.n_prices = 11
        
        # Depth of state
        self.state_depth = 1
        
        # Number of states
        self.n_states = self.n_prices ** (self.n_agents * self.state_depth)
        
        # Number of actions
        self.n_actions = self.n_prices
        
        # Experiment number
        self.experiment_number = 1
        
        # File names
        self.file_name_info_experiment = f'info_experiment_{self.experiment_number}.txt'
        
        # Random number generator
        self.rng = None
        
        # Simulation parameters
        self.n_iterations = 1000
        self.n_perfect_measurement = 100
        self.max_iterations = 100000
        
        # Experiment parameters
        self.num_experiments: int = 0
        self.tot_experiments: int = 0
        self.num_cores: int = 0
        self.num_sessions: int = 0
        self.iters_per_episode: int = 0
        self.max_num_episodes: int = 0
        self.max_iters: int = 0
        self.iters_in_perf_meas_period: int = 0
        self.print_q: int = 0
        self.cod_experiment: int = 0
        self.num_periods: int = 0
        self.num_exploration_parameters: int = 0
        
        # State and action parameters
        self.type_exploration_mechanism: int = 0
        self.depth_state0: int = 0
        self.depth_state: int = 0
        self.length_states: int = 0
        self.num_states: int = 0
        self.length_strategies: int = 0
        self.length_format_states_print: int = 0
        self.length_format_action_print: int = 0
        self.length_format_tot_experiments_print: int = 0
        self.length_format_num_sessions_print: int = 0
        
        # Game parameters
        self.type_payoff_input: int = 0
        self.num_agents: int = 0
        self.num_actions: int = 0
        self.num_demand_parameters: int = 0
        self.num_periods: int = 0
        self.num_exploration_parameters: int = 0
        
        # Switches
        self.switch_impulse_response_to_br: int = 0
        self.switch_impulse_response_to_nash: int = 0
        self.switch_impulse_response_to_all: int = 0
        self.switch_equilibrium_check: int = 0
        self.switch_q_gap_to_maximum: int = 0
        self.params_learning_trajectory: List[int] = [0, 0]
        self.switch_detailed_analysis: int = 0
        
        # Real parameters
        self.delta: float = 0.0
        self.perf_meas_period_length: float = 0.0
        self.mean_nash_profit: float = 0.0
        self.mean_coop_profit: float = 0.0
        self.gamma_singh_vives: float = 0.0
        
        # File names
        self.experiment_number: str = ""
        self.file_name_info_experiment: str = ""
        
        # Arrays
        self.converged: Optional[np.ndarray] = None
        self.index_strategies: Optional[np.ndarray] = None
        self.index_last_state: Optional[np.ndarray] = None
        self.cycle_length: Optional[np.ndarray] = None
        self.cycle_states: Optional[np.ndarray] = None
        self.cycle_prices: Optional[np.ndarray] = None
        self.index_actions: Optional[np.ndarray] = None
        self.c_states: Optional[np.ndarray] = None
        self.c_actions: Optional[np.ndarray] = None
        self.index_nash_prices: Optional[np.ndarray] = None
        self.index_coop_prices: Optional[np.ndarray] = None
        
        # Real arrays
        self.time_to_convergence: Optional[np.ndarray] = None
        self.cycle_profits: Optional[np.ndarray] = None
        self.nash_profits: Optional[np.ndarray] = None
        self.coop_profits: Optional[np.ndarray] = None
        self.max_val_q: Optional[np.ndarray] = None
        self.nash_prices: Optional[np.ndarray] = None
        self.coop_prices: Optional[np.ndarray] = None
        self.pi: Optional[np.ndarray] = None
        self.piq: Optional[np.ndarray] = None
        self.avg_pi: Optional[np.ndarray] = None
        self.avg_piq: Optional[np.ndarray] = None
        self.pg: Optional[np.ndarray] = None
        self.pgq: Optional[np.ndarray] = None
        self.avg_pg: Optional[np.ndarray] = None
        self.avg_pgq: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.discount_factors: Optional[np.ndarray] = None
        self.demand_parameters: Optional[np.ndarray] = None
        self.m_expl: Optional[np.ndarray] = None
        self.exploration_parameters: Optional[np.ndarray] = None
        self.nash_market_shares: Optional[np.ndarray] = None
        self.coop_market_shares: Optional[np.ndarray] = None
        self.prices_grids: Optional[np.ndarray] = None
        self.par_q_initialization: Optional[np.ndarray] = None
        
        # Character arrays
        self.label_states: Optional[List[str]] = None
        self.q_file_folder_name: Optional[List[str]] = None
        self.type_q_initialization: Optional[List[str]] = None

    def read_batch_variables(self, file_path: str) -> None:
        """Read input variables from batch file"""
        with open(file_path, 'r') as f:
            # Skip first line
            f.readline()
            
            # Read experiment parameters
            self.num_experiments, self.tot_experiments = map(int, f.readline().split())
            f.readline()
            self.num_cores = int(f.readline())
            f.readline()
            self.num_sessions = int(f.readline())
            f.readline()
            self.iters_per_episode = int(f.readline())
            f.readline()
            self.max_num_episodes = int(f.readline())
            f.readline()
            self.perf_meas_period_length = float(f.readline())
            f.readline()
            self.num_agents = int(f.readline())
            f.readline()
            self.depth_state0 = int(f.readline())
            self.depth_state = max(1, self.depth_state0)  # Accommodate DepthState = 0 case
            f.readline()
            self.num_prices = int(f.readline())
            
            # Calculate derived parameters
            self.length_format_tot_experiments_print = 1 + int(np.log10(self.tot_experiments))
            self.length_format_num_sessions_print = 1 + int(np.log10(self.num_sessions))
            self.max_iters = self.max_num_episodes * self.iters_per_episode
            self.iters_in_perf_meas_period = int(self.perf_meas_period_length * self.iters_per_episode)
            self.length_states = max(1, self.num_agents * self.depth_state0)
            self.length_format_states_print = self.length_states * (1 + int(np.log10(self.num_prices))) + self.length_states - 1
            self.num_states = self.num_prices ** (self.num_agents * self.depth_state0)
            self.num_periods = self.num_states + 1
            self.num_actions = self.num_prices ** self.num_agents
            self.length_strategies = self.num_agents * self.num_states
            self.length_format_action_print = int(np.log10(self.num_prices)) + 1
            
            # Read exploration mechanism type
            f.readline()
            self.type_exploration_mechanism = int(f.readline())
            self.num_exploration_parameters = self.num_agents
            
            # Read payoff input type
            f.readline()
            self.type_payoff_input = int(f.readline())
            if self.type_payoff_input == 1:
                self.num_demand_parameters = 1 + 2  # gamma, extend
            elif self.type_payoff_input == 2:
                self.num_demand_parameters = 2 * self.num_agents + 4  # a0, ai, ci, mu, extend
            elif self.type_payoff_input == 3:
                self.num_demand_parameters = 2 * self.num_agents + 4  # a0, ai, ci, mu = 0, extend
            
            # Read switches
            f.readline()
            self.switch_impulse_response_to_br = int(f.readline())
            f.readline()
            self.switch_impulse_response_to_nash = int(f.readline())
            f.readline()
            self.switch_impulse_response_to_all = int(f.readline())
            f.readline()
            self.switch_equilibrium_check = int(f.readline())
            f.readline()
            self.switch_q_gap_to_maximum = int(f.readline())
            f.readline()
            self.params_learning_trajectory = list(map(int, f.readline().split()))
            f.readline()
            self.switch_detailed_analysis = int(f.readline())
            
            # Allocate arrays
            self._allocate_arrays()
            
            # Initialize state and action indices
            self._initialize_indices()

    def _allocate_arrays(self) -> None:
        """Allocate numpy arrays for simulation"""
        self.converged = np.zeros(self.num_sessions, dtype=int)
        self.time_to_convergence = np.zeros(self.num_sessions)
        self.index_strategies = np.zeros((self.length_strategies, self.num_sessions), dtype=int)
        self.index_last_state = np.zeros((self.length_states, self.num_sessions), dtype=int)
        self.cycle_length = np.zeros(self.num_sessions, dtype=int)
        self.cycle_states = np.zeros((self.num_periods, self.num_sessions), dtype=int)
        self.cycle_prices = np.zeros((self.num_agents, self.num_periods, self.num_sessions))
        self.cycle_profits = np.zeros((self.num_agents, self.num_periods, self.num_sessions))
        self.index_actions = np.zeros((self.num_actions, self.num_agents), dtype=int)
        self.c_states = np.array([self.num_prices ** i for i in range(self.length_states-1, -1, -1)])
        self.c_actions = np.array([self.num_prices ** i for i in range(self.num_agents-1, -1, -1)])
        self.discount_factors = np.array([self.delta ** i for i in range(self.num_periods)])
        self.max_val_q = np.zeros((self.num_states, self.num_agents))
        self.demand_parameters = np.zeros(self.num_demand_parameters)
        self.exploration_parameters = np.zeros(self.num_exploration_parameters)
        self.m_expl = np.zeros(self.num_exploration_parameters)
        self.alpha = np.zeros(self.num_agents)
        self.nash_profits = np.zeros(self.num_agents)
        self.coop_profits = np.zeros(self.num_agents)
        self.pi = np.zeros((self.num_actions, self.num_agents))
        self.piq = np.zeros((self.num_actions, self.num_agents))
        self.avg_pi = np.zeros(self.num_actions)
        self.avg_piq = np.zeros(self.num_actions)
        self.pg = np.zeros((self.num_actions, self.num_agents))
        self.pgq = np.zeros((self.num_actions, self.num_agents))
        self.avg_pg = np.zeros(self.num_actions)
        self.avg_pgq = np.zeros(self.num_actions)
        self.index_nash_prices = np.zeros(self.num_agents, dtype=int)
        self.index_coop_prices = np.zeros(self.num_agents, dtype=int)
        self.nash_prices = np.zeros(self.num_agents)
        self.coop_prices = np.zeros(self.num_agents)
        self.type_q_initialization = [''] * self.num_agents
        self.par_q_initialization = np.zeros((self.num_agents, self.num_agents))
        self.nash_market_shares = np.zeros(self.num_agents)
        self.coop_market_shares = np.zeros(self.num_agents)
        self.prices_grids = np.zeros((self.num_prices, self.num_agents))
        
        # Character arrays
        self.label_states = [''] * self.num_states
        self.q_file_folder_name = [''] * self.num_agents

    def _initialize_indices(self) -> None:
        """Initialize state and action indices"""
        for i_action in range(self.num_actions):
            self.index_actions[i_action, :] = self.convert_number_base(i_action, self.num_prices, self.num_agents)

    @staticmethod
    def convert_number_base(n: int, b: int, l: int) -> np.ndarray:
        """Convert number n from base 10 to base b, generating a vector of length l"""
        result = np.zeros(l, dtype=int)
        tmp = n
        for i in range(l):
            result[l-i-1] = tmp % b + 1
            tmp = tmp // b
        return result

    def read_experiment_variables(self, file_path: str) -> None:
        """Read experiment-specific variables"""
        with open(file_path, 'r') as f:
            # Read experiment parameters
            line = f.readline().split()
            self.cod_experiment = int(line[0])
            self.print_q = int(line[1])
            self.alpha = np.array(list(map(float, line[2:2+self.num_agents])))
            self.m_expl = np.array(list(map(float, line[2+self.num_agents:2+2*self.num_agents])))
            self.delta = float(line[2+2*self.num_agents])
            self.demand_parameters = np.array(list(map(float, line[3+2*self.num_agents:3+2*self.num_agents+self.num_demand_parameters])))
            self.nash_prices = np.array(list(map(float, line[3+2*self.num_agents+self.num_demand_parameters:3+2*self.num_agents+self.num_demand_parameters+self.num_agents])))
            self.coop_prices = np.array(list(map(float, line[3+2*self.num_agents+self.num_demand_parameters+self.num_agents:3+2*self.num_agents+self.num_demand_parameters+2*self.num_agents])))
            
            # Read Q initialization parameters
            for i in range(self.num_agents):
                line = f.readline().split()
                self.type_q_initialization[i] = line[0]
                self.par_q_initialization[i, :] = np.array(list(map(float, line[1:1+self.num_agents])))
            
            # Set exploration parameters based on mechanism type
            if self.type_exploration_mechanism == 1:
                self.exploration_parameters = np.exp(-self.m_expl / self.iters_per_episode)
            elif self.type_exploration_mechanism == 2:
                self.exploration_parameters = 1.0 - 10.0 ** self.m_expl
            
            # Sanity check: with memoryless agents, only delta = 0 makes sense
            if self.depth_state0 == 0:
                self.delta = 0.0
            
            # Compute array of discount factors
            self.discount_factors = np.array([self.delta ** i for i in range(self.num_periods)]) 

# Create global instance
globals = GlobalVars() 