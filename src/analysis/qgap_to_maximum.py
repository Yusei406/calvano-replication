"""
Q-value gap analysis module for Phase 2.

Implements Q-value gap to maximum analysis as described in the paper.
Analyzes how far Q-values are from their theoretical maximum.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

# Handle imports for both package and standalone usage
try:
    from ..dtype_policy import DTYPE, array, zeros
    from ..params import SimParams
except ImportError:
    try:
        from dtype_policy import DTYPE, array, zeros
        from params import SimParams
    except ImportError:
        # Fallback for testing
        import numpy as np
        DTYPE = np.float64
        array = lambda x: np.array(x, dtype=DTYPE)
        zeros = lambda n: np.zeros(n, dtype=DTYPE)
        
        class SimParams:
            def __init__(self, config=None):
                self.n_agents = 2
                self.n_actions = 11
                self.lambda_param = 0.5
                self.a_param = 1.0
                self.demand_model = "logit"


@dataclass
class QGapStats:
    """Container for Q-gap analysis results."""
    mean_gap: float
    max_gap: float
    min_gap: float
    std_gap: float
    gap_by_action: np.ndarray
    gap_by_state_cluster: np.ndarray
    convergence_threshold: float = 1e-3


def compute_qgap(Q_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Q-value gap to maximum for each state-action pair.
    
    The Q-gap measures how far each Q-value is from the maximum Q-value
    in the same state, indicating suboptimality.
    
    Args:
        Q_matrix: Q-value matrix of shape (n_states, n_actions)
        
    Returns:
        Array of gaps for each state-action pair, same shape as Q_matrix
        
    Example:
        >>> Q = np.array([[1.0, 2.0, 1.5], [0.8, 1.2, 1.0]])
        >>> gaps = compute_qgap(Q)
        >>> # gaps[0] = [1.0, 0.0, 0.5] (max is 2.0 in first state)
        >>> # gaps[1] = [0.4, 0.0, 0.2] (max is 1.2 in second state)
    """
    if Q_matrix.ndim != 2:
        raise ValueError(f"Q_matrix must be 2D, got shape {Q_matrix.shape}")
    
    n_states, n_actions = Q_matrix.shape
    
    # Find maximum Q-value for each state
    max_q_per_state = np.max(Q_matrix, axis=1, keepdims=True)  # Shape: (n_states, 1)
    
    # Compute gap as max_Q - Q for each state-action pair
    qgap_matrix = max_q_per_state - Q_matrix
    
    # Ensure non-negative gaps (numerical precision)
    qgap_matrix = np.maximum(qgap_matrix, 0.0)
    
    return array(qgap_matrix)


def compute_qgap_statistics(Q_matrix: np.ndarray) -> QGapStats:
    """
    Compute comprehensive Q-gap statistics.
    
    Args:
        Q_matrix: Q-value matrix of shape (n_states, n_actions)
        
    Returns:
        QGapStats object with detailed statistics
    """
    qgap_matrix = compute_qgap(Q_matrix)
    
    # Overall statistics
    gaps_flat = qgap_matrix.flatten()
    mean_gap = float(np.mean(gaps_flat))
    max_gap = float(np.max(gaps_flat))
    min_gap = float(np.min(gaps_flat))
    std_gap = float(np.std(gaps_flat))
    
    # Action-wise statistics (average over states)
    gap_by_action = array(np.mean(qgap_matrix, axis=0))
    
    # State cluster statistics (group states by similarity)
    n_states, n_actions = Q_matrix.shape
    
    # Simple clustering: group states by their maximum Q-value
    max_q_values = np.max(Q_matrix, axis=1)
    
    # Create clusters based on quantiles of max Q-values
    n_clusters = min(5, n_states)  # Use up to 5 clusters
    if n_states > 1:
        cluster_boundaries = np.percentile(max_q_values, np.linspace(0, 100, n_clusters + 1))
        cluster_labels = np.digitize(max_q_values, cluster_boundaries) - 1
        cluster_labels = np.clip(cluster_labels, 0, n_clusters - 1)
    else:
        cluster_labels = np.array([0])
        n_clusters = 1
    
    # Compute average gap for each cluster
    gap_by_state_cluster = zeros(n_clusters)
    for cluster_id in range(n_clusters):
        states_in_cluster = cluster_labels == cluster_id
        if np.any(states_in_cluster):
            cluster_gaps = qgap_matrix[states_in_cluster]
            gap_by_state_cluster[cluster_id] = np.mean(cluster_gaps)
    
    return QGapStats(
        mean_gap=mean_gap,
        max_gap=max_gap,
        min_gap=min_gap,
        std_gap=std_gap,
        gap_by_action=gap_by_action,
        gap_by_state_cluster=gap_by_state_cluster
    )


def summary_qgap(Q_matrices: List[np.ndarray], agent_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Summarize Q-gap analysis across multiple agents/runs.
    
    Args:
        Q_matrices: List of Q-value matrices, one per agent or run
        agent_names: Optional names for agents/runs
        
    Returns:
        DataFrame with Q-gap summary statistics
    """
    if not Q_matrices:
        return pd.DataFrame()
    
    if agent_names is None:
        agent_names = [f"Agent_{i+1}" for i in range(len(Q_matrices))]
    
    summary_data = []
    
    for i, Q_matrix in enumerate(Q_matrices):
        if Q_matrix is None or Q_matrix.size == 0:
            continue
            
        try:
            stats = compute_qgap_statistics(Q_matrix)
            
            row = {
                'Agent': agent_names[i],
                'Mean_Gap': f"{stats.mean_gap:.6f}",
                'Max_Gap': f"{stats.max_gap:.6f}",
                'Min_Gap': f"{stats.min_gap:.6f}",
                'Std_Gap': f"{stats.std_gap:.6f}",
                'N_States': Q_matrix.shape[0],
                'N_Actions': Q_matrix.shape[1]
            }
            
            # Add action-wise gaps
            for j, gap in enumerate(stats.gap_by_action):
                row[f'Gap_Action_{j+1}'] = f"{gap:.6f}"
            
            # Add cluster-wise gaps
            for j, gap in enumerate(stats.gap_by_state_cluster):
                row[f'Gap_Cluster_{j+1}'] = f"{gap:.6f}"
            
            summary_data.append(row)
            
        except Exception as e:
            print(f"Warning: Failed to analyze Q-matrix for {agent_names[i]}: {e}")
            continue
    
    if summary_data:
        return pd.DataFrame(summary_data)
    else:
        return pd.DataFrame()


def analyze_qgap_convergence(Q_history: List[np.ndarray], threshold: float = 1e-3) -> Dict:
    """
    Analyze Q-gap convergence over time.
    
    Args:
        Q_history: List of Q-matrices over time
        threshold: Convergence threshold for gaps
        
    Returns:
        Dictionary with convergence analysis results
    """
    if not Q_history:
        return {}
    
    mean_gaps_over_time = []
    max_gaps_over_time = []
    
    for Q_matrix in Q_history:
        if Q_matrix is not None and Q_matrix.size > 0:
            gaps = compute_qgap(Q_matrix)
            mean_gaps_over_time.append(np.mean(gaps))
            max_gaps_over_time.append(np.max(gaps))
        else:
            mean_gaps_over_time.append(np.nan)
            max_gaps_over_time.append(np.nan)
    
    mean_gaps_over_time = np.array(mean_gaps_over_time)
    max_gaps_over_time = np.array(max_gaps_over_time)
    
    # Find convergence time
    convergence_time = None
    for t, (mean_gap, max_gap) in enumerate(zip(mean_gaps_over_time, max_gaps_over_time)):
        if not np.isnan(mean_gap) and mean_gap < threshold and max_gap < threshold * 10:
            convergence_time = t
            break
    
    # Calculate convergence rate (if converged)
    convergence_rate = None
    if convergence_time is not None and convergence_time > 10:
        # Fit exponential decay to the first part of convergence
        early_times = np.arange(min(convergence_time, 100))
        early_gaps = mean_gaps_over_time[early_times]
        
        # Remove NaN values
        valid_mask = ~np.isnan(early_gaps)
        if np.sum(valid_mask) > 5:
            valid_times = early_times[valid_mask]
            valid_gaps = early_gaps[valid_mask]
            
            # Fit log(gap) = a + b*t to estimate decay rate
            if np.all(valid_gaps > 0):
                log_gaps = np.log(valid_gaps)
                coeffs = np.polyfit(valid_times, log_gaps, 1)
                convergence_rate = -coeffs[0]  # Negative of slope
    
    return {
        'mean_gaps_over_time': mean_gaps_over_time,
        'max_gaps_over_time': max_gaps_over_time,
        'convergence_time': convergence_time,
        'convergence_rate': convergence_rate,
        'final_mean_gap': mean_gaps_over_time[-1] if len(mean_gaps_over_time) > 0 else np.nan,
        'final_max_gap': max_gaps_over_time[-1] if len(max_gaps_over_time) > 0 else np.nan
    } 