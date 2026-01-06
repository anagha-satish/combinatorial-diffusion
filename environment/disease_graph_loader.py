"""
Disease Graph Loader

Loads disease graphs from ICPSR data and creates testing environments.
"""

import os
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import networkx as nx

# Local imports
from environment.ICPSR_22140_processor import ICPSR22140Processor
from environment.log_junction_tree import LogJunctionTree
from environment.frontier_batch_env import BinaryFrontierEnvBatch
from utils.io_utils import load_pickle


def pick_random_cc_until_cross_threshold(
    inst_idx: int,
    G: nx.Graph,
    covariates: dict,
    statuses: dict,
    threshold: int,
    max_overshoot_pct: float = 0.1,
    min_infection_rate: float = 0.2,
    max_attempts: int = 100
) -> Tuple[nx.Graph, dict, dict]:
    """
    Sample connected components from graph until total nodes >= threshold.

    Args:
        inst_idx: Random seed for reproducible sampling
        G: NetworkX graph
        covariates: Node covariate dictionary
        statuses: Node infection status dictionary
        threshold: Minimum number of nodes to sample
        max_overshoot_pct: Maximum allowed overshoot as fraction of threshold (default 0.1 = 10%)
        min_infection_rate: Minimum required fraction of infected nodes (default 0.2 = 20%)
        max_attempts: Maximum retry attempts before falling back (default 100)

    Returns:
        Tuple of (subgraph, subgraph_covariates, subgraph_statuses)
    """
    all_cc_nodes = np.array(list(nx.connected_components(G)))
    best_attempt = None
    best_score = -float('inf')  # Track best attempt for fallback

    if len(all_cc_nodes) == 0:
        # Empty graph edge case
        return G, {}, {}

    for attempt in range(max_attempts):
        rng = np.random.default_rng(inst_idx + attempt)
        subgraph_nodes = set()
        subgraph_covariates = dict()
        subgraph_statuses = dict()
        idx_mapping = dict()

        # Shuffle CCs with different seed each attempt
        cc_order = all_cc_nodes.copy()
        rng.shuffle(cc_order)

        # Accumulate CCs until threshold crossed
        for cc_nodes in cc_order:
            subgraph_nodes.update(cc_nodes)
            for i in cc_nodes:
                idx_mapping[i] = len(idx_mapping)
                subgraph_covariates[idx_mapping[i]] = covariates[i]
                subgraph_statuses[idx_mapping[i]] = statuses[i]

            if len(subgraph_nodes) >= threshold:
                break

        # Build subgraph
        H = G.subgraph(subgraph_nodes)
        H = nx.relabel_nodes(H, idx_mapping)

        # Check constraints
        size = len(subgraph_nodes)
        total_infected = sum(subgraph_statuses.values())
        inf_rate = total_infected / size if size > 0 else 0.0

        max_size = threshold * (1 + max_overshoot_pct)
        size_ok = size <= max_size
        inf_ok = inf_rate >= min_infection_rate

        # Track best attempt (score = weighted combination of constraints)
        # Prioritize infection rate, then size constraint
        score = inf_rate - (max(0, size - max_size) / threshold) * 0.5
        if score > best_score:
            best_score = score
            best_attempt = (H, subgraph_covariates, subgraph_statuses, size, inf_rate)

        # Return if both constraints satisfied
        if size_ok and inf_ok:
            return H, subgraph_covariates, subgraph_statuses

    # Fallback: return best attempt with warning
    if best_attempt is None:
        raise RuntimeError(f"Failed to generate any valid subgraph after {max_attempts} attempts")

    H, subgraph_covariates, subgraph_statuses, size, inf_rate = best_attempt
    print(f"Warning: Could not meet all constraints after {max_attempts} attempts")
    print(f"  Threshold: {threshold}, Max size: {threshold * (1 + max_overshoot_pct):.0f}")
    print(f"  Best attempt: size={size} ({size/threshold:.1%} of threshold), infection_rate={inf_rate:.1%}")
    print(f"  Required: size <= {threshold * (1 + max_overshoot_pct):.0f}, infection_rate >= {min_infection_rate:.1%}")
    return H, subgraph_covariates, subgraph_statuses


def load_disease_graph_instance(
    std_name: str,
    cc_threshold: int,
    inst_idx: int,
    filter_sex_only: bool = False
) -> Tuple[nx.Graph, dict, np.ndarray, np.ndarray, dict]:
    """
    Load and process disease graph from ICPSR data.
    
    Args:
        std_name: Disease name ("HIV", "Gonorrhea", "Chlamydia", "Syphilis", "Hepatitis")
        cc_threshold: Minimum number of nodes in sampled connected components
        inst_idx: Instance index for reproducible sampling
        filter_sex_only: If True, only include sexual contact edges
        
    Returns:
        Tuple of (G, covariates, theta_unary, theta_pairwise, statuses)
    """
    # Path to ICPSR data (in network-disease-testing directory)
    base_path = Path(__file__).parent.parent
    data_path = base_path / "ICPSR_22140"
    
    # File paths
    tsv_file1 = str(data_path / "DS0001/22140-0001-Data.tsv")
    tsv_file2 = str(data_path / "DS0002/22140-0002-Data.tsv")
    tsv_file3 = str(data_path / "DS0003/22140-0003-Data.tsv")
    pickle_filename = str(base_path / "ICPSR_22140.pkl")
    
    print(f"Loading ICPSR data for {std_name}...")
    
    # Initialize processor
    processor = ICPSR22140Processor(
        tsv_file1, tsv_file2, tsv_file3,
        pickle_filename,
        filter_sex_only=filter_sex_only,
        multithread=False
    )
    
    # Load theta parameters from checkpoint
    print(f"Loading theta parameters for {std_name}...")
    checkpoint_path = base_path / f"ICPSR_22140/checkpoints/{std_name}.pkl"
    
    if checkpoint_path.exists():
        print(f"  ✓ Found existing checkpoint: {checkpoint_path}")
        checkpoint_dict = load_pickle(str(checkpoint_path))
        log_pl, step_idx, theta_unary, theta_pairwise, m_unary, v_unary, m_pairwise, v_pairwise = sorted(
            [key + value for key, value in checkpoint_dict.items()], 
            reverse=True
        )[0]
        print(f"  ✓ Loaded from checkpoint: iter {step_idx}, log_pl={log_pl:.4f}")
    else:
        print(f"  ⚠ No checkpoint found, fitting theta...")
        processor.fit_theta_parameters(std_name)
        theta_unary, theta_pairwise = processor.get_theta_parameters(std_name)
    
    # Get merged dataset
    full_covariates, full_statuses, full_graph, _, _, _ = processor.merged_datasets[std_name]
    
    # Sample connected components
    print(f"Sampling connected components (threshold={cc_threshold}, inst_idx={inst_idx})...")
    G, covariates, statuses = pick_random_cc_until_cross_threshold(
        inst_idx, full_graph, full_covariates, full_statuses, cc_threshold
    )
    
    print(f"✓ Loaded disease graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Infected: {sum(statuses.values())}/{len(statuses)}")
    print(f"  Covariate dim: {len(covariates[0])}")
    
    return G, covariates, theta_unary, theta_pairwise, statuses


def create_disease_env(
    G: nx.Graph,
    covariates: dict,
    theta_unary: np.ndarray,
    theta_pairwise: np.ndarray,
    budget: int,
    discount_factor: float = 0.99,
    eps: float = 0.0,
    eps_rng_seed: int = 42,
    rng_seed: int = 314159
) -> BinaryFrontierEnvBatch:
    """
    Create disease testing environment with fitted parameters.
    
    Args:
        G: NetworkX graph
        covariates: Node covariate dictionary
        theta_unary: Fitted unary parameters
        theta_pairwise: Fitted pairwise parameters
        budget: Testing budget per round
        discount_factor: Discount factor for future rewards
        eps: Noise level for theta perturbation
        eps_rng_seed: Random seed for noise
        rng_seed: Random seed for environment
        
    Returns:
        Configured BinaryFrontierEnvBatch environment
    """
    print(f"Creating disease environment (n={G.number_of_nodes()}, budget={budget})...")
    
    # Create LogJunctionTree for inference
    args = {
        'G': G,
        'covariates': covariates,
        'theta_unary': theta_unary,
        'theta_pairwise': theta_pairwise,
        'eps': eps,
        'eps_rng_seed': eps_rng_seed
    }
    
    variables = [f"X{idx}" for idx in G.nodes()]
    P = LogJunctionTree(variables, args)
    
    # Create environment
    env = BinaryFrontierEnvBatch(
        G=G,
        P=P,
        discount_factor=discount_factor,
        cc_dict=None,
        cc_root=None,
        rng_seed=rng_seed,
        budget=budget
    )
    
    print(f"✓ Environment created: {env.n} nodes, budget={env.budget}")
    print(f"  Frontier roots: {len(env.cc_root)} connected components")
    
    return env