# Standard library imports
import copy
import os
import shutil
import tempfile
from pathlib import Path

from multiprocessing import Pool

# Third-party imports
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports
from environment.abstract_joint_probability_class import AbstractJointProbabilityClass
from utils.io_utils import load_pickle, save_pickle

"""
Class for processing the ICPSR_22140 dataset
"""
class ICPSR22140Processor:
    def __init__(self, tsv_file1: str, tsv_file2: str, tsv_file3: str, pickle_filename: str, filter_sex_only: bool, multithread: bool = True) -> None:
        self.pickle_filename = pickle_filename
        self.STD_to_dfkey = dict()
        self.STD_to_dfkey["Gonorrhea"] = "GONO"
        self.STD_to_dfkey["Chlamydia"] = "CHLAM"
        self.STD_to_dfkey["Syphilis"] = "SYPH"
        self.STD_to_dfkey["HIV"] = "HIV"
        self.STD_to_dfkey["Hepatitis"] = "HBV"
        self.covariate_headers = ['LOCAL', 'RACE', 'ETHN', 'SEX', 'ORIENT', 'BEHAV', 'PRO', 'PIMP', 'JOHN', 'DEALER', 'DRUGMAN', 'THIEF', 'RETIRED', 'HWIFE', 'DISABLE', 'UNEMP', 'STREETS']
        self.multithread = multithread
        self.curated_dataset = self._extract_curated_dataset(tsv_file1, tsv_file2, tsv_file3, filter_sex_only)
        self.merged_datasets = self._merge_all_std_datasets_into_one()


        # Print dataset statistics
        for std in self.STD_to_dfkey.keys():
            covariates, statuses, G, _, _, _ = self.merged_datasets[std]
            n = G.number_of_nodes()
            m = G.number_of_edges()
            num_positive = sum([1 for u in G.nodes if statuses[u] == 1])
            diam = max([
                nx.diameter(G.subgraph(component))
                for component in nx.connected_components(G)
            ])
            tw, _ = nx.algorithms.approximation.treewidth_min_fill_in(G)
            print(f"Disease {std}: {num_positive}/{n} infected. {m} edges. diameter = {diam}. approx treewidth {tw}. covariate length: {len(covariates[0])}")

    """
    Generate covariates
    """
    def _generate_covariates(self, node_df: pd.DataFrame, rid: int, studynum: int) -> np.ndarray:
        covariates = []
        mask = (node_df["RID"] == rid) & (node_df["STUDYNUM"] == studynum)
        assert len(node_df.loc[mask, self.covariate_headers]) == 1
        for col in self.covariate_headers:
            values = sorted([int(x) for x in set(node_df[col].values)])
            one_hot = [0] * len(values)
            idx = values.index(node_df.loc[mask, col].iloc[0])
            one_hot[idx] = 1
            covariates += one_hot
        return np.array(covariates)

    """
    Extract dataset from ICPSR_22140
    - curated_dataset[std] is a list of datasets with keys {"Gonorrhea", "Chlamydia", "Syphilis", "HIV", "Hepatitis"}
    - Each dataset is a dictionary with keys {"studynum", "graph", "covariates", "statuses"}
    """
    def _extract_curated_dataset(self, tsv_file1: str, tsv_file2: str, tsv_file3: str, filter_sex_only: bool) -> dict:
        if not os.path.isfile(self.pickle_filename):
            node_df = pd.read_csv(tsv_file1, sep='\t', dtype=str)
            df2 = pd.read_csv(tsv_file2, sep='\t', dtype=str)
            df3 = pd.read_csv(tsv_file3, sep='\t', dtype=str)

            # Reorder df3 columns to match df2    
            assert set(df2.columns) == set(df3.columns)
            df3 = df3[df2.columns]

            # Stack rows of both edge files and reset index
            assert df2.columns.equals(df3.columns)
            edge_df = pd.concat([df2, df3], ignore_index=True)

            # Convert columns of interest to integers or NaN
            columns_to_convert = (
                self.covariate_headers
                + ["STUDYNUM", "RID", "ID1", "ID2", "TIETYPE"]
                + [f"{dfkey}1" for dfkey in self.STD_to_dfkey.values()]
                + [f"{dfkey}2" for dfkey in self.STD_to_dfkey.values()]
            )
            for col in columns_to_convert:
                if col in node_df.columns:
                    node_df[col] = pd.to_numeric(node_df[col], errors='coerce').astype('Int64')
                if col in edge_df.columns:
                    edge_df[col] = pd.to_numeric(edge_df[col], errors='coerce').astype('Int64')

            # Filter and keep only sex edges
            if filter_sex_only:
                sex_filter = edge_df[edge_df["TIETYPE"] == 3]
            else:
                sex_filter = edge_df

            curated_dataset = {std: [] for std in self.STD_to_dfkey.keys()}
            for std, dfkey in self.STD_to_dfkey.items():
                std_sex_filter = sex_filter[sex_filter[f"{dfkey}1"].isin({0,1}) & sex_filter[f"{dfkey}2"].isin({0,1})]
                graphs = {studynum: nx.Graph() for studynum in set(node_df["STUDYNUM"])}
                digraphs = {studynum: nx.DiGraph() for studynum in set(node_df["STUDYNUM"])}
                statuses = {studynum: dict() for studynum in set(node_df["STUDYNUM"])}
                for _, row in std_sex_filter.iterrows():
                    studynum, u, v, u_status, v_status = row["STUDYNUM"], row["ID1"], row["ID2"], row[f"{dfkey}1"], row[f"{dfkey}2"]
                    if u not in statuses[studynum].keys():
                        statuses[studynum][u] = u_status
                    else:
                        statuses[studynum][u] = max(u_status, statuses[studynum][u])
                    if v not in statuses[studynum].keys():
                        statuses[studynum][v] = v_status
                    else:
                        statuses[studynum][v] = max(v_status, statuses[studynum][v])
                    graphs[studynum].add_edge(u, v)
                    digraphs[studynum].add_edge(u, v)
                for studynum in set(node_df["STUDYNUM"]):
                    G = graphs[studynum]
                    DG = digraphs[studynum]
                    assert G.number_of_nodes() == DG.number_of_nodes()
                    if G.number_of_nodes() > 0:
                        # Create new dataset and store into curated dataset
                        new_dataset = dict()
                        individual_mapping = dict()
                        individual_covariates = dict()
                        individual_statuses = dict()
                        for u in G.nodes:
                            individual_mapping[u] = len(individual_mapping)
                            individual_covariates[individual_mapping[u]] = self._generate_covariates(node_df, u, studynum)
                            individual_statuses[individual_mapping[u]] = statuses[studynum][u]
                        G = nx.relabel_nodes(G, individual_mapping)
                        DG = nx.relabel_nodes(DG, individual_mapping)

                        # Compute roots for graph and digraph
                        G_roots = []
                        for u in G.nodes:
                            u_is_root = True
                            for v in G.neighbors(u):
                                if (v, u) in DG.edges:
                                    u_is_root = False
                                    break
                            if u_is_root:
                                G_roots.append(u)
                        DG_roots = [node for node in DG.nodes if DG.in_degree(node) == 0]

                        # Store dataset
                        new_dataset["studynum"] = studynum
                        new_dataset["covariates"] = individual_covariates
                        new_dataset["statuses"] = individual_statuses
                        new_dataset["graph"] = G
                        new_dataset["digraph"] = DG
                        new_dataset["graph_roots"] = G_roots
                        new_dataset["digraph_roots"] = DG_roots
                        curated_dataset[std].append(new_dataset)
            
            # Store curated_dataset to file
            save_pickle(curated_dataset, self.pickle_filename)
        
        # Load curated_dataset from file and output
        curated_dataset = load_pickle(self.pickle_filename)
        return curated_dataset

    def _merge_all_std_datasets_into_one(self) -> dict:
        merged_datasets = dict()
        for std in self.STD_to_dfkey.keys():
            sz = 0
            overall_covariates = dict()
            overall_statuses = dict()
            overall_graph = nx.Graph()
            overall_digraph = nx.DiGraph()
            overall_graph_roots = []
            overall_digraph_roots = []
            for std_dataset in self.curated_dataset[std]:
                covariates, statuses, G, DG, G_roots, DG_roots = (
                    std_dataset['covariates'],
                    std_dataset['statuses'],
                    std_dataset['graph'],
                    std_dataset['digraph'],
                    std_dataset['graph_roots'],
                    std_dataset['digraph_roots']
                )
                n = G.number_of_nodes()
                assert DG.number_of_nodes() == n
                for idx in range(n):
                    overall_covariates[idx + sz] = covariates[idx]
                    overall_statuses[idx + sz] = statuses[idx]
                overall_graph.add_nodes_from([idx + sz for idx in G.nodes])
                overall_digraph.add_nodes_from([idx + sz for idx in DG.nodes])
                for u, v in G.edges():
                    overall_graph.add_edge(u + sz, v + sz)
                for u, v in DG.edges():
                    overall_digraph.add_edge(u + sz, v + sz)
                overall_graph_roots += [u + sz for u in G_roots]
                overall_digraph_roots += [u + sz for u in DG_roots]
                sz += n
            merged_datasets[std] = [
                overall_covariates,
                overall_statuses,
                overall_graph,
                overall_digraph,
                np.array(overall_graph_roots),
                np.array(overall_digraph_roots)
            ]
        return merged_datasets

    def _load_from_checkpoint(self, checkpoint_filename: str) -> tuple[dict, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert os.path.exists(checkpoint_filename)
        checkpoint_dict = load_pickle(checkpoint_filename)
        log_pl, step_idx, theta_unary, theta_pairwise, m_unary, v_unary, m_pairwise, v_pairwise = sorted([key + value for key, value in checkpoint_dict.items()], reverse=True)[0]
        print(f"Just loaded from {checkpoint_filename}. Checkpoint iter {step_idx}, log_pseudolikelihood = {log_pl:.4f}")
        return checkpoint_dict, step_idx, theta_unary, theta_pairwise, m_unary, v_unary, m_pairwise, v_pairwise

    def get_dataset_for_fitting_theta(self, dataset_key: str) -> tuple[dict, dict, nx.Graph, nx.DiGraph, np.ndarray, np.ndarray]:
        return self.merged_datasets[dataset_key]

    def get_theta_parameters(self, dataset_key: str) -> tuple[np.ndarray, np.ndarray]:
        base_path = Path(__file__).parent.parent
        checkpoint_filename = str(base_path / f"ICPSR_22140/checkpoints/{dataset_key}.pkl")
        if not os.path.exists(checkpoint_filename):
            self.fit_theta_parameters(dataset_key)
        _, _, theta_unary, theta_pairwise, _, _, _, _ = self._load_from_checkpoint(checkpoint_filename)
        return theta_unary, theta_pairwise

    """
    Compute pseudo MLE estimates of theta via gradient ascent while storing the checkpoints periodically
    checkpoint_dict[(log_pseudolikelihood, step_idx)] = (theta_unary, theta_pairwise, m_unary, v_unary, m_pairwise, v_pairwise)
    """
    def fit_theta_parameters(self, std_name: str) -> None:
        base_path = Path(__file__).parent.parent
        checkpoint_filename = str(base_path / f"ICPSR_22140/checkpoints/{std_name}.pkl")
        os.makedirs(os.path.dirname(checkpoint_filename), exist_ok=True)

        covariates, statuses, G, _, _, _ = self.get_dataset_for_fitting_theta(std_name)
        n = G.number_of_nodes()
        covariate_length = len(covariates[0])

        # Pre-compute unary and pairwise sum vectors for gradient computation later
        self.memo_unary_vector_sum_for_gradient = dict()
        self.memo_pairwise_vector_sum_for_gradient = dict()
        for i in range(n):
            c_i = covariates[i]
            self.memo_unary_vector_sum_for_gradient[i] = AbstractJointProbabilityClass.f_unary(1, c_i) - AbstractJointProbabilityClass.f_unary(0, c_i)
            pairwise_vector_sum = np.zeros(AbstractJointProbabilityClass.compute_theta_length(covariate_length, 2))
            for j in G.neighbors(i):
                u, v = min(i,j), max(i,j)
                c_u = covariates[u]
                c_v = covariates[v]
                if u == i:
                    x_v = statuses[v]
                    pairwise_vector_sum += AbstractJointProbabilityClass.f_pairwise(1, x_v, c_u, c_v) - AbstractJointProbabilityClass.f_pairwise(0, x_v, c_u, c_v)
                else:
                    x_u = statuses[u]
                    pairwise_vector_sum += AbstractJointProbabilityClass.f_pairwise(x_u, 1, c_u, c_v) - AbstractJointProbabilityClass.f_pairwise(x_u, 0, c_u, c_v)
            self.memo_pairwise_vector_sum_for_gradient[i] = pairwise_vector_sum

        if os.path.exists(checkpoint_filename):
            # Load theta checkpoint from file and continue fitting
            checkpoint_dict, step_idx, theta_unary, theta_pairwise, m_unary, v_unary, m_pairwise, v_pairwise = self._load_from_checkpoint(checkpoint_filename)
        else:
            # Initialize to small random numbers
            checkpoint_dict = dict()
            step_idx = 1
            theta_unary = np.random.randn(AbstractJointProbabilityClass.compute_theta_length(covariate_length, 1)) * 0.01
            theta_pairwise = np.random.randn(AbstractJointProbabilityClass.compute_theta_length(covariate_length, 2)) * 0.01
            m_unary = np.zeros_like(theta_unary)
            v_unary = np.zeros_like(theta_unary)
            m_pairwise = np.zeros_like(theta_pairwise)
            v_pairwise = np.zeros_like(theta_pairwise)

        # Adam hyperparameters from ChatGPT
        lr = 1e-2
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        max_iter = 500

        # Run until max iter or when both gradients are near zero
        if step_idx < max_iter:
            for it in tqdm(range(step_idx, max_iter+1), desc=f"{checkpoint_filename}", leave=False):
                self.memo_local_log_ZProb = dict()
                grad_unary, grad_pairwise = self._compute_gradients(std_name, theta_unary, theta_pairwise)

                # Adam update for theta_unary
                m_unary = beta1 * m_unary + (1 - beta1) * grad_unary
                v_unary = beta2 * v_unary + (1 - beta2) * (grad_unary ** 2)
                m_hat_unary = m_unary / (1 - beta1 ** step_idx)
                v_hat_unary = v_unary / (1 - beta2 ** step_idx)
                theta_unary += lr * m_hat_unary / (np.sqrt(v_hat_unary) + eps)

                # Adam update for theta_pairwise
                m_pairwise = beta1 * m_pairwise + (1 - beta1) * grad_pairwise
                v_pairwise = beta2 * v_pairwise + (1 - beta2) * (grad_pairwise ** 2)
                m_hat_pairwise = m_pairwise / (1 - beta1 ** step_idx)
                v_hat_pairwise = v_pairwise / (1 - beta2 ** step_idx)
                theta_pairwise += lr * m_hat_pairwise / (np.sqrt(v_hat_pairwise) + eps)

                # if np.linalg.norm(unary_gradient) < eps and np.linalg.norm(pairwise_gradient) < eps:
                if np.linalg.norm(grad_unary) < eps and np.linalg.norm(grad_pairwise) < eps:
                    break

                if ((it > 0 and it % 10 == 0) or it == max_iter):
                    self.memo_local_log_ZProb = dict()
                    log_pl = self._compute_log_pseudo_likelihood(std_name, theta_unary, theta_pairwise)
                    norm1 = np.linalg.norm(grad_unary)
                    norm2 = np.linalg.norm(grad_pairwise)
                    print(f"{checkpoint_filename} | Iter {it} | log_pseudolikelihood = {log_pl:.4f}, ||unary grad|| = {norm1:.4f}, ||pairwise grad|| = {norm2:.4f}")
                    
                    # Add checkpoint and store to file atomically (i.e. other threads will not read a half-written file)
                    checkpoint_dict[(log_pl, it)] = (theta_unary.copy(), theta_pairwise.copy(), m_unary.copy(), v_unary.copy(), m_pairwise.copy(), v_pairwise.copy())
                    tmp_fd, tmp_path = tempfile.mkstemp()
                    os.close(tmp_fd) # Close the file descriptor returned by mkstemp
                    save_pickle(checkpoint_dict, tmp_path)
                    shutil.move(tmp_path, checkpoint_filename) # Atomically replace the old checkpoint file

    """
    Single-thread task for computing gradients for thetas
    """
    def _compute_gradient_for_index(self, args: tuple) -> tuple[np.ndarray, np.ndarray]:
        dataset_key, theta_unary, theta_pairwise, i, status_i = args

        # We want to avoid computing the value of the partition function Z
        # coeff = x_i - p1 = x_i - Zp1/(Zp0 + Zp1)
        # Taking exponentiation np.exp(logZp0) and np.exp(logZp1) can cause numerical issues, so we use the log-sum-exp trick
        logZp0 = self._compute_local_log_ZProb(dataset_key, theta_unary, theta_pairwise, i, 0)
        logZp1 = self._compute_local_log_ZProb(dataset_key, theta_unary, theta_pairwise, i, 1)
        coeff = status_i - np.exp(logZp1 - AbstractJointProbabilityClass.logsumexp(np.array([logZp0, logZp1])))
        unary_update = coeff * self.memo_unary_vector_sum_for_gradient[i]
        pairwise_update = coeff * self.memo_pairwise_vector_sum_for_gradient[i]
        return unary_update, pairwise_update

    """
    Compute gradients for thetas
    """
    def _compute_gradients(self, dataset_key: str, theta_unary: np.ndarray, theta_pairwise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, statuses, _, _, _, _ = self.get_dataset_for_fitting_theta(dataset_key)
        unary_gradient = np.zeros_like(theta_unary)
        pairwise_gradient = np.zeros_like(theta_pairwise)
        all_args = [(dataset_key, theta_unary, theta_pairwise, i, statuses[i]) for i in range(len(statuses))]
        if self.multithread:
            # Multithread version: may not solve in order
            with Pool() as pool:
                for unary_update, pairwise_update in tqdm(
                    pool.imap_unordered(self._compute_gradient_for_index, all_args),
                    total=len(all_args),
                    desc="Computing gradients for thetas"
                ):
                    unary_gradient += unary_update
                    pairwise_gradient += pairwise_update
        else:
            # Single thread version
            for args in tqdm(all_args, desc=f"Solving (single thread)"):
                unary_update, pairwise_update = self._compute_gradient_for_index(args)
                unary_gradient += unary_update
                pairwise_gradient += pairwise_update

        return unary_gradient, pairwise_gradient
    
    """
    Single-thread task for computing log pseudo-likelihood
    """
    def _compute_log_pseudo_likelihood_for_index(self, args: tuple) -> tuple[int, float]:
        dataset_key, theta_unary, theta_pairwise, i = args
        terms = [
            self._compute_local_log_ZProb(dataset_key, theta_unary, theta_pairwise, i, b)
            for b in [0, 1]
        ]
        value = max(terms) + np.log(1 + np.exp(min(terms) - max(terms)))
        return i, value

    """
    Compute log pseudo-likelihood
    """
    def _compute_log_pseudo_likelihood(self, dataset_key: str, theta_unary: np.ndarray, theta_pairwise: np.ndarray) -> float:
        _, _, G, _, _, _ = self.get_dataset_for_fitting_theta(dataset_key)
        n = G.number_of_nodes()

        # Compute each denominator term for pseudo-log-likelihood using
        # log( exp(a) + exp(b) ) = max(a,b) + log(1 + exp(min(a,b) - max(a,b))),
        # where each term is of the form Z * Pr(X_i = b, X_{-i} = x_{-i} ; theta)
        log_denominators = [0.0] * n
        all_args = [(dataset_key, theta_unary, theta_pairwise, i) for i in range(n)]
        if self.multithread:
            # Multithread version: may not solve in order
            with Pool() as pool:
                for idx, log_denom_value in tqdm(
                    pool.imap_unordered(self._compute_log_pseudo_likelihood_for_index, all_args),
                    total=len(all_args),
                    desc="Computing log pseudo-likelihood"
                ):
                    log_denominators[idx] = log_denom_value
        else:
            # Single thread version
            for args in tqdm(all_args, desc=f"Solving (single thread)"):
                idx, log_denom_value = self._compute_log_pseudo_likelihood_for_index(args)
                log_denominators[idx] = log_denom_value

        # Compute the common numerator term log(Z * Pr(X = x ; theta))
        log_numerator = self._compute_log_ZProb(dataset_key, theta_unary, theta_pairwise)
        log_pseudo_likelihood = n * log_numerator - sum(log_denominators)
        assert log_pseudo_likelihood <= 0
        return log_pseudo_likelihood

    """
    Computes log( Z * Pr(X = x ; theta) )
    = log( exp(sum_i <unary> + sum_{i,j} <pairwise>)
    = sum_i <unary> + sum_{i,j} <pairwise>
    """
    def _compute_log_ZProb(self, dataset_key: str, theta_unary: np.ndarray, theta_pairwise: np.ndarray) -> float:
        # Arbitrarily use i = 0 and b = statuses[i]. Any index i works.
        _, statuses, _, _, _, _ = self.get_dataset_for_fitting_theta(dataset_key)
        return self._compute_local_log_ZProb(dataset_key, theta_unary, theta_pairwise, 0, statuses[0])

    """
    Computes log( Z * Pr(X_i = b, X_{-i} = x_{-i} ; theta) )
    = log( exp(sum_i <unary> + sum_{i,j} <pairwise>)
    = sum_i <unary> + sum_{i,j} <pairwise>
    """
    def _compute_local_log_ZProb(self, dataset_key: str, theta_unary: np.ndarray, theta_pairwise: np.ndarray, i: int, b: int) -> float:
        covariates, statuses, G, _, _, _ = self.get_dataset_for_fitting_theta(dataset_key)
        n = G.number_of_nodes()
        assert 0 <= i and i <= n
        assert b == 0 or b == 1

        key = (i, b)
        if key not in self.memo_local_log_ZProb.keys():
            val = 0.0

            # Compute contributions from theta_unary
            unary_vectors = np.stack([
                AbstractJointProbabilityClass.f_unary(b, covariates[i])
                if y == i
                else AbstractJointProbabilityClass.f_unary(statuses[y], covariates[y])
                for y in range(n)
            ]).astype(np.float64)
            val += np.sum(theta_unary @ unary_vectors.T)

            # Compute contributions from theta_pairwise
            pairwise_vectors1 = np.stack([
                AbstractJointProbabilityClass.f_pairwise(statuses[y], statuses[j], covariates[y], covariates[j])
                for y in range(n)
                for j in G.neighbors(y)
                if y != i and j != i and j > y
            ]).astype(np.float64)
            val += np.sum(theta_pairwise @ pairwise_vectors1.T)
            if next(G.neighbors(i), None) is not None:
                pairwise_vectors2 = np.stack([
                    AbstractJointProbabilityClass.f_pairwise(b, statuses[j], covariates[i], covariates[j])
                    if i < j
                    else AbstractJointProbabilityClass.f_pairwise(statuses[j], b, covariates[j], covariates[i])
                    for j in G.neighbors(i)
                ])
                val += np.sum(theta_pairwise @ pairwise_vectors2.T)
            
            # Store
            self.memo_local_log_ZProb[key] = val
        return self.memo_local_log_ZProb[key]