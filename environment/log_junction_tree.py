# Standard library imports
import itertools

# Third-party imports
import numpy as np
from scipy.special import logsumexp

# Local imports
from environment.abstract_joint_probability_class import AbstractJointProbabilityClass
from environment.log_factor import LogFactor

class LogJunctionTree(AbstractJointProbabilityClass):
    def __init__(self, variables: list[str], args: dict) -> None:
        super().__init__(variables, args)

    def _setup(self) -> None:
        G = self.args['G']
        covariates = self.args['covariates']
        theta_unary = self.args['theta_unary']
        theta_pairwise = self.args['theta_pairwise']
        
        assert G.number_of_nodes() == self.n
        assert len(covariates) == self.n
        covariate_length = len(covariates[0])
        assert len(theta_unary) == self.compute_theta_length(covariate_length, 1)
        assert len(theta_pairwise) == self.compute_theta_length(covariate_length, 2)

        # Noise scale is proportional to the magnitude of each coordinate, i.e., noise is coordinate wise [-eps * |theta|, eps * |theta|]
        eps = self.args.get('eps', 0)
        rng_seed = self.args.get('eps_rng_seed', 42)
        rng = np.random.default_rng(rng_seed)
        theta_unary = self.args['theta_unary'] + eps * (2 * rng.random(len(theta_unary)) - 1) * np.abs(self.args['theta_unary'])
        theta_pairwise = self.args['theta_pairwise'] + eps * (2 * rng.random(len(theta_pairwise)) - 1) * np.abs(self.args['theta_pairwise'])

        # unary_factors:   { var: 1D array of size domains[var] }
        # pairwise_factors:{ (u,v): 2D array of shape (dom[u], dom[v]) }
        X = [f"X{i}" for i in range(self.n)]
        self.domains = {v: 2 for v in self.variables}
        self.factors = []

        # Add unary factors
        for i in range(self.n):
            c_i = covariates[i]
            log_phi_0 = theta_unary @ self.f_unary(0, c_i)
            log_phi_1 = theta_unary @ self.f_unary(1, c_i)
            self.unary_factors[X[i]] = np.array([log_phi_0, log_phi_1])
            self.factors.append(LogFactor([f"X{i}"], self.unary_factors[X[i]], is_log=True))

        # Add pairwise factors
        for i in range(self.n):
            c_i = covariates[i]
            for j in G.neighbors(i):
                if i < j:
                    c_j = covariates[j]
                    log_phi_00 = theta_pairwise @ self.f_pairwise(0, 0, c_i, c_j)
                    log_phi_01 = theta_pairwise @ self.f_pairwise(0, 1, c_i, c_j)
                    log_phi_10 = theta_pairwise @ self.f_pairwise(1, 0, c_i, c_j)
                    log_phi_11 = theta_pairwise @ self.f_pairwise(1, 1, c_i, c_j)
                    key = frozenset([X[i], X[j]])
                    self.pairwise_factors[key] = np.array(
                        [log_phi_00, log_phi_01, log_phi_10, log_phi_11]
                    ).reshape(self.domains[f"X{i}"], self.domains[f"X{j}"])
                    self.factors.append(LogFactor([f"X{i}", f"X{j}"], self.pairwise_factors[key], is_log=True))
        
        # Precompute elimination order by min-fill
        # Build adjacency among variables
        adj = {v:set() for v in self.variables}
        for (u, v) in self.pairwise_factors:
            adj[u].add(v)
            adj[v].add(u)
            
        # Min-fill heuristic
        elim_order = []
        adj_copy = {v: set(nbrs) for v, nbrs in adj.items()}
        while adj_copy:
            # Pick v minimizing fill-in edges
            def fillin_cost(x):
                nbrs = adj_copy[x]
                return sum(1 for a,b in itertools.combinations(nbrs,2) if b not in adj_copy[a])
            v = min(adj_copy, key=lambda x: (fillin_cost(x), len(adj_copy[x])))
            elim_order.append(v)
            nbrs = adj_copy[v]
            
            # Fill in clique among neighbors
            for a, b in itertools.combinations(nbrs, 2):
                adj_copy[a].add(b)
                adj_copy[b].add(a)
            
            # Remove v
            for nbr in nbrs:
                adj_copy[nbr].remove(v)
            del adj_copy[v]
        self.elim_order = elim_order
        
    def compute_conditional_probability(self, query_dict: dict[str,int], evidence_dict: dict[str,int]) -> float:
        query_vars = list(query_dict.keys())
        query_vals = [query_dict[v] for v in query_vars]
        evidence_vars = list(evidence_dict.keys())
        evidence_vals = [evidence_dict[v] for v in evidence_vars]
        output = self.query(query_vars, query_vals, evidence_vars, evidence_vals)
        return output

    def query(self, query_vars, query_vals, evidence_vars, evidence_vals):
        """
        Return P(query_vars = query_vals | evidence_vars = evidence_vals).
        - query_vars, evidence_vars: lists of variable names
        - query_vals, evidence_vals: lists of the same length of integer assignments
        """
        # 1) Incorporate evidence
        factors = []
        for f in self.factors:
            f_red = f
            for v, val in zip(evidence_vars, evidence_vals):
                f_red = f_red.reduce(v, val)
            factors.append(f_red)

        # 2) Eliminate all other variables in elim_order
        to_eliminate = [
            v for v in self.elim_order
            if v not in query_vars and v not in evidence_vars
        ]
        for v in to_eliminate:
            # Gather factors involving v
            related = [f for f in factors if v in f.vars]
            if not related:
                continue
            # Multiply them and then marginalize out v
            f_prod = related[0]
            for f in related[1:]:
                f_prod = f_prod.multiply(f, self.domains)
            f_marg = f_prod.marginalize(v)
            # Replace old factors
            factors = [f for f in factors if f not in related] + [f_marg]

        # 3) Multiply remaining factors → joint over query_vars
        joint = factors[0]
        for f in factors[1:]:
            joint = joint.multiply(f, self.domains)

        # 4) Extract the single log‐numerator entry
        #    Build index in the exact order of joint.vars
        idx = tuple(
            query_vals[query_vars.index(v)]
            for v in joint.vars
        )
        log_num = joint.table[idx]

        # 5) Compute log‐denominator over ALL entries
        log_den = logsumexp(joint.table, axis=None)  

        # 6) Exponentiate & return as Python float
        prob = np.exp(log_num - log_den)
        return prob.item()
    
    def sample_full_unconditional(self) -> np.ndarray:
        """
        Fast unconditional sampling using forward sampling with ancestral ordering.
        Much faster than computing marginals for each node independently.
        
        Uses the elimination order (tree structure) to sample variables in an
        ancestral order where each variable is sampled conditioned only on its
        already-sampled neighbors.
        
        Returns:
            np.ndarray: Binary vector of length n representing infection status
        """
        rng = np.random.default_rng()
        X = np.zeros(self.n, dtype=int)
        
        # Sample in reverse elimination order (approximately ancestral)
        for var in reversed(self.elim_order):
            idx = int(var[1:])  # Extract index from "Xi"
            
            # Get unary factor for this variable
            log_phi = self.unary_factors[var].copy()  # [log_phi_0, log_phi_1]
            
            # Add contributions from pairwise factors with already-sampled neighbors
            for other_var in self.variables:
                if other_var == var:
                    continue
                other_idx = int(other_var[1:])
                
                # Check if this neighbor has been sampled
                key = frozenset([var, other_var])
                if key in self.pairwise_factors:
                    # Get pairwise factor
                    pairwise = self.pairwise_factors[key]  # 2x2 matrix in log space
                    
                    # Add contribution based on neighbor's value
                    if var < other_var:
                        # var is first dimension
                        log_phi[0] += pairwise[0, X[other_idx]]
                        log_phi[1] += pairwise[1, X[other_idx]]
                    else:
                        # var is second dimension
                        log_phi[0] += pairwise[X[other_idx], 0]
                        log_phi[1] += pairwise[X[other_idx], 1]
            
            # Convert log probabilities to actual probabilities
            # Normalize: p(X_i = 1) = exp(log_phi_1) / (exp(log_phi_0) + exp(log_phi_1))
            log_Z = logsumexp(log_phi)
            p1 = np.exp(log_phi[1] - log_Z)
            
            # Sample
            X[idx] = 1 if rng.random() < p1 else 0
        
        return X
    
    def sample_world_given(self, evidence_dict: dict) -> np.ndarray:
        """
        Sample a full world assignment using Gibbs sampling given evidence.
        Required by BinaryFrontierEnvBatch._sample_full_world().
        """
        # Simple Gibbs sampler
        rng = np.random.default_rng(42)
        X = rng.integers(0, 2, size=self.n, dtype=int)
        
        # Set evidence
        for var, val in evidence_dict.items():
            idx = int(var[1:])  # Extract index from "Xi"
            X[idx] = val
        
        # Run Gibbs sweeps
        for _ in range(10):  # 10 sweeps
            order = rng.permutation(self.n)
            for k in order:
                var_k = f"X{k}"
                if var_k in evidence_dict:
                    continue  # Don't resample evidence
                
                # Build evidence from current state
                evidence = {f"X{i}": int(X[i]) for i in range(self.n) if i != k}
                evidence.update(evidence_dict)
                
                # Compute conditional probability
                p1 = self.compute_conditional_probability({var_k: 1}, evidence)
                X[k] = 1 if rng.random() < p1 else 0
        
        return X
