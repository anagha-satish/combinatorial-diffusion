# binary_frontier_environment_batch.py

from __future__ import annotations
import copy
from typing import Optional, Dict, Tuple

import networkx as nx
import numpy as np

from environment.abstract_joint_probability_class import AbstractJointProbabilityClass


class BinaryFrontierEnvBatch:
    """
    Graph frontier testing simulator with batch reveals.

    Key points:
      - A full hidden world X ~ P is sampled ONCE per episode (self.world_X).
      - step(action_vector) reveals all nodes in action_vector from the SAME world_X.
      - frontier_mask_from_status(status) deterministically recomputes the valid-action mask.
      - Rewards: sum of revealed positives in the action set.
    """

    def __init__(
        self,
        G: nx.Graph,
        P: AbstractJointProbabilityClass,
        discount_factor: float,
        cc_dict: Optional[dict] = None,
        cc_root: Optional[list] = None,
        rng_seed: int = 314159,
        budget: Optional[int] = None,
    ) -> None:
        assert 0.0 < discount_factor < 1.0
        self.P = copy.deepcopy(P)
        self.n = G.number_of_nodes()
        assert self.n == P.n
        self.discount_factor = float(discount_factor)
        self.rng = np.random.default_rng(rng_seed)
        # relabel nodes as X0, X1, ...
        self.G = nx.relabel_nodes(G, {i: f"X{i}" for i in range(self.n)})
        self.tests_done = 0
        self.status = np.full(self.n, -1, dtype=int)  # -1 = unknown, 0/1 = revealed
        self.world_X = None  # full hidden assignment for current episode
        self.budget = int(budget) if budget is not None else self.n  # default: no budget limit

        # convenient alias for approximator / DQN
        self.num_nodes = self.n

        # Pre-process per-CC root (highest marginal) for frontier expansion
        if cc_dict is not None and cc_root is not None:
            self.cc_dict = cc_dict
            self.cc_root = cc_root
        else:
            self.cc_dict = dict()
            self.cc_root = []
            for cc_nodes in nx.connected_components(self.G):
                self.cc_dict[frozenset(cc_nodes)] = len(self.cc_dict)
                indices = [int(v[1:]) for v in cc_nodes]
                marginal_prob1s = [(self.get_marginal_prob1(idx), idx) for idx in indices]
                # choose argmax P(X_i=1 | current empty status)
                self.cc_root.append(sorted(marginal_prob1s, reverse=True)[0][1])

        # A cached valid mask that callers (e.g., DQN) may store after reset/step
        self._last_valid_mask_np: Optional[np.ndarray] = None

    # ---------- episode control ----------

    def _sample_full_world(self) -> np.ndarray:
        """
        Draw a full assignment X ~ P once per episode.
        Prefer exact sampler if you have it; otherwise fall back.
        """
        if hasattr(self.P, "sample_full_unconditional"):
            return np.array(self.P.sample_full_unconditional(), dtype=int)
        elif hasattr(self.P, "sample_world_given"):
            return np.array(self.P.sample_world_given(evidence_dict={}), dtype=int)
        else:
            # Fallback: i.i.d. sampling from node marginals
            X = np.zeros(self.n, dtype=int)
            empty = np.full(self.n, -1, dtype=int)
            for i in range(self.n):
                p1 = self.get_marginal_prob1(i, observed_status=empty)
                X[i] = int(self.rng.random() <= p1)
            return X

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.tests_done = 0
        self.status[:] = -1
        self.world_X = self._sample_full_world()
        mask = self.frontier_mask_from_status(self.status)
        self._last_valid_mask_np = mask.copy()
        return self.status.copy(), mask

    # ---------- probabilities ----------

    def get_status_and_factors(self) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Returns:
          status (copy),
          unary_factors (dict: "X_i" -> factor),
          pairwise_factors (dict: frozenset({"X_i","X_j"}) -> factor)
        """
        return self.status.copy(), self.P.unary_factors.copy(), self.P.pairwise_factors.copy()

    def compute_conditional_probability(self, query_dict: dict, observation_dict: dict) -> float:
        assert len(set(query_dict.keys()).intersection(observation_dict.keys())) == 0
        return float(self.P.compute_conditional_probability(query_dict, observation_dict))

    def get_marginal_prob1(self, index: int, observed_status: Optional[np.ndarray] = None) -> float:
        status = self.status if observed_status is None else observed_status
        val = status[index]
        if val == 1:
            return 1.0
        if val == 0:
            return 0.0
        query_dict = {f"X{index}": 1}
        observation_dict = {f"X{i}": int(status[i]) for i in range(self.n) if status[i] != -1}
        return self.compute_conditional_probability(query_dict, observation_dict)

    # ---------- frontier / mask ----------

    def _frontier_set_from_status(self, status: np.ndarray) -> set[int]:
        tested = set([f"X{i}" for i in range(self.n) if status[i] != -1])
        frontier = set()
        for cc_nodes in nx.connected_components(self.G):
            if len(cc_nodes.intersection(tested)) == 0:
                # choose precomputed root for this CC
                argmax_in_cc = self.cc_root[self.cc_dict[frozenset(cc_nodes)]]
                frontier.add(argmax_in_cc)
            else:
                # standard frontier: neighbors of tested nodes that are untested
                for v in cc_nodes:
                    if v not in tested and len(set(self.G.neighbors(v)).intersection(tested)) > 0:
                        frontier.add(int(v[1:]))
        return frontier

    def frontier_mask_from_status(self, status: np.ndarray) -> np.ndarray:
        fset = self._frontier_set_from_status(status)
        mask = np.zeros(self.n, dtype=int)
        for i in fset:
            mask[i] = 1
        return mask

    # ---------- stepping ----------

    def step(self, action_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Non-adaptive batch step (vector): action_vector[i] ∈ {0,1} must be from PRE-step frontier.
        Reveals from fixed world_X. Reward = sum revealed positives.
        """
        action_vector = np.asarray(action_vector, dtype=int)
        assert action_vector.shape == (self.n,)
        pre_mask = self.frontier_mask_from_status(self.status)
        # ensure feasibility w.r.t frontier
        assert np.all(action_vector[pre_mask == 0] == 0), "Chosen nodes must be from the current frontier."
        k = int(action_vector.sum())
        assert k <= self.budget, f"Exceeds budget: {k} > {self.budget}"

        reward = 0.0
        for i in np.flatnonzero(action_vector):
            self.status[i] = int(self.world_X[i])
            reward += float(self.status[i])

        self.tests_done += k

        next_mask = self.frontier_mask_from_status(self.status)
        done = (self.tests_done == self.n)
        self._last_valid_mask_np = next_mask.copy()
        return self.status.copy(), next_mask, reward, done

    # Optional: single-index helper (kept for compatibility)
    def step_single(self, action_idx: int) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        vec = np.zeros(self.n, dtype=int)
        vec[int(action_idx)] = 1
        return self.step(vec)

    # ---------- extra methods for DQN + BatchGraphApproximator ----------

    def observation(self) -> np.ndarray:
        """
        Observation vector used as scenario embedding in approximator.
        Here we simply use the current status (cast to float).
        """
        return self.status.astype(float).copy()

    def allowed_mask(self) -> np.ndarray:
        """
        Mask of currently allowed nodes (frontier) based on internal status.
        """
        return self.frontier_mask_from_status(self.status).astype(int)

    def random_feasible_action(self) -> np.ndarray:
        """
        Random feasible batch action respecting budget and allowed_mask.
        """
        mask = self.allowed_mask().astype(bool)
        inds = np.flatnonzero(mask)
        B = self.budget
        k = min(B, len(inds))
        a = np.zeros(self.n, dtype=float)
        if k > 0:
            picks = self.rng.choice(inds, size=k, replace=False)
            a[picks] = 1.0
        return a

    def project_to_feasible(self, a: np.ndarray) -> np.ndarray:
        """
        Project a possibly infeasible action vector to one that respects
        current allowed_mask and budget.
        """
        a = np.asarray(a, dtype=float).copy()
        mask = self.allowed_mask().astype(bool)
        a[~mask] = 0.0
        B = self.budget
        if a.sum() > B:
            inds = np.flatnonzero(a > 0.5)
            if len(inds) > B:
                drop = self.rng.choice(inds, size=(len(inds) - B), replace=False)
                a[drop] = 0.0
        return a

    # aliases for older naming patterns
    def get_random_action(self) -> np.ndarray:
        return self.random_feasible_action()

    def get_approximator(self):
        """
        Return the approximator class used by DQN to embed the Q-network into a MILP.
        """
        from approximator.batch_graph_approximator import BatchGraphApproximator
        return BatchGraphApproximator