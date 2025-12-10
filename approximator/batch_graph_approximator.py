# approximator/batch_graph_approximator.py

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB

# If you keep a tiny base class, you can replace this with your actual import.
class Approximator:
    pass


class BatchGraphApproximator(Approximator):
    """
    MILP-based action selector for batch node testing on a graph-like environment.

    Expected env interface (trim, adapt as needed):
      - env.observation() -> np.ndarray             # features for scenario embedding
      - env.num_nodes (int)                         # number of decision variables
      - env.budget (int)                            # how many nodes to select
      - env.allowed_mask() -> np.ndarray[bool]      # which indices are currently eligible (mask length == num_nodes)
      - env.random_feasible_action() -> np.ndarray  # fallback action (respects budget & mask), optional
      - (optional) env.project_to_feasible(a) -> np.ndarray  # project to feasible if available
    """

    def __init__(self, env, model_type: str = "NN-E"):
        self.env = env
        self.model_type = model_type
        self._var_order: List[gp.Var] = []  # ordered first-stage variables

    def solve_from_coeffs(self, c: np.ndarray) -> np.ndarray:
        c = np.asarray(c, dtype=float).reshape(-1)
        n = int(self.env.num_nodes)
        if c.shape[0] != n:
            raise ValueError(...)

        mask = self._current_allowed_mask(n)

        # Per-step budget: same logic as env._current_step_budget()
        if getattr(self.env, "round_budget", None) is not None:
            B = min(int(self.env.budget), int(self.env.round_budget))
        else:
            B = int(self.env.budget)

        scores = c.copy()
        scores[~mask] = -1e9

        a = np.zeros(n, dtype=float)
        k = min(B, int(mask.sum()))
        if k > 0 and np.isfinite(scores).any():
            topk = np.argpartition(-scores, k - 1)[:k]
            a[topk] = 1.0

        if hasattr(self.env, "project_to_feasible") and callable(self.env.project_to_feasible):
            try:
                a = np.asarray(self.env.project_to_feasible(a), dtype=float).reshape(-1)
            except Exception as e:
                print("[WARN] BatchGraphApproximator.solve_from_coeffs: "
                    f"project_to_feasible failed with {e}; using pre-projection a.")

        return a



    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------
    def approximate(
        self,
        network,                            # torch.nn.Module: input = concat([a, scenario]) -> scalar Q(s, a)
        mipper_cls,                         # class like Net2MIPPerScenario
        n_scenarios: int = 1,
        gap: float = 0.02,
        time_limit: int = 600,
        threads: int = 1,
        log_file: Optional[str] = None,
        scenario_embedding: Optional[np.ndarray] = None,
        scenario_probs: Optional[Sequence[float]] = None,
    ) -> Dict:
        """
        Build MILP, embed the Q-network via mipper, solve, and return the selected batch action.

        Returns:
            {
              'time': wall_clock,
              'predicted_obj': obj_value_or_nan,
              'sol': action_vector (np.ndarray),
              'solving_results': {'time':[], 'primal':[], 'dual':[], 'incumbent':[]},
              'solving_time': gurobi_runtime
            }
        """

        # 1) Build master MIP and ordered first-stage vars
        master_mip = self.get_master_mip()
        first_stage_vars = self._get_first_stage_variables(master_mip)  # ordered list
        first_stage_dict = {i: v for i, v in enumerate(first_stage_vars)}  # Net2MIP expects a dict

        # 2) Scenarios
        scenarios = self._prepare_scenarios(n_scenarios, scenario_embedding)
        if scenario_probs is None:
            scenario_probs = [1.0 / len(scenarios)] * len(scenarios)

        # 3) Diagnostics container
        solving_results = {"time": [], "primal": [], "dual": [], "incumbent": []}

        def callback(model: gp.Model, where: int):
            if where == gp.GRB.Callback.MIPSOL:
                solving_results["time"].append(model.cbGet(gp.GRB.Callback.RUNTIME))
                solving_results["primal"].append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST))
                solving_results["dual"].append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND))
                vals = model.cbGetSolution(first_stage_vars)  # list aligned with our order
                solving_results["incumbent"].append(vals)

        # 4) Embed neural Q(s,a) into MILP
        mipper = mipper_cls(
            first_stage_mip=master_mip,
            first_stage_vars=first_stage_dict,
            network=network,
            scenario_representations=scenarios,
            scenario_probs=scenario_probs,
        )
        approximator_mip = mipper.get_mip()

        # 5) Solver params
        approximator_mip.Params.LogToConsole = 0
        if log_file:
            approximator_mip.Params.LogFile = log_file
        approximator_mip.Params.TimeLimit = time_limit
        approximator_mip.Params.MIPGap = gap
        approximator_mip.Params.Threads = threads

        # 6) Solve
        t0 = time.time()
        approximator_mip.optimize(callback)
        wall = time.time() - t0

        # 7) Extract solution
        try:
            sol = self._get_first_stage_solution(approximator_mip)
            obj_val = approximator_mip.objVal
        except Exception as e:
            print(f"[WARN] Failed to extract solution (status={approximator_mip.Status}): {e}")
            sol = self._fallback_feasible_action()
            obj_val = float("nan")

        return {
            "time": wall,
            "predicted_obj": obj_val,
            "sol": sol,
            "solving_results": solving_results,
            "solving_time": getattr(approximator_mip, "Runtime", np.nan),
        }

    # --------------------------------------------------------------------------
    # Internals: model construction & solution I/O
    # --------------------------------------------------------------------------
    def get_master_mip(self) -> gp.Model:
        """
        Create the MILP with:
          - binary action vars action_i
          - budget constraint sum_i action_i <= B
          - allowed-mask constraint action_i <= mask_i
        """
        n = int(self.env.num_nodes)
        B = int(self.env.budget)
        mask = self._current_allowed_mask(n)

        model = gp.Model("batch_graph_selection")
        model.Params.OutputFlag = 0

        # Binary selection variables
        action = [model.addVar(vtype=GRB.BINARY, name=f"action_{i}") for i in range(n)]

        # Budget
        model.addConstr(gp.quicksum(action) <= B, name="budget")

        # Allowed mask
        for i in range(n):
            if not mask[i]:
                # force to 0 if not currently allowed
                model.addConstr(action[i] == 0, name=f"elig_{i}")

        # (No linear objective term here; Net2MIP will attach the Q(s,a) objective.)
        model.update()
        # Remember var order for consistent extraction
        self._var_order = action
        return model

    def _get_first_stage_variables(self, model: gp.Model) -> List[gp.Var]:
        """
        Return the ordered list of first-stage vars. We store in self._var_order
        during _build_master_mip() to avoid searching by name later.
        """
        if not self._var_order:
            # Fallback: gather by name order (less robust)
            vars_by_idx = []
            for i in range(self.env.num_nodes):
                v = model.getVarByName(f"action_{i}")
                if v is None:
                    raise RuntimeError(f"Missing var action_{i}")
                vars_by_idx.append(v)
            self._var_order = vars_by_idx
        return self._var_order

    def _get_first_stage_solution(self, model: gp.Model) -> np.ndarray:
        """
        Extract solution in the same order as _get_first_stage_variables.
        """
        if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
            raise RuntimeError("Model not solved to a state with an available incumbent.")
        vars_ordered = self._get_first_stage_variables(model)
        x = np.array([abs(v.X) for v in vars_ordered], dtype=float)
        # (Optional) round to {0,1} if you want strict binaries:
        x = (x >= 0.5).astype(float)
        return x

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------
    def _current_allowed_mask(self, n: int) -> np.ndarray:
        """
        Pull current allowed/eligible indices from env. If not provided by env,
        default to all-ones (all allowed).
        """
        if hasattr(self.env, "allowed_mask") and callable(self.env.allowed_mask):
            m = np.asarray(self.env.allowed_mask()).astype(bool)
            if m.shape[0] != n:
                raise ValueError("allowed_mask length mismatch.")
            return m
        return np.ones(n, dtype=bool)

    def _prepare_scenarios(
        self,
        n_scenarios: int,
        scenario_embedding: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Produce a list of scenario vectors. Default: 1 scenario := current observation.
        """
        if scenario_embedding is None:
            emb = np.asarray(self.env.observation()).reshape(-1)
            return [emb for _ in range(max(1, n_scenarios))]
        # If the user passed a tensor/ndarray, normalize to list-of-1
        if torch.is_tensor(scenario_embedding):
            return [scenario_embedding.detach().cpu().numpy().reshape(-1)]
        arr = np.asarray(scenario_embedding).reshape(-1)
        return [arr]

    def _fallback_feasible_action(self) -> Optional[np.ndarray]:
        """
        If solve fails, try to produce a feasible action from env.
        """
        try:
            if hasattr(self.env, "random_feasible_action"):
                a = np.asarray(self.env.random_feasible_action(), dtype=float)
            else:
                # naive fallback: choose up to B allowed positions uniformly
                n = int(self.env.num_nodes)
                B = int(self.env.budget)
                mask = self._current_allowed_mask(n)
                inds = np.flatnonzero(mask)
                k = min(B, len(inds))
                if k == 0:
                    return np.zeros(n, dtype=float)
                pick = np.random.choice(inds, size=k, replace=False)
                a = np.zeros(n, dtype=float)
                a[pick] = 1.0
            if hasattr(self.env, "project_to_feasible"):
                a = np.asarray(self.env.project_to_feasible(a), dtype=float)
            return a
        except Exception:
            return None