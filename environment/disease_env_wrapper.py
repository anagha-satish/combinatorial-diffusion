# environment/disease_env_wrapper.py
import numpy as np
from environment.frontier_batch_env import BinaryFrontierEnvBatch

class DiseaseEnvWrapper:
    """
    Wrap BinaryFrontierEnvBatch to look like a standard env
    for run_dpmd_only.

    Observation = concat(status, frontier_mask).
    Action      = 0/1 vector of length n (same as DPMD outputs).
    """

    def __init__(self, base_env: BinaryFrontierEnvBatch):
        self.base = base_env
        self.n_arms = base_env.num_nodes     # what DPMD uses for "action_dim"
        self.num_nodes = base_env.num_nodes  # keep alias

    def reset(self):
        status, mask = self.base.reset()
        obs = np.concatenate([status, mask]).astype(np.float32)
        return obs

    def step(self, action_vec):
        # In case your policy produces soft actions:
        action_vec = np.asarray(action_vec, dtype=float)
        action_vec = (action_vec > 0.5).astype(int)

        status, mask, reward, done = self.base.step(action_vec)
        obs = np.concatenate([status, mask]).astype(np.float32)
        info = {"mask": mask}
        return obs, float(reward), bool(done), info

    # Convenience so your old code still works if it touches these:
    def observation(self):
        return self.base.observation()

    def allowed_mask(self):
        return self.base.allowed_mask()

    def random_feasible_action(self):
        return self.base.random_feasible_action()

    def project_to_feasible(self, a):
        return self.base.project_to_feasible(a)

    def get_approximator(self):
        return self.base.get_approximator()
