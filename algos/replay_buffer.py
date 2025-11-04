# algos/replay_buffer.py
from __future__ import annotations
import numpy as np
from typing import NamedTuple


class Experience(NamedTuple):
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_obs: np.ndarray
    done: np.ndarray
    action_star: np.ndarray
    policy_id: np.ndarray


class ReplayBuffer:
    """
    Simple uniform replay buffer.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self.obs_buf        = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf        = np.zeros((capacity, act_dim), dtype=np.float32)
        self.coeff_star_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew_buf        = np.zeros((capacity,),        dtype=np.float32)
        self.next_obs_buf   = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done_buf       = np.zeros((capacity,),        dtype=np.float32)
        self.policy_id_buf  = np.zeros((capacity,),        dtype=np.int64)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs,
        act_exec,
        rew,
        next_obs,
        done,
        *,
        coeff_star=None,
        policy_id: int = 0
    ):
        """Add a single transition."""
        i = self.ptr

        # Store with safe shaping/clipping to buffer dims
        self.obs_buf[i]      = np.asarray(obs,      dtype=np.float32).reshape(-1)[: self.obs_dim]
        self.act_buf[i]      = np.asarray(act_exec, dtype=np.float32).reshape(-1)[: self.act_dim]
        self.rew_buf[i]      = float(np.asarray(rew,  dtype=np.float32).reshape(-1)[0])
        self.next_obs_buf[i] = np.asarray(next_obs, dtype=np.float32).reshape(-1)[: self.obs_dim]
        self.done_buf[i]     = float(np.asarray(done, dtype=np.float32).reshape(-1)[0])

        if coeff_star is not None:
            self.coeff_star_buf[i] = np.asarray(coeff_star, dtype=np.float32).reshape(-1)[: self.act_dim]

        self.policy_id_buf[i] = int(policy_id)

        self.ptr  = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _choose_indices(self, batch_size: int) -> np.ndarray:
        assert self.size > 0, "ReplayBuffer is empty"
        B = int(batch_size)
        replace = (self.size < B)
        return np.random.choice(self.size, size=B, replace=replace)

    def sample(self, batch_size: int):
        """
        Uniform sampling over all filled entries (with replacement if needed).
        """
        idxs = self._choose_indices(batch_size)
        return (
            self.obs_buf[idxs],
            self.act_buf[idxs],
            self.rew_buf[idxs],
            self.next_obs_buf[idxs],
            self.done_buf[idxs],
            self.coeff_star_buf[idxs],
            self.policy_id_buf[idxs],
        )

    def sample_by_policy(self, batch_size: int, policy_version: int):
        return self.sample(batch_size)

    def sample_simple(self, batch_size: int) -> Experience:
        """Uniform sample that returns an Experience NamedTuple."""
        (o, a, r, no, d, cs, pid) = self.sample(batch_size)
        return Experience(
            obs=o, action=a, reward=r, next_obs=no, done=d,
            action_star=cs, policy_id=pid
        )

    @property
    def size_filled(self) -> int:
        return self.size
