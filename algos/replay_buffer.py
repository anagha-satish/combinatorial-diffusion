import numpy as np
import jax.numpy as jnp
from diffusion_online.relax.utils.experience import Experience

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((capacity,), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)
        self.ptr, self.size = 0, 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return Experience(
            obs=jnp.asarray(self.obs_buf[idxs]),
            action=jnp.asarray(self.act_buf[idxs]),
            reward=jnp.asarray(self.rew_buf[idxs]),
            next_obs=jnp.asarray(self.next_obs_buf[idxs]),
            done=jnp.asarray(self.done_buf[idxs]),
        )