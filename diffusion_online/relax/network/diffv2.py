from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple, Union

import jax, jax.numpy as jnp
import haiku as hk
import math

from ..network.blocks import Activation, DistributionalQNet2, DACERPolicyNet, QNet
from ..network.common import WithSquashedGaussianPolicy
from ..utils.diffusion import GaussianDiffusion
from ..utils.jax_utils import random_key_from_data

class Diffv2Params(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    target_policy: hk.Params
    log_alpha: jax.Array


@dataclass
class Diffv2Net:
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    num_timesteps: int
    act_dim: int
    num_particles: int
    target_entropy: float
    noise_scale: float
    beta_schedule_scale: float
    beta_schedule_type: str = 'linear'

    @property
    def diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(self.num_timesteps, 
                                 self.beta_schedule_scale,
                                 self.beta_schedule_type)

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_p, log_alpha, q1_params, q2_params = policy_params

        def model_fn(t, z):
            return self.policy(policy_p, obs, z, t)

        z0 = self.diffusion.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
        z0 = z0 + jax.random.normal(key, z0.shape) * jnp.exp(log_alpha) * self.noise_scale
        c0 = self.map_z_to_c(z0)
        return c0


    def get_batch_actions(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array, q_func):
        batch_flatten_obs = obs.repeat(self.num_particles, axis=0)
        batch_flatten_c = self.get_action(key, policy_params, batch_flatten_obs)
        batch_q = q_func(batch_flatten_obs, batch_flatten_c).reshape(-1, self.num_particles)
        max_q_idx = batch_q.argmax(axis=1)
        Cs = batch_flatten_c.reshape(obs.shape[0], self.num_particles, self.act_dim)
        best_c = jax.vmap(lambda row, i: row[i])(Cs, max_q_idx)
        return best_c


    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        key = random_key_from_data(obs)
        policy_params, log_alpha, q1_params, q2_params = policy_params
        no_explore = -jnp.inf
        policy_tuple = (policy_params, no_explore, q1_params, q2_params)
        return self.get_action(key, policy_tuple, obs)  # returns c

    def q_evaluate(self, key: jax.Array, q_params: hk.Params, obs: jax.Array, act: jax.Array):
        q_mean = self.q(q_params, obs, act)
        q_std = jnp.zeros_like(q_mean)
        z = jax.random.normal(key, q_mean.shape)
        z = jnp.clip(z, -3.0, 3.0)
        q_value = q_mean + q_std * z
        return q_mean, q_std, q_value

    def map_z_to_c(self, z: jax.Array) -> jax.Array:
        # sphere projection
        eps = 1e-8
        norm = jnp.linalg.norm(z, axis=-1, keepdims=True) + eps
        return z / norm


def create_diffv2_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    num_particles: int = 4,
    noise_scale: float = 0.05,
    target_entropy_scale: float = 0.9,
    beta_schedule_scale: float = 0.3,
    ) -> Tuple[Diffv2Net, Diffv2Params]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(hk.transform(lambda obs, act, t: DACERPolicyNet(diffusion_hidden_sizes, activation)(obs, act, t)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, policy_key = jax.random.split(key, 3)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs, act, 0)
        target_policy_params = policy_params
        log_alpha = jnp.array(math.log(5), dtype=jnp.float32)
        return Diffv2Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = Diffv2Net(q=q.apply, policy=policy.apply, num_timesteps=num_timesteps, act_dim=act_dim, 
                    target_entropy=-act_dim*target_entropy_scale, num_particles=num_particles, noise_scale=noise_scale,
                    beta_schedule_scale=beta_schedule_scale)
    return net, params
