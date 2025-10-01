from __future__ import annotations

from typing import NamedTuple, Tuple, Dict
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import optax
import haiku as hk

from diffusion_online.relax.algorithm.base import Algorithm
from diffusion_online.relax.network.diffv2 import Diffv2Net, Diffv2Params
from diffusion_online.relax.utils.experience import Experience
from diffusion_online.relax.utils.typing_utils import Metric



class Diffv2OptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class Diffv2TrainState(NamedTuple):
    params: Diffv2Params
    opt_state: Diffv2OptStates
    step: jnp.ndarray
    entropy: jnp.ndarray
    running_mean: jnp.ndarray
    running_std: jnp.ndarray


class DPMD(Algorithm):

    def __init__(
        self,
        agent: Diffv2Net,
        params: Diffv2Params,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha_lr: float = 3e-2,
        lr_schedule_end: float = 5e-5,
        tau: float = 0.005,
        delay_alpha_update: int = 250,
        delay_update: int = 2,
        reward_scale: float = 0.2,
        num_samples: int = 200,
        use_ema: bool = True,
        lambda_temp: float = 1.0,
        M_smoothing: int = 2,
        J_smoothing: int = 2,
        kappa: float = 20.0
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.num_samples = num_samples
        self.use_ema = use_ema
        self.lambda_temp = float(lambda_temp)
        self.M_smoothing = int(M_smoothing)
        self.J_smoothing = int(J_smoothing)
        self.kappa = float(kappa)

        # Optimizers
        self.optim = optax.adam(lr)
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.alpha_optim = optax.adam(alpha_lr)

        # Initial train state
        self.state = Diffv2TrainState(
            params=params,
            opt_state=Diffv2OptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0),
        )

        def _spherical_perturb(key: jax.Array, mu: jnp.ndarray, kappa: float) -> jnp.ndarray:
            eps = 1e-8
            epsilon = jax.random.normal(key, mu.shape)
            dot = jnp.sum(epsilon * mu, axis=-1, keepdims=True)
            epsilon_tan = epsilon - dot * mu
            sigma = 1.0 / jnp.sqrt(kappa + 1e-6)
            z = mu + sigma * epsilon_tan
            return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + eps)

        

        @jax.jit
        def stateless_update(
            key: jax.Array, state: Diffv2TrainState, data: Experience
        ) -> Tuple[Diffv2TrainState, Dict[str, jnp.ndarray]]:
            obs, action, reward, next_obs, done = (
                data.obs, data.action, data.reward, data.next_obs, data.done
            )
            (q1_params, q2_params,
             target_q1_params, target_q2_params,
             policy_params, target_policy_params,
             log_alpha) = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state = state.opt_state
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std

            (next_eval_key, new_eval_key, new_q1_eval_key, new_q2_eval_key, log_alpha_key, diffusion_time_key, diffusion_noise_key) = jax.random.split(key, 7)

            reward_scaled = reward * self.reward_scale

            #min Q used for Bellman equation
            def get_min_q(s, a):
                q1 = self.agent.q(q1_params, s, a)
                q2 = self.agent.q(q2_params, s, a)
                return jnp.minimum(q1, q2)

            B = next_obs.shape[0]
            M, J = self.M_smoothing, self.J_smoothing

            # Sample c^(m) ~ pi(s')
            def sample_base_c(key_m):
                return self.agent.get_action(
                    key_m, (target_policy_params, log_alpha, q1_params, q2_params), next_obs
                )

            m_keys = jax.random.split(next_eval_key, M)
            cs_base = jax.vmap(sample_base_c)(m_keys)

            def _normalize(c):
                n = jnp.linalg.norm(c, axis=-1, keepdims=True) + 1e-8
                return c / n
            cs_base = _normalize(cs_base)

            # sample c^(m,j) ~ K_k(. | c^(m))
            def perturb_M_once(key_m, c_m):
                j_keys = jax.random.split(key_m, J)
                return jax.vmap(lambda kk: _spherical_perturb(kk, c_m, self.kappa))(j_keys)

            perturb_keys = jax.random.split(log_alpha_key, M)
            chat = jax.vmap(perturb_M_once)(perturb_keys, cs_base)

            # evaluate full equation
            def q_min_on(cand):
                q1 = self.agent.q(target_q1_params, next_obs, cand)
                q2 = self.agent.q(target_q2_params, next_obs, cand)
                return jnp.minimum(q1, q2)

            q_vals = jax.vmap(lambda c_m: jax.vmap(lambda c_mj: q_min_on(c_mj))(c_m))(chat)

            Vhat = jnp.mean(q_vals, axis=(0, 1))


            y_kappa = reward_scaled + (1.0 - done) * self.gamma * Vhat


            # critic update
            def q_loss_fn(q_params: hk.Params) -> Tuple[jax.Array, jax.Array]:
                q_pred = self.agent.q(q_params, obs, action)
                loss = jnp.mean((q_pred - y_kappa) ** 2)
                return loss, q_pred

            (q1_loss, q1_pred), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2_pred), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)

            # Apply critic parameter updates
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)


            def policy_loss_fn(policy_p: hk.Params):
                s = obs
                c0 = action

                q1_c0 = self.agent.q(q1_params, s, c0)
                q2_c0 = self.agent.q(q2_params, s, c0)
                q_min = jnp.minimum(q1_c0, q2_c0)


                w = jnp.exp(jax.lax.stop_gradient(q_min) / self.lambda_temp)

                eps = 1e-8
                z0 = c0 / (jnp.linalg.norm(c0, axis=-1, keepdims=True) + eps)

                t = jax.random.randint(diffusion_time_key, (s.shape[0],), 0, self.agent.num_timesteps)

                eps_key = diffusion_noise_key
                def denoiser(t_i, z_t_i):
                    return self.agent.policy(policy_p, s, z_t_i, t_i)

                loss = self.agent.diffusion.weighted_p_loss(
                    eps_key, w, denoiser, t, jax.lax.stop_gradient(z0)
                )

                q_mean, q_std = q_min.mean(), q_min.std()
                return loss, (w, q_mean, q_std)



            (policy_loss, (w, q_mean, q_std)), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)

            def log_alpha_loss_fn(cur_log_alpha: jax.Array) -> jax.Array:
                approx_entropy = 0.5 * self.agent.act_dim * jnp.log(
                    2 * jnp.pi * jnp.e * (0.1 * jnp.exp(cur_log_alpha)) ** 2
                )
                return -cur_log_alpha * (-jax.lax.stop_gradient(approx_entropy) + self.agent.target_entropy)

            def param_update(optim, params, grads, opt_state):
                upd, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, upd)
                return new_params, new_opt_state

            def maybe_delay(should, f_yes, f_no, *args):
                return jax.lax.cond(should, lambda _: f_yes(*args), lambda _: f_no(*args), operand=None)

            def delay_param_update(optim, params, grads, opt_state):
                return maybe_delay(
                    (step % self.delay_update) == 0,
                    lambda p, g, s: param_update(optim, p, g, s),
                    lambda p, g, s: (p, s),
                    params, grads, opt_state
                )

            def delay_alpha_param_update(optim, cur_log_alpha, opt_state):
                return maybe_delay(
                    (step % self.delay_alpha_update) == 0,
                    lambda a, s: param_update(optim, a, jax.grad(log_alpha_loss_fn)(a), s),
                    lambda a, s: (a, s),
                    cur_log_alpha, opt_state
                )

            def delay_target_update(src_params, tgt_params, tau):
                return maybe_delay(
                    (step % self.delay_update) == 0,
                    lambda s, t, tau_: optax.incremental_update(s, t, tau_),
                    lambda s, t, tau_: t,
                    src_params, tgt_params, tau
                )

            policy_params, policy_opt_state = delay_param_update(
                self.policy_optim, policy_params, policy_grads, policy_opt_state
            )
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(
                self.alpha_optim, log_alpha, log_alpha_opt_state
            )
            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

            new_running_mean = running_mean + 0.001 * (q_mean - running_mean)
            new_running_std = running_std + 0.001 * (q_std - running_std)

            new_state = Diffv2TrainState(
                params=Diffv2Params(
                    q1_params,
                    q2_params,
                    target_q1_params,
                    target_q2_params,
                    policy_params,
                    target_policy_params,
                    log_alpha
                ),
                opt_state=Diffv2OptStates(
                    q1=q1_opt_state, q2=q2_opt_state,
                    policy=policy_opt_state, log_alpha=log_alpha_opt_state
                ),
                step=step + 1,
                entropy=jnp.float32(0.0),
                running_mean=new_running_mean,
                running_std=new_running_std,
            )


            info: Dict[str, jnp.ndarray] = {
                "q1_loss": q1_loss,
                "q1_mean": jnp.mean(q1_pred),
                "q1_max": jnp.max(q1_pred),
                "q1_min": jnp.min(q1_pred),
                "q2_loss": q2_loss,
                "policy_loss": policy_loss,
                "alpha": jnp.exp(log_alpha),
                "w_mean": jnp.mean(w),
                "w_std": jnp.std(w),
                "w_min": jnp.min(w),
                "w_max": jnp.max(w),
                "q_at_c0_mean": q_mean,
                "q_at_c0_std": q_std,
                "running_q_mean": new_running_mean,
                "running_q_std": new_running_std,
            }
            return new_state, info

        def _get_target_policy_params(self):
            p = self.state.params
            # tolerate both spellings
            return getattr(p, "target_policy", getattr(p, "target_poicy"))

        def get_policy_params_to_save(self):
            return (self._get_target_policy_params(), self.state.params.log_alpha,
            self.state.params.q1, self.state.params.q2)

        def get_policy_params(self):
            p = self.state.params
            return (p.policy, p.log_alpha, p.q1, p.q2)



        @jax.jit
        def stateless_get_action(key: jax.Array, policy_params_tuple, obs_np):
            return self.agent.get_action(key, policy_params_tuple, obs_np)

        @jax.jit
        def stateless_get_deterministic_action(policy_params_tuple, obs_np):
            return self.agent.get_deterministic_action(policy_params_tuple, obs_np)

        self._implement_common_behavior(
            stateless_update,
            stateless_get_action,
            stateless_get_deterministic_action,
        )


    def get_policy_params(self):
        p = self.state.params
        return (p.policy, p.log_alpha, p.q1, p.q2)

    def get_policy_params_to_save(self):
        p = self.state.params
        return (p.target_policy, p.log_alpha, p.q1, p.q2)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params(), obs)
        return np.asarray(action)
