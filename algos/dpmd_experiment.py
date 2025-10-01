from __future__ import annotations

from typing import Callable, List, Tuple
import os
import numpy as np

import jax
import jax.numpy as jnp

# Diffv2 stack
from diffusion_online.relax.network.diffv2 import (
    create_diffv2_net,
    Diffv2Net,
    Diffv2Params,
)
from algos.dpmd_jax import DPMD
from diffusion_online.relax.utils.experience import Experience
from algos.replay_buffer import ReplayBuffer


def _reset_obs(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) and len(out) == 2 else out

def _step_unpack(out):
    if len(out) == 5:
        obs, rew, term, trunc, info = out
        return obs, rew, (term or trunc), info
    else:
        obs, rew, done, info = out
        return obs, rew, done, info


def build_dpmd(
    obs_dim: int,
    act_dim: int,
    *,
    seed: int = 0,
    num_timesteps: int = 20,
    num_particles: int = 8,
    hidden_sizes: Tuple[int, int] = (256, 256),
    diffusion_hidden_sizes: Tuple[int, int] = (256, 256),
):
    key = jax.random.PRNGKey(seed)

    agent, params = create_diffv2_net(
        key=key,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=hidden_sizes,
        diffusion_hidden_sizes=diffusion_hidden_sizes,
        num_timesteps=num_timesteps,
        num_particles=num_particles,
        noise_scale=0.05,
        target_entropy_scale=0.9,
        beta_schedule_scale=0.3,
    )

    learner = DPMD(
        agent=agent,
        params=params,
        gamma=0.99,
        lr=1e-4,
        alpha_lr=3e-2,
        lr_schedule_end=5e-5,
        tau=0.005,
        delay_alpha_update=250,
        delay_update=2,
        reward_scale=0.2,
        num_samples=agent.num_particles,
        use_ema=True,
        lambda_temp=1.0,
    )
    return learner, agent



def collect_one_traj(env, learner, key, horizon, linear_solver):
    traj = []
    total = 0.0
    obs = _reset_obs(env)

    for _ in range(horizon):
        key, sub = jax.random.split(key)

        K = learner.agent.num_particles
        keys = jax.random.split(sub, K)
        cs = [np.asarray(learner.get_action(k_i, np.asarray(obs))) for k_i in keys]

        xs = [linear_solver(c_i) for c_i in cs]

        qs = []
        for c_i in cs:
            c_i_jnp = jnp.asarray(c_i)[None]
            obs_jnp = jnp.asarray(obs)[None]
            q1 = learner.agent.q(learner.state.params.q1, obs_jnp, c_i_jnp)
            q2 = learner.agent.q(learner.state.params.q2, obs_jnp, c_i_jnp)
            q_min = jnp.minimum(q1, q2)
            q_scalar = jnp.squeeze(q_min)
            qs.append(float(q_scalar))

        i_star = int(np.argmax(qs))
        c_best = cs[i_star]
        action = xs[i_star]

        out = env.step(action)
        next_obs, rew, done, _ = _step_unpack(out)
        r_scalar = float(np.sum(rew)) if isinstance(rew, (list, np.ndarray)) else float(rew)

        traj.append((obs, c_best, r_scalar, next_obs, float(done)))
        total += r_scalar
        obs = next_obs
        if done:
            break

    return traj, total


def train_one_step_with_diffv2(
    learner: DPMD,
    batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    key: jax.Array,
):
    obs_b, coeff_b, rew_b, next_obs_b, done_b = batch
    exp = Experience(
        obs=obs_b,
        action=coeff_b,
        reward=rew_b,
        next_obs=next_obs_b,
        done=done_b,
    )
    new_state, info = learner.update(key, exp)
    return {k: float(v) for k, v in info.items()}



def evaluate_policy(env, learner, horizon, linear_solver, episodes=5, seed=0):
    key = jax.random.PRNGKey(seed)
    total_r, total_steps = 0.0, 0

    for _ in range(episodes):
        obs = _reset_obs(env)
        for t in range(horizon):
            key, sub = jax.random.split(key)
            c = np.asarray(learner.get_action(sub, np.asarray(obs)))
            a = linear_solver(c)
            out = env.step(a)
            obs2, r, done, _ = _step_unpack(out)
            r_scalar = float(np.sum(r)) if isinstance(r, (list, np.ndarray)) else float(r)
            total_r += r_scalar
            total_steps += 1
            obs = obs2
            if done:
                break

    return total_r / max(1, total_steps)


def run_dpmd_only(
    env,
    t_env,
    horizon: int,
    budget: int,
    n_episodes_eval: int,
    seed: int,
    linear_solver: Callable[[np.ndarray], np.ndarray],
):
    key = jax.random.PRNGKey(seed)
    obs0 = _reset_obs(env)
    obs_dim = int(np.asarray(obs0).shape[-1])

    learner, _agent = build_dpmd(obs_dim, env.n_arms, seed=seed)

    warmup_steps   = 256
    batch_size     = 64
    update_iters   = 1  
    train_updates  = 500 

    # 1) Buffer
    buffer = ReplayBuffer(capacity=100_000, obs_dim=obs_dim, act_dim=env.n_arms)

    while buffer.size < warmup_steps:
        traj, _ = collect_one_traj(env, learner, key, horizon, linear_solver)
        for (obs, c, r, next_obs, done) in traj:
            buffer.add(obs, c, r, next_obs, done)

    # 2) Training
    for _ in range(train_updates):
        exp = buffer.sample(min(batch_size, buffer.size))
        key, sub = jax.random.split(key)
        _ = learner.update(sub, exp)

    # 3) Evaluate
    rewards_all: List[float] = []
    for _ in range(n_episodes_eval):
        key, sub = jax.random.split(key)
        traj, _ = collect_one_traj(env, learner, sub, horizon, linear_solver)
        if len(traj) < horizon:
            traj += [(None, None, 0.0, None, 1.0)] * (horizon - len(traj))
        rewards_all += [t[2] for t in traj]

    avg_r = evaluate_policy(env, learner, horizon, linear_solver, episodes=5, seed=seed)
    print(f"[eval] avg per-step reward: {avg_r:.3f}")

    return np.array(rewards_all, dtype=np.float32)

