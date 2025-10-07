# algos/dpmd_experiment_rf.py
from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np
import torch
from algos.dpmd_rf import DPMD, DPMDConfig, Experience
from models.rfm.service import rfm_service

def _reset_obs(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out

def _step_unpack(out):
    if len(out) == 5:
        obs, rew, term, trunc, info = out
        return obs, rew, (term or trunc), info
    obs, rew, done, info = out
    return obs, rew, done, info

def _flat(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)

def build_dpmd(obs_dim: int, act_dim: int, *, seed: int = 0, num_particles: int = 8) -> DPMD:
    torch.manual_seed(seed); np.random.seed(seed)
    rfm_service.init(obs_dim=obs_dim, act_dim=act_dim, lr=1e-4, seed=seed, force=True)
    cfg = DPMDConfig(num_particles=num_particles)
    return DPMD(obs_dim=obs_dim, act_dim=act_dim, cfg=cfg)

def collect_one_traj(env, learner: DPMD, horizon: int,
                     linear_solver: Callable[[np.ndarray], np.ndarray],
                     K: int) -> tuple[list[tuple], float]:
    traj, total = [], 0.0
    obs = _flat(_reset_obs(env))
    for _ in range(horizon):
        Cs = learner.sample_candidates(obs, K=K)     # [K, D]
        Xs = [linear_solver(c_i) for c_i in Cs]
        qs = learner.score_actions(obs, Cs)          # [K]
        i = int(np.argmax(qs))
        c_best, action = Cs[i], Xs[i]
        out = env.step(action)
        next_obs, rew, done, _ = _step_unpack(out)
        next_obs = _flat(next_obs)
        r_scalar = float(np.sum(rew)) if isinstance(rew, (list, np.ndarray)) else float(rew)
        traj.append((obs, c_best, r_scalar, next_obs, float(done)))
        total += r_scalar
        obs = next_obs
        if done:
            break
    return traj, total

def train_one_step(learner: DPMD,
                   batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> dict:
    obs_b, coeff_b, rew_b, next_obs_b, done_b = batch
    B = len(obs_b)

    def _to_B(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32)
        # Make [B, ...]
        if arr.ndim == 0:
            return np.full((B,), float(arr), dtype=np.float32)
        if arr.ndim == 1:
            # Already [B]
            return arr[:B]
        # Flatten trailing dims and take the first element per row
        arr2 = arr.reshape(arr.shape[0], -1)
        if arr2.shape[0] != B:
            arr2 = arr2[:B] if arr2.shape[0] > B else np.pad(arr2, ((0, B - arr2.shape[0]), (0, 0)))
        return arr2[:, 0].astype(np.float32, copy=False)

    exp = Experience(
        obs=np.asarray(obs_b, dtype=np.float32).reshape(B, -1),
        action=np.asarray(coeff_b, dtype=np.float32).reshape(B, -1),
        reward=_to_B(rew_b),
        next_obs=np.asarray(next_obs_b, dtype=np.float32).reshape(B, -1),
        done=_to_B(done_b),
    )
    return learner.update(exp)


def evaluate_policy(env, learner: DPMD, horizon: int, linear_solver, episodes=5, seed=0):
    np.random.seed(seed)
    rs, steps = 0.0, 0
    for _ in range(episodes):
        obs = _flat(_reset_obs(env))
        for _ in range(horizon):
            Cs = learner.sample_candidates(obs, K=learner.cfg.num_particles)
            Xs = [linear_solver(c_i) for c_i in Cs]
            qs = learner.score_actions(obs, Cs)
            action = Xs[int(np.argmax(qs))]
            obs2, r, done, _ = _step_unpack(env.step(action))
            obs = _flat(obs2)
            r_scalar = float(np.sum(r)) if isinstance(r, (list, np.ndarray)) else float(r)
            rs += r_scalar; steps += 1
            if done: break
    return rs / max(1, steps)

def run_dpmd_only(env, t_env, horizon: int, budget: int, n_episodes_eval: int,
                  seed: int, linear_solver: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    np.random.seed(seed); torch.manual_seed(seed)

    obs0 = _flat(_reset_obs(env))
    obs_dim = int(obs0.size)
    act_dim = int(env.n_arms)

    learner = build_dpmd(obs_dim, act_dim, seed=seed, num_particles=8)

    # --- warmup with random trajectories (using the current RFM sampler anyway) ---
    from algos.replay_buffer import ReplayBuffer
    buffer = ReplayBuffer(capacity=100_000, obs_dim=obs_dim, act_dim=act_dim)

    warmup_steps = 256
    while buffer.size < warmup_steps:
        traj, _ = collect_one_traj(env, learner, horizon, linear_solver, K=learner.cfg.num_particles)
        for (obs_val, c_val, r_val, next_obs_val, done_val) in traj:
            buffer.add(obs_val, c_val, r_val, next_obs_val, done_val)

    # --- training ---
    batch_size, train_updates = 64, 500
    for _ in range(train_updates):
        b = buffer.sample(min(batch_size, buffer.size))
        train_one_step(learner, b)

    # --- evaluation (return per-step rewards to match driver contract) ---
    rewards_all: List[float] = []
    for _ in range(n_episodes_eval):
        traj, _ = collect_one_traj(env, learner, horizon, linear_solver, K=learner.cfg.num_particles)
        # pad to fixed horizon
        traj += [(None, None, 0.0, None, 1.0)] * (horizon - len(traj))
        rewards_all.extend([t[2] for t in traj])

    avg_r = evaluate_policy(env, learner, horizon, linear_solver, episodes=5, seed=seed)
    print(f"[eval] avg per-step reward: {avg_r:.3f}")

    return np.array(rewards_all, dtype=np.float32)

__all__ = ["run_dpmd_only"]

