# algos/dpmd_experiment_rf.py
from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np
import torch
from algos.dpmd_rf import DPMD, DPMDConfig, Experience
from models.rfm.service import rfm_service
import matplotlib.pyplot as plt
import os
import datetime

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
    # Initialize the global RFM policy service with the final (obs_dim, act_dim).
    rfm_service.init(obs_dim=obs_dim, act_dim=act_dim, lr=1e-4, seed=seed, force=True)
    # Create the DPMD learner
    cfg = DPMDConfig(num_particles=num_particles)
    return DPMD(obs_dim=obs_dim, act_dim=act_dim, cfg=cfg)

def collect_one_traj(env, learner: DPMD, horizon: int,
                     linear_solver: Callable[[np.ndarray], np.ndarray],
                     K: int) -> tuple[list[tuple], float]:
    """Collect one trajectory using the current policy and critics."""
    traj, total = [], 0.0
    obs = _flat(_reset_obs(env))
    for _ in range(horizon):
        # Sample K base latents from policy without noise
        Cs_base = learner.sample_candidates(obs, K=K)     # [K, D]

        # Score and pick c*
        qs = learner.score_actions(obs, Cs_base)          # [K]
        i = int(np.argmax(qs))
        c_star = Cs_base[i] 

        # Draw on-sphere perturbation
        c_hat = rfm_service.perturb(c_star, kappa=learner.cfg.kappa_exec, J=1)[0, 0]  # [D]

        # Map coefficients to a feasible action
        action = linear_solver(c_hat)

        # Step environment
        out = env.step(action)
        next_obs, rew, done, _ = _step_unpack(out)
        next_obs = _flat(next_obs)
        r_scalar = float(np.sum(rew)) if isinstance(rew, (list, np.ndarray)) else float(rew)
        
        # Store values in trajectory
        traj.append((obs, c_hat, r_scalar, next_obs, float(done)))
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
            Xs = [linear_solver(c_i) for c_i in Cs]  # mapped env actions
            qs = learner.score_actions(obs, Cs)
            action = Xs[int(np.argmax(qs))]          # best-Q action
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

    # Create DPMD learner
    learner = build_dpmd(obs_dim, act_dim, seed=seed, num_particles=8)

    # Warmup replay buffer
    from algos.replay_buffer import ReplayBuffer
    buffer = ReplayBuffer(capacity=100_000, obs_dim=obs_dim, act_dim=act_dim)

    warmup_steps = 256
    while buffer.size < warmup_steps:
        traj, _ = collect_one_traj(env, learner, horizon, linear_solver, K=learner.cfg.num_particles)
        for (obs_val, c_val, r_val, next_obs_val, done_val) in traj:
            buffer.add(obs_val, c_val, r_val, next_obs_val, done_val)

    # Training loop
    batch_size, train_updates = 64, 500

    loss_history = {"q1": [], "q2": [], "policy": []}

    for it in range(train_updates):
        # Sample a mini-batch
        b = buffer.sample(min(batch_size, buffer.size))   # tuple of (obs, act, rew, next_obs, done)
        B = len(b[0])                                     # actual batch size this step
        info = train_one_step(learner, b)
        loss_history["q1"].append(info["q1_loss"])
        loss_history["q2"].append(info["q2_loss"])
        loss_history["policy"].append(info["policy_loss"])

    # Plot & save loss curves
    os.makedirs("loss_curves", exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plt.figure(figsize=(6,4))
    plt.plot(loss_history["q1"], label="Q1 loss")
    plt.plot(loss_history["q2"], label="Q2 loss")
    plt.plot(loss_history["policy"], label="Policy (RFM) loss")
    plt.xlabel("Training iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("DPMD-RF Training Losses")
    plt.tight_layout()
    plt.savefig(f"loss_curves/dpmd_rf_losses_{ts}.png", dpi=200)
    plt.close()

    # Evaluation
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

