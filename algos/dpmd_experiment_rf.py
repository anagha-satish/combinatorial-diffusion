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

# -----------------------------
# Builder
# -----------------------------
def build_dpmd(obs_dim: int, act_dim: int, *, seed: int = 0, cfg: DPMDConfig | None = None) -> DPMD:
    """
    Initialize the RFM policy service and build the DPMD learner.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    rfm_service.init(obs_dim=obs_dim, act_dim=act_dim, lr=(cfg.lr if cfg else 1e-4), seed=seed, force=True)
    if hasattr(rfm_service, "sync_target_from_current"):
        rfm_service.sync_target_from_current()
    cfg = cfg or DPMDConfig()
    return DPMD(obs_dim=obs_dim, act_dim=act_dim, cfg=cfg)

# -----------------------------
# Trajectory collection
# -----------------------------
def collect_one_traj(env,
                     learner: DPMD,
                     horizon: int,
                     linear_solver: Callable[[np.ndarray], np.ndarray],
                     K: int,
                     *,
                     use_noise: bool = True,
                     policy_version: int = 0) -> tuple[List[tuple], float]:
    """
    One trajectory under current policy
    """
    traj, total = [], 0.0
    obs = _flat(_reset_obs(env))
    for _ in range(horizon):
        # 1) Sample K candidates, score, pick best c*
        Cs = learner.sample_candidates(obs, K=K)             # [K, D]
        qs = learner.score_actions(obs, Cs)                   # [K]
        i = int(np.argmax(qs))
        c_star = Cs[i]

        # 2) Optional on-sphere exec noise
        if use_noise:
            c_exec = rfm_service.perturb(c_star, kappa=learner.cfg.kappa_exec, J=1)[0, 0]
        else:
            c_exec = c_star

        # 3) Map to combinatorial action via linear_solver, step env
        action = linear_solver(c_exec)                        # map to feasible combinatorial action
        out = env.step(action)
        next_obs, rew, done, _ = _step_unpack(out)
        next_obs = _flat(next_obs)

        # 4) Store (s, tilde c, r, s', done, c*, policy_version)
        r_scalar = float(np.sum(rew)) if isinstance(rew, (list, np.ndarray)) else float(rew)
        traj.append((obs, c_exec, r_scalar, next_obs, float(done), c_star, policy_version))
        total += r_scalar
        obs = next_obs
        if done:
            break
    return traj, total

# -----------------------------
# Training wrapper
# -----------------------------
def train_one_step(learner: DPMD,
                   batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> dict:
    """
    Wrap a ReplayBuffer batch into Experience and call learner.update().
    """
    (obs_b, coeff_exec_b, rew_b, next_obs_b, done_b, coeff_star_b, policy_id_b) = batch
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
        action=np.asarray(coeff_exec_b, dtype=np.float32).reshape(B, -1),
        reward=_to_B(rew_b),
        next_obs=np.asarray(next_obs_b, dtype=np.float32).reshape(B, -1),
        done=_to_B(done_b),
        action_star=np.asarray(coeff_star_b, dtype=np.float32).reshape(B, -1),
        policy_id=np.asarray(policy_id_b, dtype=np.int64).reshape(B),
    )
    return learner.update(exp)

# -----------------------------
# End-to-end loop
# -----------------------------
def run_dpmd_only(
    env,
    horizon: int,
    budget: int,
    n_episodes_eval: int,
    seed: int,
    linear_solver,
    *,
    cfg: DPMDConfig,
    warmup_steps: int,
    batch_size: int,
    train_updates: int,
) -> np.ndarray:
    ...
    """
    End-to-end training/eval:
      • Warmup replay with exploration noise
      • Alternating collect/train with MD-weighted RFM actor (π_old batches)
      • Final greedy evaluation (no exec noise)
    """
    np.random.seed(seed); torch.manual_seed(seed)

    obs0 = _flat(_reset_obs(env))
    obs_dim = int(obs0.size)
    act_dim = int(env.n_arms)

    # Build learner (initializes RFM service)
    learner = build_dpmd(obs_dim, act_dim, seed=seed, cfg=cfg)

    # Replay buffer
    from algos.replay_buffer import ReplayBuffer
    buffer = ReplayBuffer(capacity=100_000, obs_dim=obs_dim, act_dim=act_dim)

    # Warmup: collect with noise under policy_version = 0
    while buffer.size_filled < warmup_steps:
        traj, _ = collect_one_traj(
            env, learner, horizon, linear_solver,
            K=learner.cfg.num_particles,
            use_noise=True,
            policy_version=learner.policy_version,
        )
        for (obs_val, c_exec, r_val, next_obs_val, done_val, c_star_val, pol_id) in traj:
            buffer.add(obs_val, c_exec, r_val, next_obs_val, done_val,
                       coeff_star=c_star_val, policy_id=pol_id)
        learner.policy_version += 1

    # Train loop
    loss_history = {"q1": [], "q2": [], "policy": []}
    for _ in range(train_updates):
        # Collect with current policy
        traj, _ = collect_one_traj(
            env, learner, horizon, linear_solver,
            K=learner.cfg.num_particles,
            use_noise=True,
            policy_version=learner.policy_version,
        )
        for (obs_val, c_exec, r_val, next_obs_val, done_val, c_star_val, pol_id) in traj:
            buffer.add(obs_val, c_exec, r_val, next_obs_val, done_val,
                       coeff_star=c_star_val, policy_id=pol_id)

        # Define π_old = previous version
        pi_old_version = max(0, learner.policy_version - 1)

        # Sample a π_old batch
        b_old = buffer.sample_by_policy(min(batch_size, buffer.size_filled),
                                        policy_version=pi_old_version)

        # Update learner
        info = train_one_step(learner, b_old)
        loss_history["q1"].append(info["q1_loss"])
        loss_history["q2"].append(info["q2_loss"])
        loss_history["policy"].append(info["policy_loss"])

        # Advance version after collect+update
        learner.policy_version += 1

    # Loss plot
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

    # Greedy evaluation
    rewards_all: List[float] = []
    for _ in range(n_episodes_eval):
        obs = _flat(_reset_obs(env))
        for t in range(horizon):
            Cs = learner.sample_candidates(obs, K=learner.cfg.num_particles)
            Xs = [linear_solver(c_i) for c_i in Cs]
            qs = learner.score_actions(obs, Cs)
            action = Xs[int(np.argmax(qs))]
            obs2, r, done, _ = _step_unpack(env.step(action))
            r_scalar = float(np.sum(r)) if isinstance(r, (list, np.ndarray)) else float(r)
            rewards_all.append(r_scalar)
            obs = _flat(obs2)
            if done:
                rewards_all.extend([0.0] * (horizon - t - 1))  # pad episode length
                break

    return np.asarray(rewards_all, dtype=np.float32)
