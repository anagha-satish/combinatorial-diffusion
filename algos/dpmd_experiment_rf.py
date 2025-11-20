# algos/dpmd_experiment_rf.py
from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np
import torch
from algos.dpmd_rf import DPMD, DPMDConfig, Experience
from models.rfm.service import rfm_service
import matplotlib.pyplot as plt


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
        arr = np.asarray(vec)

        # Case 1: scalar -> broadcast to length B
        if arr.ndim == 0:
            return np.full((B,), float(arr), dtype=np.float32)

        # Ensure first dim matches B (trim or pad rows)
        if arr.shape[0] != B:
            if arr.shape[0] > B:
                arr = arr[:B]
            else:
                pad_shape = [(0, B - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
                arr = np.pad(arr, pad_shape, mode="constant")

        # Squeeze trailing singleton dims (e.g., [B,1] -> [B])
        while arr.ndim > 1 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)

        # Final shape must be [B]
        if arr.ndim != 1:
            raise ValueError(f"Expected rewards/dones with shape [B] or [B,1], got {arr.shape}")

        return arr.astype(np.float32, copy=False)


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
    """
    DQN-style training loop (episodes × horizon, update per step),
    followed by final greedy evaluation.
    """
    np.random.seed(seed); torch.manual_seed(seed)

    # Infer dims
    obs0 = _flat(_reset_obs(env))
    obs_dim = int(obs0.size)
    act_dim = int(env.n_arms)

    # Build learner (initializes RFM service)
    learner = build_dpmd(obs_dim, act_dim, seed=seed, cfg=cfg)
    assert act_dim == env.n_arms, f"act_dim mismatch: {act_dim} vs env.n_arms={env.n_arms}"

    # Replay buffer
    from algos.replay_buffer import ReplayBuffer
    buffer = ReplayBuffer(capacity=100_000, obs_dim=obs_dim, act_dim=act_dim)


    # -------- Warmup: collect transitions with exec noise --------
    steps_collected = 0
    while steps_collected < warmup_steps:
        obs = _flat(_reset_obs(env))
        for _ in range(horizon):
            # sample K candidates -> greedy c*
            Cs = learner.sample_candidates(obs, K=learner.cfg.num_particles)
            qs = learner.score_actions(obs, Cs)
            i = int(np.argmax(qs))
            c_star = Cs[i]

            # executed coefficient on sphere (noise)
            c_exec = rfm_service.perturb(c_star, kappa=learner.cfg.kappa_exec, J=1)[0, 0]

            # map to feasible combinatorial action -> env step
            action = linear_solver(c_exec)
            out = env.step(action)
            next_obs, rew, done, _ = _step_unpack(out)
            next_obs = _flat(next_obs)
            r_scalar = float(np.sum(rew)) if isinstance(rew, (list, np.ndarray)) else float(rew)

            buffer.add(obs, c_exec, r_scalar, next_obs, float(done),
                       coeff_star=c_star, policy_id=learner.policy_version)

            obs = next_obs
            steps_collected += 1
            if done or steps_collected >= warmup_steps:
                break
        learner.policy_version += 1

    # -------- Tiny myopic pretraining: critics only (gamma=0) --------
    pretrain_critic_iters = min(64, max(8, warmup_steps // max(1, 32 * batch_size)))

    for _ in range(int(pretrain_critic_iters)):
        b = buffer.sample(min(batch_size, buffer.size_filled))
        learner.pretrain_critics_step(Experience(*b))

    # Hard resync targets and actor after pretrain
    if hasattr(rfm_service, "sync_target_from_current"):
        rfm_service.sync_target_from_current()
    for p_t, p in zip(learner.tq1.parameters(), learner.q1.parameters()):
        p_t.data.copy_(p.data)
    for p_t, p in zip(learner.tq2.parameters(), learner.q2.parameters()):
        p_t.data.copy_(p.data)


    # -------- Training: episodic, per-step updates--------
    import os, datetime, matplotlib.pyplot as plt
    os.makedirs("loss_curves", exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    loss_history = {"q1": [], "q2": [], "policy": []}

    for ep in range(int(train_updates)):   # interpret train_updates as #episodes
        obs = _flat(_reset_obs(env))
        for t in range(horizon):
            # act: greedy among K candidates, then add exec noise
            Cs = learner.sample_candidates(obs, K=learner.cfg.num_particles)
            qs = learner.score_actions(obs, Cs)
            j = int(np.argmax(qs))
            c_star = Cs[j]
            c_exec = rfm_service.perturb(c_star, kappa=learner.cfg.kappa_exec, J=1)[0, 0]

            # step env
            action = linear_solver(c_exec)
            out = env.step(action)
            next_obs, rew, done, _ = _step_unpack(out)
            next_obs = _flat(next_obs)
            r_scalar = float(np.sum(rew)) if isinstance(rew, (list, np.ndarray)) else float(rew)

            # push transition
            buffer.add(obs, c_exec, r_scalar, next_obs, float(done),
                       coeff_star=c_star, policy_id=learner.policy_version)

            # one update per step
            MIN_WARM_BATCHES = 3 * batch_size
            if buffer.size_filled >= max(batch_size, MIN_WARM_BATCHES):
                b = buffer.sample(batch_size)
                info = train_one_step(learner, b)
                loss_history["q1"].append(info["q1_loss"])
                loss_history["q2"].append(info["q2_loss"])
                loss_history["policy"].append(info["policy_loss"])

            obs = next_obs
            if done:
                break

        # version bump once per episode
        learner.policy_version += 1

    # -------- Save loss plot --------
    plt.figure(figsize=(6,4))
    # plt.plot(loss_history["q1"], label="Q1 loss")
    # plt.plot(loss_history["q2"], label="Q2 loss")
    plt.plot(loss_history["policy"], label="Policy (RFM) loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("DPMD-RF Training Losses (episodic)")
    plt.tight_layout()
    plt.savefig(f"loss_curves/dpmd_rf_losses_{ts}.png", dpi=200)
    plt.close()

    # -------- Greedy evaluation --------
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
