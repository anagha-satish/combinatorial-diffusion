# algos/dpmd_experiment_rf_disease_gnn.py
from __future__ import annotations

from datetime import datetime
from typing import List, Tuple
import numpy as np
import torch

from algos.replay_buffer import ReplayBuffer
from algos.dpmd_rf_disease_gnn import DPMDGraphDisease, DPMDGraphConfig, Experience
from models.rfm.service_gnn import rfm_service_gnn


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _reset_obs(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _step_unpack(out):
    if isinstance(out, tuple) and len(out) == 4:
        a, b, c, d = out
        if isinstance(b, np.ndarray) and b.shape == np.asarray(a).shape:
            return a, c, d, {}
        return a, b, c, d
    if isinstance(out, tuple) and len(out) == 5:
        obs, rew, term, trunc, info = out
        return obs, rew, (term or trunc), info
    obs, rew, done, info = out
    return obs, rew, done, info


def build_graph_features_from_env(env) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    status, unary_factors, pairwise_factors = env.base.get_status_and_factors()
    n = len(status)

    node_cov = np.array(
        [np.asarray(f).flatten() for f in unary_factors.values()], dtype=np.float32
    )
    assert node_cov.shape == (n, 2), f"expected node_cov [n,2], got {node_cov.shape}"

    edge_list = [
        [min([int(Xidx[1:]) for Xidx in uv]), max([int(Xidx[1:]) for Xidx in uv])]
        for uv in pairwise_factors.keys()
    ]
    edge_index = np.asarray(edge_list, dtype=np.int64).T
    edge_attr = np.asarray(
        [np.asarray(tbl).flatten() for tbl in pairwise_factors.values()],
        dtype=np.float32,
    )
    assert (
        edge_attr.shape[1] == 4
    ), f"expected edge_attr second dim 4, got {edge_attr.shape}"

    edge_index_dir = np.concatenate([edge_index, edge_index[[1, 0], :]], axis=1)
    edge_attr_dir = np.concatenate([edge_attr, edge_attr], axis=0)
    return node_cov, edge_index_dir, edge_attr_dir


def run_dpmd_rf_disease_gnn(
    env,
    horizon: int,
    budget: int,
    n_episodes_eval: int,
    seed: int,
    linear_solver,
    *,
    cfg: DPMDGraphConfig,
    warmup_steps: int,
    batch_size: int,
    train_updates: int,
    log_every_n_episodes: int = 10,
) -> Tuple[np.ndarray, DPMDGraphDisease]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    node_cov, edge_index, edge_attr = build_graph_features_from_env(env)
    n = env.n_arms

    learner = DPMDGraphDisease(
        n_nodes=n,
        node_covariates=node_cov,
        edge_index=edge_index,
        edge_attr=edge_attr,
        cfg=cfg,
    )

    # init diffusion policy (GNN actor)
    rfm_service_gnn.init(
        node_base_dim=cfg.node_in_dim,  # 3
        edge_in_dim=cfg.edge_in_dim,  # 4
        act_dim=n,
        lr=cfg.lr,  # 1e-4,
        seed=seed,
        force=True,
    )

    # Replay stores status vectors now
    buffer = ReplayBuffer(capacity=100_000, obs_dim=n, act_dim=n)

    # -------- Warmup --------
    steps_collected = 0
    while steps_collected < warmup_steps:
        status = np.asarray(_reset_obs(env), dtype=np.float32).reshape(-1)  # [n]

        for _ in range(horizon):
            C = learner.sample_candidates(
                status, K=cfg.num_particles, use_target=False
            )  # [K,n]
            qs = learner.score_actions(status, C)  # [K]
            c_star = C[int(np.argmax(qs))]

            c_exec = rfm_service_gnn.perturb(
                c_star.reshape(1, -1), kappa=cfg.kappa_exec, J=1
            )[0, 0]

            action = linear_solver(c_exec)
            status2, rew, done, _ = _step_unpack(env.step(action))
            status2 = np.asarray(status2, dtype=np.float32).reshape(-1)

            r_scalar = (
                float(np.sum(rew))
                if isinstance(rew, (list, np.ndarray))
                else float(rew)
            )

            buffer.add(
                status,
                c_exec,
                r_scalar,
                status2,
                float(done),
                coeff_star=c_star,
                policy_id=learner.policy_version,
            )

            status = status2
            steps_collected += 1
            if done or steps_collected >= warmup_steps:
                break

        learner.policy_version += 1

    # -------- tiny myopic pretrain critics --------
    pretrain_critic_iters = min(64, max(8, warmup_steps // max(1, 32 * batch_size)))
    for _ in range(int(pretrain_critic_iters)):
        b = buffer.sample(min(batch_size, buffer.size_filled))
        exp = Experience(*b)
        learner.pretrain_critics_step(exp)

    # -------- Training --------
    total_steps = 0
    episode_rewards: List[float] = []
    recent_losses: dict[str, List[float]] = {}
    recent_q_values: List[float] = []

    for _ep in range(int(train_updates)):
        status = np.asarray(_reset_obs(env), dtype=np.float32).reshape(-1)
        ep_reward = 0.0

        for _t in range(horizon):
            C = learner.sample_candidates(status, K=cfg.num_particles, use_target=False)
            qs = learner.score_actions(status, C)
            c_star = C[int(np.argmax(qs))]
            c_exec = rfm_service_gnn.perturb(
                c_star.reshape(1, -1), kappa=cfg.kappa_exec, J=1
            )[0, 0]

            recent_q_values.append(float(np.mean(qs)))

            action = linear_solver(c_exec)
            status2, rew, done, _ = _step_unpack(env.step(action))
            status2 = np.asarray(status2, dtype=np.float32).reshape(-1)

            r_scalar = (
                float(np.sum(rew))
                if isinstance(rew, (list, np.ndarray))
                else float(rew)
            )
            ep_reward += r_scalar

            buffer.add(
                status,
                c_exec,
                r_scalar,
                status2,
                float(done),
                coeff_star=c_star,
                policy_id=learner.policy_version,
            )

            if buffer.size_filled >= max(batch_size, 3 * batch_size):
                b = buffer.sample(batch_size)
                exp = Experience(*b)
                loss_dict = learner.update(exp)
                if loss_dict is not None:
                    for key, val in loss_dict.items():
                        if key not in recent_losses:
                            recent_losses[key] = []
                        recent_losses[key].append(float(val))

            total_steps += 1

            status = status2
            if done:
                break

        episode_rewards.append(ep_reward)
        learner.policy_version += 1

        if (_ep + 1) % log_every_n_episodes == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            avg_q = np.mean(recent_q_values[-50:]) if recent_q_values else 0.0
            loss_strs = " | ".join(
                f"{k}={np.mean(v[-50:]):.4f}" for k, v in recent_losses.items() if v
            )
            print(
                f"[{_timestamp()}] episode={_ep + 1} | "
                f"avg_reward={avg_reward:.4f} | avg_q={avg_q:.4f} | "
                f"buffer_size={buffer.size_filled}"
                + (f" | {loss_strs}" if loss_strs else "")
            )

    print(
        f"[{_timestamp()}] Training complete | total_steps={total_steps} | "
        f"final_avg_reward={np.mean(episode_rewards[-10:]):.4f}"
    )

    # -------- Greedy eval (no exec noise) --------
    rewards_all: List[float] = []
    for _ in range(n_episodes_eval):
        status = np.asarray(_reset_obs(env), dtype=np.float32).reshape(-1)

        for t in range(horizon):
            C = learner.sample_candidates(status, K=cfg.num_particles, use_target=False)
            qs = learner.score_actions(status, C)
            c_star = C[int(np.argmax(qs))]

            action = linear_solver(c_star)
            status2, r, done, _ = _step_unpack(env.step(action))
            r_scalar = (
                float(np.sum(r)) if isinstance(r, (list, np.ndarray)) else float(r)
            )
            rewards_all.append(r_scalar)

            status = np.asarray(status2, dtype=np.float32).reshape(-1)
            if done:
                rewards_all.extend([0.0] * (horizon - t - 1))
                break

    return np.asarray(rewards_all, dtype=np.float32), learner
