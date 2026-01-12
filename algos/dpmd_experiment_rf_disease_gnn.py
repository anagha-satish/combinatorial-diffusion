# algos/dpmd_experiment_rf_disease_gnn.py
from __future__ import annotations

import os
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def evaluate_detection_curve(
    learner,
    env,
    linear_solver,
    n_episodes_eval: int = 25,
    gamma: float = 0.99,
):
    """
    Evaluate a trained DPMD policy on the disease environment.

    Returns:
        x: [T] mean fraction of population tested
        y: [T] mean fraction of positive cases detected (normalized)
        y_std: [T] std of fraction detected across episodes
        traj_rows: list of dicts for CSV export (matching random driver format)
    """
    n = getattr(env, "num_nodes", getattr(env, "n", None))
    if n is None:
        raise AttributeError("Env must have attribute `num_nodes` or `n`.")

    B = int(getattr(env, "budget", 1))
    max_steps = int(np.ceil(n / B)) + 1

    all_tested = []
    all_detected = []
    traj_rows = []

    for ep in range(n_episodes_eval):
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs = reset_out[0]
        else:
            obs = reset_out
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        step_rewards = []
        cum_tests_list = []
        cum_pos_list = []
        tested_frac = []
        cum_pos = 0.0

        for _ in range(max_steps):
            Cs = learner.sample_candidates(obs, K=learner.cfg.num_particles)
            qs = learner.score_actions(obs, Cs)
            j = int(np.argmax(qs))
            c_star = Cs[j]
            action = linear_solver(c_star)

            step_out = env.step(action)

            if isinstance(step_out, tuple) and len(step_out) == 4:
                a, b, c, d = step_out
                if isinstance(b, np.ndarray) and b.shape == np.asarray(a).shape:
                    next_obs = a
                    reward = c
                    done = d
                else:
                    next_obs = a
                    reward = b
                    done = c
            elif isinstance(step_out, tuple) and len(step_out) == 5:
                next_obs, reward, terminated, truncated, _info = step_out
                done = bool(terminated or truncated)
            else:
                try:
                    next_obs, reward, done = step_out[0], step_out[1], step_out[2]
                except Exception:
                    break

            r_scalar = float(reward)
            cum_pos += r_scalar
            obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)

            if hasattr(env, "tests_done"):
                tests_done = float(env.tests_done)
            else:
                status = obs
                tests_done = float(np.count_nonzero(status > -0.5))

            step_rewards.append(r_scalar)
            cum_tests_list.append(tests_done)
            cum_pos_list.append(cum_pos)
            tested_frac.append(tests_done / float(n))

            if done:
                break

        if len(step_rewards) == 0:
            step_rewards = [0.0]
            cum_tests_list = [0.0]
            cum_pos_list = [0.0]
            tested_frac = [0.0]

        total_pos = max(cum_pos_list[-1], 1.0)
        detected_frac = [cp / total_pos for cp in cum_pos_list]

        # Compute discounted cumulative reward
        discounts = gamma ** np.arange(len(step_rewards))
        disc_cum = np.cumsum(np.array(step_rewards) * discounts)

        # Build trajectory rows for this episode
        for t in range(len(step_rewards)):
            traj_rows.append(
                {
                    "episode": ep,
                    "step": t,
                    "reward": step_rewards[t],
                    "discounted_cum_reward": disc_cum[t],
                    "cum_tests": cum_tests_list[t],
                    "cum_positives": cum_pos_list[t],
                    "frac_tested": tested_frac[t],
                    "frac_detected": detected_frac[t],
                }
            )

        # Pad to max_steps for averaging
        while len(tested_frac) < max_steps:
            tested_frac.append(tested_frac[-1])
            detected_frac.append(detected_frac[-1])

        all_tested.append(tested_frac)
        all_detected.append(detected_frac)

    all_tested = np.array(all_tested)
    all_detected = np.array(all_detected)

    x = all_tested.mean(axis=0)
    y = all_detected.mean(axis=0)
    y_std = all_detected.std(axis=0)

    return x, y, y_std, traj_rows


def save_eval_results(
    x: np.ndarray,
    y: np.ndarray,
    y_std: np.ndarray,
    traj_rows: list,
    results_dir: str,
    episode: int,
):
    """Save detection curve plot and trajectory CSV."""
    traj_path = f"{results_dir}/trajectories_ep{episode}.csv"
    pd.DataFrame(traj_rows).to_csv(traj_path, index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, linestyle="-", color="tab:blue", label="DPMD-RF")
    plt.fill_between(x, y - y_std, y + y_std, color="tab:blue", alpha=0.25)
    plt.axvline(x=0.5, linestyle=":", color="gray", alpha=0.7)
    plt.xlabel("Fraction of population tested")
    plt.ylabel("Fraction of positive cases detected")
    plt.title(f"Detection Curve (Episode {episode})")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    graph_path = f"{results_dir}/detection_curve_ep{episode}.png"
    plt.savefig(graph_path, dpi=200)
    plt.close()

    print(f"  [eval] saved {graph_path}, {traj_path}")


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
    eval_cooldown_episodes: int = 5,
    results_dir: str = "results",
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
            C = learner.sample_candidates(status, K=cfg.num_particles)  # [K,n]
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

    # Training AUC tracking
    max_training_auc: float = 0.0
    last_eval_episode: int = -eval_cooldown_episodes  # Allow eval from episode 0
    episode_aucs: List[float] = []

    # CSV logging for training AUC
    os.makedirs(results_dir, exist_ok=True)
    auc_csv_path = f"{results_dir}/training_auc.csv"
    with open(auc_csv_path, "w") as f:
        f.write("episode,train_auc,train_auc_MA10,eval_auc\n")

    for _ep in range(int(train_updates)):
        status = np.asarray(_reset_obs(env), dtype=np.float32).reshape(-1)
        ep_reward = 0.0

        # Track per-step metrics for training AUC
        ep_cum_tests = 0
        ep_cum_positives = 0.0
        ep_tested_list = []
        ep_positives_list = []

        for _t in range(horizon):
            C = learner.sample_candidates(status, K=cfg.num_particles)
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

            # Track metrics for training AUC
            ep_cum_tests += budget
            ep_cum_positives += r_scalar
            ep_tested_list.append(ep_cum_tests)
            ep_positives_list.append(ep_cum_positives)

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

        # Compute training AUC for this episode
        total_positives = ep_cum_positives if ep_cum_positives > 0 else 1.0
        ep_tested_frac = np.array(ep_tested_list) / n
        ep_detected_frac = np.array(ep_positives_list) / total_positives
        ep_auc = (
            np.trapezoid(ep_detected_frac, ep_tested_frac)
            if len(ep_tested_frac) > 1
            else 0.0
        )
        episode_aucs.append(ep_auc)

        episode_rewards.append(ep_reward)
        learner.policy_version += 1

        # Compute moving average of training AUC (for logging and eval trigger)
        avg_training_auc = (
            np.mean(episode_aucs[-10:]) if len(episode_aucs) >= 10 else 0.0
        )

        # Check for new max training AUC and run eval if cooldown allows
        # Only trigger if we have enough episodes for a meaningful moving average
        eval_auc_value = None
        episodes_since_last_eval = _ep - last_eval_episode
        if (
            len(episode_aucs) >= 10
            and avg_training_auc > max_training_auc
            and episodes_since_last_eval >= eval_cooldown_episodes
        ):
            max_training_auc = avg_training_auc
            last_eval_episode = _ep

            os.makedirs(results_dir, exist_ok=True)
            x, y, y_std, traj_rows = evaluate_detection_curve(
                learner, env, linear_solver, n_episodes_eval=10, gamma=cfg.gamma
            )
            eval_auc_value = np.trapezoid(y, x)
            save_eval_results(x, y, y_std, traj_rows, results_dir, _ep + 1)
            print(
                f"  [new max] train_auc_MA10={avg_training_auc:.4f} -> eval_auc={eval_auc_value:.4f}"
            )

        # Append to CSV after each episode (include eval_auc if computed)
        eval_auc_str = f"{eval_auc_value:.6f}" if eval_auc_value is not None else ""
        with open(auc_csv_path, "a") as f:
            f.write(f"{_ep},{ep_auc:.6f},{avg_training_auc:.6f},{eval_auc_str}\n")

        if (_ep + 1) % log_every_n_episodes == 0:
            os.makedirs(results_dir, exist_ok=True)

            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            avg_q = np.mean(recent_q_values[-50:]) if recent_q_values else 0.0
            loss_strs = " | ".join(
                f"{k}={np.mean(v[-50:]):.4f}" for k, v in recent_losses.items() if v
            )
            print(
                f"[{_timestamp()}] episode={_ep + 1} | "
                f"avg_reward={avg_reward:.4f} | train_auc_MA10={avg_training_auc:.4f} | avg_q={avg_q:.4f}"
                + (f" | {loss_strs}" if loss_strs else "")
            )

    print(
        f"[{_timestamp()}] Training complete | total_steps={total_steps} | "
        f"final_avg_reward={np.mean(episode_rewards[-10:]):.4f} | "
        f"final_train_auc_MA10={np.mean(episode_aucs[-10:]):.4f}"
    )

    # -------- Greedy eval (no exec noise) --------
    rewards_all: List[float] = []
    for _ in range(n_episodes_eval):
        status = np.asarray(_reset_obs(env), dtype=np.float32).reshape(-1)

        for t in range(horizon):
            C = learner.sample_candidates(status, K=cfg.num_particles)
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
