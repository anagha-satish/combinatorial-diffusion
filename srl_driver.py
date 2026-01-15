#!/usr/bin/env python3
"""
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import random
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import optim
import matplotlib.pyplot as plt

from environment.disease_graph_loader import load_disease_graph_instance, create_disease_env
from approximator.batch_graph_approximator import BatchGraphApproximator

from algos.graph_actor import GraphActorPhi
from algos.graph_critic import DoubleGraphCritic
from algos.disease_graph import DiseaseGraphBuilder
from algos.buffer import ReplayBuffer
from algos.algo import train_step_double
from algos.co_layer import make_constraint_solver_from_approximator


# ------------------------------------------------------------------
# Env wrapper to match DPMD-style interface
# ------------------------------------------------------------------
class DiseaseSRLEnv:
    def __init__(self, base_env):
        self.base = base_env
        self.n_arms = base_env.num_nodes
        self.budget = base_env.budget
        self.discount_factor = base_env.discount_factor
        self.num_nodes = base_env.num_nodes
        self._last_mask = None

    def reset(self):
        status, mask = self.base.reset()
        self._last_mask = mask
        return status.copy()

    def step(self, action_vec):
        status, mask, reward, done = self.base.step(action_vec)
        self._last_mask = mask
        info = {}
        return status.copy(), float(reward), bool(done), info

    def get_mask(self):
        if self._last_mask is not None:
            return self._last_mask
        return self.base.allowed_mask()

    def observation(self):
        return self.base.observation()

    def allowed_mask(self):
        return self.base.allowed_mask()

    def random_feasible_action(self):
        return self.base.random_feasible_action()

    def project_to_feasible(self, a):
        return self.base.project_to_feasible(a)

    @property
    def tests_done(self):
        return self.base.tests_done

    @property
    def world_X(self):
        return self.base.world_X


# ------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------
def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def reseed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_obs(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def to_mask(m) -> np.ndarray:
    m = np.asarray(m).reshape(-1)
    if m.dtype == np.bool_:
        return m.astype(np.float32)
    return (m > 0.5).astype(np.float32)


# ------------------------------------------------------------------
# Constraint solver (same approximator used by DPMD)
# ------------------------------------------------------------------
def build_constraint_solver_for_disease(env: DiseaseSRLEnv, device: torch.device):
    approx = BatchGraphApproximator(env)
    return make_constraint_solver_from_approximator(approx, device=device)


# ------------------------------------------------------------------
# DPMD-style evaluation for SRL (metric definition matches exactly)
# ------------------------------------------------------------------
def evaluate_detection_curve_srl_dpmdstyle(
    actor: GraphActorPhi,
    env: DiseaseSRLEnv,
    graph_builder: DiseaseGraphBuilder,
    device: torch.device,
    k: int,
    n_episodes_eval: int = 10,
    gamma: float = 0.99,
    constraint_solver=None,
):
    """
    Match DPMD evaluate_detection_curve():
      - max_steps = ceil(n/B)+1
      - tests_done comes from env.tests_done
      - cum_pos is sum of rewards
      - frac_detected is cum_pos / final_cum_pos (policy-normalized)
      - record discounted_cum_reward for CSV parity
    """
    n = getattr(env, "num_nodes", getattr(env.base, "n", None))
    if n is None:
        raise AttributeError("Env must have attribute `num_nodes` or `n`.")
    n = int(n)

    B = int(getattr(env, "budget", 1))
    max_steps = int(np.ceil(n / B)) + 1

    all_tested = []
    all_detected = []
    traj_rows = []

    for ep in range(n_episodes_eval):
        obs = to_obs(env.reset())

        step_rewards: List[float] = []
        cum_tests_list: List[float] = []
        cum_pos_list: List[float] = []
        tested_frac: List[float] = []
        cum_pos = 0.0

        for t in range(max_steps):
            batch_s = graph_builder.batch_from_status_batch(obs.reshape(1, -1)).to(device)
            with torch.no_grad():
                mask_np = to_mask(env.get_mask())
                mask_t = torch.as_tensor(mask_np, device=device, dtype=torch.float32).unsqueeze(0)
                a = actor.act_greedy(batch_s, k=k, mask=mask_t, constraint_solver=constraint_solver)

            action = (a.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.float32)
            action = env.project_to_feasible(action).astype(np.float32)

            next_obs, r, done, _ = env.step(action)
            obs = to_obs(next_obs)

            r_scalar = float(r)
            cum_pos += r_scalar

            tests_done = float(env.tests_done)

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

        total_pos = max(cum_pos_list[-1], 1.0)  # EXACT DPMD normalization
        detected_frac = [cp / total_pos for cp in cum_pos_list]

        discounts = gamma ** np.arange(len(step_rewards))
        disc_cum = np.cumsum(np.array(step_rewards) * discounts)

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
    run_tag: str = "",
    label: str = "SRL",
):
    os.makedirs(results_dir, exist_ok=True)

    traj_path = f"{results_dir}/trajectories_ep{episode}_{run_tag}.csv"
    pd.DataFrame(traj_rows).to_csv(traj_path, index=False)

    # Add (0,0) start like DPMD
    x2 = np.concatenate([[0.0], x])
    y2 = np.concatenate([[0.0], y])
    ystd2 = np.concatenate([[0.0], y_std])

    plt.figure(figsize=(8, 4))
    plt.plot(x2, y2, linestyle="-", label=label)
    plt.fill_between(x2, y2 - ystd2, y2 + ystd2, alpha=0.25)
    plt.axvline(x=0.5, linestyle=":", color="gray", alpha=0.7)
    plt.xlabel("Fraction of population tested")
    plt.ylabel("Fraction of positive cases detected (normalized)")
    plt.title(f"Detection Curve (Episode {episode})")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()

    graph_path = f"{results_dir}/detection_curve_ep{episode}_{run_tag}.png"
    plt.savefig(graph_path, dpi=200)
    plt.close()


def save_final_eval_results(
    x: np.ndarray,
    y: np.ndarray,
    y_std: np.ndarray,
    traj_rows: list,
    results_dir: str,
    run_tag: str,
    gamma: float,
    label: str = "SRL",
):
    os.makedirs(results_dir, exist_ok=True)

    npz_path = f"{results_dir}/eval_results_{run_tag}.npz"
    np.savez(npz_path, x=x, y=y, y_std=y_std)
    print("Saved eval vectors to:", npz_path)

    traj_path = f"{results_dir}/trajectories_{run_tag}.csv"
    pd.DataFrame(traj_rows).to_csv(traj_path, index=False)
    print("Saved trajectories to:", traj_path)

    x2 = np.concatenate([[0.0], x])
    y2 = np.concatenate([[0.0], y])
    ystd2 = np.concatenate([[0.0], y_std])

    plt.figure(figsize=(8, 4))
    plt.plot(x2, y2, linestyle="-", label=label)
    plt.fill_between(x2, y2 - ystd2, y2 + ystd2, alpha=0.25)
    plt.axvline(x=0.5, linestyle=":", color="gray", alpha=0.7)
    plt.xlabel("Fraction of population tested")
    plt.ylabel("Fraction of positive cases detected (normalized)")
    plt.title(f"Policies with discount = {gamma}")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()

    png_path = f"{results_dir}/disease_detection_curve_{run_tag}.png"
    plt.savefig(png_path, dpi=200)
    print("Saved plot to:", png_path)
    plt.close()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SRL baseline on disease graphs (DPMD-comparable)")

    parser.add_argument("-D", "--std_name", type=str, default="HIV",
                        help="{HIV, Gonorrhea, Chlamydia, Syphilis, Hepatitis}")
    parser.add_argument("-T", "--cc_threshold", type=int, default=300,
                        help="minimum nodes from connected components")
    parser.add_argument("-I", "--inst_idx", type=int, default=0,
                        help="instance index controlling CC sampling")
    parser.add_argument("-B", "--budget", type=int, default=5,
                        help="batch size per step")
    parser.add_argument("-V", "--n_episodes_eval", type=int, default=10,
                        help="episodes to evaluate after training")
    parser.add_argument("-s", "--seed", type=int, default=0)

    # Core RL hyperparams (keep aligned where applicable)
    parser.add_argument("--gamma", type=float, default=0.99)

    # SRL-specific training budgets (episode-based like DPMD)
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="steps of random/exploration before updates begin")
    parser.add_argument("--train_updates", type=int, default=2000,
                        help="number of training episodes (matches DPMD train_updates semantics)")
    parser.add_argument("--batch_size", type=int, default=64)

    # Logging and evaluation
    parser.add_argument("--log_every", type=int, default=10,
                        help="Evaluate + save plot every N episodes (like DPMD)")
    parser.add_argument("--eval_cooldown", type=int, default=1,
                        help="(kept for parity; SRL runs eval on log_every cadence)")

    # Model + algo defaults (keep your previous defaults)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=3e-4)
    parser.add_argument("--polyak", type=float, default=0.995)
    parser.add_argument("--buffer_size", type=int, default=200_000)

    # CO-layer / SRL knobs
    parser.add_argument("--k", type=int, default=None, help="defaults to budget if None")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--sigma_b", type=float, default=0.5)
    parser.add_argument("--sigma_f", type=float, default=0.1)

    # Device + results
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory for saving plots and evaluation results")

    args = parser.parse_args()

    reseed_all(args.seed)

    # Unique run tag
    k_val = args.k if args.k is not None else args.budget
    run_tag = (
        f"{args.std_name}"
        f"_T{args.cc_threshold}"
        f"_B{args.budget}"
        f"_gamma{args.gamma}"
        f"_seed{args.seed}"
        f"_train{args.train_updates}"
        f"_k{k_val}"
        f"_m{args.m}"
    )

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print("--------------------------------------------------------")
    print("Load Disease Graph")
    print("--------------------------------------------------------")

    G, covariates, theta_unary, theta_pairwise, statuses = load_disease_graph_instance(
        std_name=args.std_name,
        cc_threshold=args.cc_threshold,
        inst_idx=args.inst_idx,
    )

    print("graph stats")
    print(f"  disease: {args.std_name}")
    print(f"  nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    print(
        f"  infected: {sum(statuses.values())}/{len(statuses)} "
        f"({100*sum(statuses.values())/len(statuses):.1f}%)"
    )
    print(f"  covariate dim: {len(covariates[0])}")

    print("--------------------------------------------------------")
    print("Create Disease Environment")
    print("--------------------------------------------------------")

    base_env = create_disease_env(
        G,
        covariates,
        theta_unary,
        theta_pairwise,
        budget=args.budget,
        discount_factor=args.gamma,
        rng_seed=args.seed,
    )

    env = DiseaseSRLEnv(base_env)

    # IMPORTANT: match DPMD episode horizon
    horizon = int(env.base.n)
    print(f"environment: n={env.base.n}, budget={env.budget}")
    print(f"Using device: {device}")

    print("--------------------------------------------------------")
    print("Build constraint solver + graph builder")
    print("--------------------------------------------------------")

    constraint_solver = build_constraint_solver_for_disease(env, device)
    graph_builder = DiseaseGraphBuilder.from_env(env, device=device)

    print("--------------------------------------------------------")
    print("Init SRL actor/critic")
    print("--------------------------------------------------------")

    actor = GraphActorPhi(
        node_in_dim=3,
        edge_in_dim=4,
        hidden=args.hidden,
        layers=3,
    ).to(device)

    critic = DoubleGraphCritic(
        node_in_dim=3,
        edge_in_dim=4,
        hidden=args.hidden,
        layers=3,
    ).to(device)

    target_critic = DoubleGraphCritic(
        node_in_dim=3,
        edge_in_dim=4,
        hidden=args.hidden,
        layers=3,
    ).to(device)
    target_critic.load_state_dict(critic.state_dict())

    opt_actor = optim.Adam(actor.parameters(), lr=args.lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=args.lr_critic)

    buffer = ReplayBuffer(capacity=args.buffer_size)

    os.makedirs(args.results_dir, exist_ok=True)
    auc_csv_path = f"{args.results_dir}/training_auc_{run_tag}.csv"
    with open(auc_csv_path, "w") as f:
        f.write("episode,train_auc,train_auc_MA10,eval_auc\n")

    print("--------------------------------------------------------")
    print("Train + Evaluate SRL baseline")
    print("--------------------------------------------------------")

    # Best eval AUC tracking
    best_eval_auc: float = 0.0
    best_eval_episode: int = 0

    # Track training AUC (optional parity with DPMD logs)
    episode_aucs: List[float] = []

    # Training loop: run for train_updates episodes
    t0 = time.time()
    for ep in range(1, int(args.train_updates) + 1):
        obs = to_obs(env.reset())
        done = False

        # per-episode tracking for "training AUC"
        ep_cum_tests = 0.0
        ep_cum_pos = 0.0
        ep_tested_list = []
        ep_pos_list = []

        # episode rollout
        for t in range(horizon):
            mask_np = to_mask(env.get_mask())
            batch_s = graph_builder.batch_from_status_batch(obs.reshape(1, -1)).to(device)

            # exploration action for data collection
            with torch.no_grad():
                mask_t = torch.as_tensor(mask_np, device=device, dtype=torch.float32).unsqueeze(0)
                a = actor.act_with_noise(
                    batch_s,
                    k=k_val,
                    sigma_f=args.sigma_f,
                    mask=mask_t,
                    constraint_solver=constraint_solver,
                )

            action = (a.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.float32)
            action = env.project_to_feasible(action).astype(np.float32)

            next_obs, r, done, _ = env.step(action)
            next_obs = to_obs(next_obs)

            r_scalar = float(r)

            # track per-step metrics for training AUC
            ep_cum_tests += float(args.budget)
            ep_cum_pos += r_scalar
            ep_tested_list.append(ep_cum_tests)
            ep_pos_list.append(ep_cum_pos)

            # store transition
            next_mask_np = to_mask(env.get_mask())
            buffer.push(
                {
                    "s": obs,
                    "a": action,
                    "r": r_scalar,
                    "s_next": next_obs,
                    "done": float(1.0 if done else 0.0),
                    "mask": mask_np,
                    "mask_next": next_mask_np,
                }
            )

            obs = next_obs
            if done:
                break

            # updates (off-policy)
            if len(buffer) >= args.warmup_steps and len(buffer) >= args.batch_size:
                batch = buffer.sample(args.batch_size, device=device)
                train_step_double(
                    actor_phi=actor,
                    critic=critic,
                    target_critic=target_critic,
                    opt_actor=opt_actor,
                    opt_critic=opt_critic,
                    batch=batch,
                    k=k_val,
                    m=args.m,
                    tau=args.tau,
                    sigma_b=args.sigma_b,
                    sigma_f=args.sigma_f,
                    gamma=args.gamma,
                    polyak=args.polyak,
                    target_update="soft",
                    do_hard_update=False,
                    delay_actor=1,
                    step_idx=(ep * horizon + t),
                    pa_noise="gumbel",
                    constraint_solver=constraint_solver,
                    graph_builder=graph_builder,
                    use_graph=True,
                )

        # compute training AUC for episode
        n_nodes = int(getattr(env, "num_nodes", getattr(env.base, "n", env.n_arms)))
        total_pos = ep_cum_pos if ep_cum_pos > 0 else 1.0
        ep_tested_frac = np.array(ep_tested_list, dtype=float) / float(n_nodes)
        ep_detected_frac = np.array(ep_pos_list, dtype=float) / float(total_pos)
        ep_auc = np.trapezoid(ep_detected_frac, ep_tested_frac) if len(ep_tested_frac) > 1 else 0.0
        episode_aucs.append(float(ep_auc))
        avg_training_auc = float(np.mean(episode_aucs[-10:])) if len(episode_aucs) >= 10 else 0.0

        eval_auc_value: Optional[float] = None

        if ep % int(args.log_every) == 0:
            x, y, y_std, traj_rows = evaluate_detection_curve_srl_dpmdstyle(
                actor=actor,
                env=env,
                graph_builder=graph_builder,
                device=device,
                k=k_val,
                n_episodes_eval=int(args.n_episodes_eval),
                gamma=args.gamma,
                constraint_solver=constraint_solver,
            )
            eval_auc_value = float(np.trapezoid(y, x))

            save_eval_results(
                x=x,
                y=y,
                y_std=y_std,
                traj_rows=traj_rows,
                results_dir=args.results_dir,
                episode=ep,
                run_tag=run_tag,
                label="SRL",
            )

            if eval_auc_value > best_eval_auc:
                best_eval_auc = eval_auc_value
                best_eval_episode = ep

            print(
                f"[{_timestamp()}] episode={ep} | "
                f"train_auc_MA10={avg_training_auc:.4f} -> eval_auc={eval_auc_value:.4f} | "
                f"best_eval_auc={best_eval_auc:.4f} (ep={best_eval_episode})"
            )

        # write to CSV each episode
        eval_auc_str = f"{eval_auc_value:.6f}" if eval_auc_value is not None else ""
        with open(auc_csv_path, "a") as f:
            f.write(f"{ep},{ep_auc:.6f},{avg_training_auc:.6f},{eval_auc_str}\n")

    elapsed = time.time() - t0

    print("--------------------------------------------------------")
    print("Final eval (like DPMD driver does at the end)")
    print("--------------------------------------------------------")

    x, y, y_std, traj_rows = evaluate_detection_curve_srl_dpmdstyle(
        actor=actor,
        env=env,
        graph_builder=graph_builder,
        device=device,
        k=k_val,
        n_episodes_eval=int(args.n_episodes_eval),
        gamma=args.gamma,
        constraint_solver=constraint_solver,
    )
    save_final_eval_results(
        x=x,
        y=y,
        y_std=y_std,
        traj_rows=traj_rows,
        results_dir=args.results_dir,
        run_tag=run_tag,
        gamma=args.gamma,
        label="SRL",
    )

    print("--------------------------------------------------------")
    print("Results")
    print("--------------------------------------------------------")
    print(
        f"{args.std_name} disease testing | n={env.base.n}, budget={args.budget}, gamma={args.gamma}"
    )
    if best_eval_episode > 0:
        print(f"  best eval AUC: {best_eval_auc:.4f} (achieved at episode {best_eval_episode})")
        print(f"  best-episode plot: {args.results_dir}/detection_curve_ep{best_eval_episode}_{run_tag}.png")
        print(f"  best-episode traj: {args.results_dir}/trajectories_ep{best_eval_episode}_{run_tag}.csv")
    print(f"  runtime: {elapsed:.2f} seconds")
    print("[done]")


if __name__ == "__main__":
    main()
