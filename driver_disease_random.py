#!/usr/bin/env python3
"""
Driver for Disease Graph Random Testing

Random policy:
  - At each step, choose up to `budget` random nodes from the current frontier
    (via env.random_feasible_action()).

Plot metric (matches paper):
  x = fraction of population tested
  y = fraction of positive cases detected (undiscounted)

Saves:
  - summary.csv
  - trajectories.csv
  - final_summary.csv
  - PNG plot (curve + interaction graph)
"""

import os
import sys
import argparse
import datetime
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import stats
import torch
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import patches
from tqdm import tqdm

from environment.disease_graph_loader import (
    load_disease_graph_instance,
    create_disease_env,
)

RESULTS_ROOT = "./results"


def run_random_budget(env, init_states, horizon=None):
    """
    Random policy respecting env.budget.

    Returns dict with:
      rewards      : (E*T,) positives found per step
      cum_pos      : (E*T,) cumulative positives found (undiscounted)
      cum_tests    : (E*T,) cumulative tests performed
      total_pos    : (E,) total positives in hidden world per episode
      horizon      : T
      n_episodes   : E
    """
    E = len(init_states)
    if horizon is None:
        horizon = env.n

    rewards = np.zeros(E * horizon, dtype=float)
    cum_pos = np.zeros(E * horizon, dtype=float)
    cum_tests = np.zeros(E * horizon, dtype=float)
    total_pos = np.zeros(E, dtype=float)

    with tqdm(
        total=E, desc="random", file=sys.stdout, mininterval=1.0, ncols=100
    ) as pbar:
        for ep in range(E):
            status, _ = env.reset()
            total_pos[ep] = float(np.sum(env.world_X))

            # ensure identical initial visible states
            if init_states[ep] is not None:
                env.status = init_states[ep].copy()
                status = env.status.copy()

            ep_cum_pos = float(np.sum(status == 1))
            ep_cum_tests = float(np.sum(status != -1))

            last_reward = 0.0

            for t in range(horizon):
                # random feasible batch action (respects budget)
                action = env.random_feasible_action().astype(int)
                k = int(action.sum())

                next_status, _, reward, done = env.step(action)

                ep_cum_pos += float(reward)
                ep_cum_tests += float(k)

                idx = ep * horizon + t
                rewards[idx] = float(reward)
                cum_pos[idx] = ep_cum_pos
                cum_tests[idx] = ep_cum_tests

                status = next_status
                last_reward = float(reward)

                if done:
                    if t + 1 < horizon:
                        rewards[idx + 1 : ep * horizon + horizon] = 0.0
                        cum_pos[idx + 1 : ep * horizon + horizon] = ep_cum_pos
                        cum_tests[idx + 1 : ep * horizon + horizon] = ep_cum_tests
                    break

            pbar.update(1)
            pbar.set_postfix({"reward": f"{last_reward:.3f}"})

    return {
        "rewards": rewards,
        "cum_pos": cum_pos,
        "cum_tests": cum_tests,
        "total_pos": total_pos,
        "horizon": horizon,
        "n_episodes": E,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-D", "--std_name", type=str, default="HIV")
    parser.add_argument("-T", "--cc_threshold", type=int, default=300)
    parser.add_argument("-I", "--inst_idx", type=int, default=0)
    parser.add_argument("-B", "--budget", type=int, default=5)
    parser.add_argument("-G", "--discount", type=float, default=0.99)
    parser.add_argument("-V", "--n_episodes_eval", type=int, default=50)
    parser.add_argument("-p", "--prefix", type=str, default="")
    parser.add_argument(
        "--load_graph_from",
        type=str,
        default=None,
        help="Path to cached graph pickle file. If provided, skips graph generation.",
    )
    args = parser.parse_args()

    start_time = datetime.datetime.now()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(RESULTS_ROOT):
        os.makedirs(RESULTS_ROOT)

    # -------- Load graph --------
    if args.load_graph_from is not None:
        # Load from cache
        print(f"Loading graph from cache: {args.load_graph_from}")
        print(
            f"  Note: Ignoring cc_threshold={args.cc_threshold}, inst_idx={args.inst_idx}"
        )

        from environment.disease_graph_loader import load_graph_cache

        G, covariates, theta_unary, theta_pairwise, statuses = load_graph_cache(
            cache_path=args.load_graph_from, expected_std_name=args.std_name
        )
    else:
        # Standard graph generation
        from environment.disease_graph_loader import save_graph_cache

        G, covariates, theta_unary, theta_pairwise, statuses = (
            load_disease_graph_instance(
                std_name=args.std_name,
                cc_threshold=args.cc_threshold,
                inst_idx=args.inst_idx,
            )
        )

        # Save to cache (always overwrite if exists)
        cache_path = save_graph_cache(
            G=G,
            covariates=covariates,
            theta_unary=theta_unary,
            theta_pairwise=theta_pairwise,
            statuses=statuses,
            std_name=args.std_name,
            inst_idx=args.inst_idx,
            cc_threshold=args.cc_threshold,
        )
        print(f"Graph cached to: {cache_path}")

    # -------- Environment --------
    env = create_disease_env(
        G,
        covariates,
        theta_unary,
        theta_pairwise,
        budget=args.budget,
        discount_factor=args.discount,
        rng_seed=args.seed,
    )

    horizon = env.n

    # -------- Initial states --------
    init_states = []
    for _ in range(args.n_episodes_eval):
        s, _ = env.reset()
        init_states.append(s)

    # -------- Run random --------
    out = run_random_budget(env, init_states, horizon=horizon)

    # -------- Output directory --------
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(
        RESULTS_ROOT,
        f"{args.prefix}{args.std_name}_T{args.cc_threshold}_B{args.budget}_seed{args.seed}_{ts}",
    )
    os.makedirs(run_dir, exist_ok=True)

    # -------- CSVs --------
    E = out["n_episodes"]
    T = out["horizon"]
    rewards_2d = out["rewards"].reshape(E, T)

    discounts = (args.discount ** np.arange(T))[None, :]
    disc_cum = (rewards_2d * discounts).cumsum(axis=1)
    final_disc = disc_cum[:, -1]

    summary_df = pd.DataFrame(
        [
            {
                "seed": args.seed,
                "std_name": args.std_name,
                "cc_threshold": args.cc_threshold,
                "inst_idx": args.inst_idx,
                "n_nodes": env.n,
                "budget": args.budget,
                "n_episodes_eval": args.n_episodes_eval,
                "algo": "random",
                "disc_mean": float(final_disc.mean()),
                "disc_sem": float(stats.sem(final_disc)) if E > 1 else 0.0,
                "timestamp": ts,
            }
        ]
    )
    summary_df.to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    traj_rows = []
    cum_pos_2d = out["cum_pos"].reshape(E, T)
    cum_tests_2d = out["cum_tests"].reshape(E, T)
    for ep in range(E):
        denom = max(out["total_pos"][ep], 1.0)
        for t in range(T):
            traj_rows.append(
                {
                    "episode": ep,
                    "step": t,
                    "reward": rewards_2d[ep, t],
                    "discounted_cum_reward": disc_cum[ep, t],
                    "cum_tests": cum_tests_2d[ep, t],
                    "cum_positives": cum_pos_2d[ep, t],
                    "frac_tested": cum_tests_2d[ep, t] / env.n,
                    "frac_detected": cum_pos_2d[ep, t] / denom,
                }
            )
    pd.DataFrame(traj_rows).to_csv(
        os.path.join(run_dir, "trajectories.csv"), index=False
    )

    final_summary_df = pd.DataFrame(
        [
            {
                "seed": args.seed,
                "std_name": args.std_name,
                "cc_threshold": args.cc_threshold,
                "inst_idx": args.inst_idx,
                "n_nodes": env.n,
                "budget": args.budget,
                "n_episodes_eval": args.n_episodes_eval,
                "total_runtime_seconds": (
                    datetime.datetime.now() - start_time
                ).total_seconds(),
                "random_disc_mean": float(final_disc.mean()),
            }
        ]
    )
    final_summary_df.to_csv(os.path.join(run_dir, "final_summary.csv"), index=False)

    # -------- Plot --------
    mean_x = np.mean(cum_tests_2d / env.n, axis=0)
    mean_y = np.mean(
        [cum_pos_2d[ep] / max(out["total_pos"][ep], 1.0) for ep in range(E)], axis=0
    )

    fig, (ax_curve, ax_graph) = plt.subplots(1, 2, figsize=(12, 5))

    ax_curve.plot(mean_x, mean_y, lw=2, linestyle="dotted", label="random")
    ax_curve.axvline(0.5, color="gray", linestyle="--")
    ax_curve.set_xlabel("Fraction of population tested")
    ax_curve.set_ylabel("Fraction of positive cases detected")
    ax_curve.set_xlim(0, 1)
    ax_curve.set_ylim(0, 1)
    ax_curve.set_title("Random")

    pos = nx.spring_layout(G, seed=args.seed)
    roots = set(env.cc_root)
    nx.draw(
        G,
        pos,
        node_color=["red" if i in roots else "blue" for i in G.nodes()],
        node_size=10,
        ax=ax_graph,
        with_labels=False,
    )
    for r in roots:
        if r in pos:
            ax_graph.add_patch(
                patches.Circle(pos[r], 0.05, fill=False, edgecolor="red", lw=2)
            )
    ax_graph.set_title(f"{args.std_name} sex interaction graph")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            run_dir,
            f"random_{args.std_name}_T{args.cc_threshold}_B{args.budget}_seed{args.seed}.png",
        ),
        dpi=300,
    )
    plt.close("all")

    print(f"✓ Random (budget-respecting) run saved to {run_dir}")


if __name__ == "__main__":
    main()
