# driver_tune.py
"""
Grid/parallel tuner for DPMD-RF.

Examples
--------
# 1) Simple sweep over two hyperparameters and three seeds:
python driver_tune.py \
  --lr 1e-4 4e-4 1e-3 \
  --num_particles 8 12 16 \
  --seed 0 1 2 \
  --out results.csv --jobs 3

# 2) Change environment and training budgets too:
python driver_tune.py \
  --rmab_type routing \
  --n_arms 20 --n_actions 3 --budget 2 --horizon 10 \
  --warmup_steps 1000 --train_updates 2000 --batch_size 64 \
  --lr 2e-4 4e-4 --num_particles 8 12 \
  --kappa_exec 16 28 --kappa_smooth 16 28 \
  --seed 0 1 --jobs 2

Notes
-----
- Every flag accepts one or many values; a full Cartesian product is run.
- Results are written to CSV (default: tune_results.csv) and printed (top 10 by avg reward).
- Uses the same code paths as driver_dpmd_rf.py (no subprocesses, imports the same modules).
"""

import argparse
import itertools
import json
import math
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import torch

# Import the same building blocks used by your driver
from environment.rmab_instances import (
    get_rmab_sigmoid, get_scheduling, get_constrained, get_routing
)
from algos.repo_bridge import linear_solver_approx
from algos.dpmd_experiment_rf import run_dpmd_only
from algos.dpmd_rf import DPMDConfig


# ----------------------------
# Utilities copied/adapted to match driver defaults
# ----------------------------
def reseed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np_seed = seed % (2**32 - 1)
    import numpy as _np
    _np.random.seed(np_seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def build_env(kind: str, n_arms: int, n_actions: int, budget: int, horizon: int):
    if kind == 'sigmoid':
        return get_rmab_sigmoid(n_arms, n_actions, budget, horizon)
    elif kind == 'constrained':
        return get_constrained(horizon, n_arms, budget, rmab_type='multistate')
    elif kind == 'scheduling':
        return get_scheduling(horizon, n_arms, budget, rmab_type='multistate')
    elif kind == 'routing':
        return get_routing(horizon, n_arms, budget, rmab_type='multistate')
    else:
        raise NotImplementedError(kind)


def build_env_quiet(kind: str, n_arms: int, n_actions: int, budget: int, horizon: int, seed: int):
    # Keep side-effects minimal; silence builders if they print
    import io, contextlib
    reseed_all(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        env, _ = build_env(kind, n_arms, n_actions, budget, horizon)

    if hasattr(env, "seed") and callable(getattr(env, "seed")):
        try:
            env.seed(seed)
        except TypeError:
            pass
    if hasattr(env, "reset_rng") and callable(getattr(env, "reset_rng")):
        try:
            env.reset_rng(seed)
        except TypeError:
            pass
    return env


# ----------------------------
# Trial execution
# ----------------------------
def run_trial(trial_args):
    """
    trial_args: dict of fully-specified arguments for a single run.
    Returns a result dict with metrics + params.
    """
    # Defensive copy (process-safe)
    p = dict(trial_args)
    t0 = time.time()

    try:
        # Build env + solver
        env = build_env_quiet(
            p['rmab_type'], p['n_arms'], p['n_actions'],
            p['budget'], p['horizon'], p['seed']
        )
        linear_solver = linear_solver_approx(env)

        # Pack config
        cfg = DPMDConfig(
            gamma=p['gamma'],
            lr=p['lr'],
            tau=p['tau'],
            delay_update=p['delay_update'],
            reward_scale=p['reward_scale'],
            num_particles=p['num_particles'],
            w_clip=p['w_clip'],
            kappa_exec=p['kappa_exec'],
            kappa_smooth=p['kappa_smooth'],
            M_smooth=p['M_smooth'],
            J_smooth=p['J_smooth'],
            flow_steps=p['flow_steps'],
            lambda_start=p['lambda_start'],
            lambda_end=p['lambda_end'],
            lambda_steps=p['lambda_steps'],
            q_norm_clip=p['q_norm_clip'],
            q_running_beta=p['q_running_beta'],
        )

        rewards = run_dpmd_only(
            env,
            horizon=p['horizon'],
            budget=p['budget'],
            n_episodes_eval=p['n_episodes_eval'],
            seed=p['seed'],
            linear_solver=linear_solver,
            cfg=cfg,
            warmup_steps=p['warmup_steps'],
            batch_size=p['batch_size'],
            train_updates=p['train_updates'],
        )
        elapsed = time.time() - t0

        rewards = list(map(float, rewards))
        avg = float(np.mean(rewards)) if len(rewards) else 0.0
        std = float(np.std(rewards)) if len(rewards) else 0.0

        return {
            "ok": True,
            "avg_reward": avg,
            "std_reward": std,
            "rewards_len": len(rewards),
            "elapsed_sec": round(elapsed, 3),
            "params": p,
            # Store compact reward summary; full list optional (can be large)
            "rewards_head": rewards[:10],
        }

    except Exception as e:
        elapsed = time.time() - t0
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "elapsed_sec": round(elapsed, 3),
            "params": p,
        }


# ----------------------------
# CLI & grid expansion
# ----------------------------
def add_list_arg(parser, name, type_, default, help_):
    # Every flag accepts one or many values
    parser.add_argument(f"--{name}", type=type_, nargs="+", default=[default], help=help_)


def parse_args():
    p = argparse.ArgumentParser(description="Grid/parallel tuner for DPMD-RF")

    # Environment config
    add_list_arg(p, "rmab_type", str, "routing", "{sigmoid, scheduling, constrained, routing}")
    add_list_arg(p, "n_arms", int, 20, "Number of arms")
    add_list_arg(p, "n_actions", int, 3, "Number of actions")
    add_list_arg(p, "budget", int, 2, "Budget per step")
    add_list_arg(p, "horizon", int, 10, "Horizon")
    add_list_arg(p, "n_episodes_eval", int, 5, "Evaluation episodes per run")
    add_list_arg(p, "seed", int, 0, "Random seed(s)")

    # Training budgets
    add_list_arg(p, "warmup_steps", int, 1000, "Steps collected before updates")
    add_list_arg(p, "train_updates", int, 2000, "Training updates (episodes)")
    add_list_arg(p, "batch_size", int, 64, "Batch size")

    # Core RL hyperparameters
    add_list_arg(p, "gamma", float, 0.99, "Discount")
    add_list_arg(p, "tau", float, 0.005, "Target update rate")
    add_list_arg(p, "delay_update", int, 2, "Actor/target delay")
    add_list_arg(p, "reward_scale", float, 1.0, "Reward scale before targets")

    # DPMD / RFM hyperparameters
    add_list_arg(p, "lr", float, 4e-4, "Learning rate")
    add_list_arg(p, "num_particles", int, 12, "K: candidate actions per state")

    # Mirror-descent schedule
    add_list_arg(p, "lambda_start", float, 2.0, "Start temp")
    add_list_arg(p, "lambda_end", float, 0.8, "End temp")
    add_list_arg(p, "lambda_steps", int, 10000, "Temp steps")
    add_list_arg(p, "w_clip", float, 4.0, "Weight clamp")

    # vMF execution noise
    add_list_arg(p, "kappa_exec", float, 28.0, "vMF kappa (execution)")

    # Smoothed Bellman operator
    add_list_arg(p, "kappa_smooth", float, 28.0, "vMF kappa (target smoothing)")
    add_list_arg(p, "M_smooth", int, 16, "Num target-policy candidates M")
    add_list_arg(p, "J_smooth", int, 1, "vMF perturbations per candidate J")

    # RFM integration
    add_list_arg(p, "flow_steps", int, 36, "RFM integration steps")

    # Stability
    add_list_arg(p, "q_norm_clip", float, 3.0, "Clip normalized Q")
    add_list_arg(p, "q_running_beta", float, 0.05, "EMA coeff for Q running stats")

    # Orchestration
    p.add_argument("--jobs", type=int, default=max(1, cpu_count() // 2),
                   help="Parallel worker processes")
    p.add_argument("--out", type=str, default="tune_results.csv",
                   help="Path to CSV output")
    p.add_argument("--save_jsonl", type=str, default="",
                   help="Optional JSONL with full results (one JSON per line)")
    p.add_argument("--topk", type=int, default=10,
                   help="Show top-K in stdout summary")
    p.add_argument("--dry_run", action="store_true",
                   help="Only print expanded grid size, do not execute")

    return p.parse_args()


def expand_grid(ns):
    # Build ordered list of (name, list_of_values)
    grid_spec = []
    for k, v in vars(ns).items():
        # skip orchestration flags
        if k in {"jobs", "out", "save_jsonl", "topk", "dry_run"}:
            continue
        # v is already a list due to nargs="+"
        if isinstance(v, list):
            grid_spec.append((k, v))
    # Cartesian product
    names = [k for k, _ in grid_spec]
    values_lists = [vals for _, vals in grid_spec]
    for combo in itertools.product(*values_lists):
        yield dict(zip(names, combo))


def write_csv(path, rows, header=None):
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if header is None and rows:
        header = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ns = parse_args()
    trials = list(expand_grid(ns))
    total = len(trials)
    print(f"[tuner] Expanded grid: {total} runs (jobs={ns.jobs})")

    if ns.dry_run:
        return

    # Execute (parallel if jobs > 1)
    if ns.jobs > 1:
        with Pool(processes=ns.jobs) as pool:
            results = list(pool.imap_unordered(run_trial, trials))
    else:
        results = [run_trial(t) for t in trials]

    # Sort by avg reward (descending), then elapsed (ascending)
    ok_results = [r for r in results if r.get("ok")]
    bad_results = [r for r in results if not r.get("ok")]

    ok_results.sort(key=lambda r: (r["avg_reward"], -r["elapsed_sec"]), reverse=True)

    # Prepare CSV rows
    csv_rows = []
    for r in ok_results:
        flat = dict(
            ok=r["ok"],
            avg_reward=round(r["avg_reward"], 6),
            std_reward=round(r["std_reward"], 6),
            rewards_len=r["rewards_len"],
            elapsed_sec=r["elapsed_sec"],
        )
        # Flatten selected params into CSV columns
        for k, v in r["params"].items():
            flat[k] = v
        csv_rows.append(flat)

    # Include failures at the end of CSV for visibility
    for r in bad_results:
        flat = dict(
            ok=False,
            avg_reward=float("nan"),
            std_reward=float("nan"),
            rewards_len=0,
            elapsed_sec=r["elapsed_sec"],
            error=r.get("error", "unknown"),
        )
        for k, v in r["params"].items():
            flat[k] = v
        csv_rows.append(flat)

    # Write CSV
    if csv_rows:
        write_csv(ns.out, csv_rows)
        print(f"[tuner] Wrote {len(csv_rows)} rows to {ns.out}")

    # Optional JSONL dump (with full params + small reward head)
    if ns.save_jsonl:
        os.makedirs(os.path.dirname(ns.save_jsonl) or ".", exist_ok=True)
        with open(ns.save_jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"[tuner] Wrote JSONL details to {ns.save_jsonl}")

    # Console summary
    print("\n=== TOP RESULTS ===")
    show_k = min(ns.topk, len(ok_results))
    for i in range(show_k):
        r = ok_results[i]
        p = r["params"]
        print(
            f"#{i+1:02d} avg={r['avg_reward']:.6f} ±{r['std_reward']:.6f} "
            f"(len={r['rewards_len']}, {r['elapsed_sec']:.1f}s) | "
            f"lr={p['lr']} K={p['num_particles']} seed={p['seed']} "
            f"exec_kappa={p['kappa_exec']} smooth_kappa={p['kappa_smooth']} "
            f"rmab={p['rmab_type']}"
        )

    if bad_results:
        print(f"\n[warn] {len(bad_results)} runs failed; last error: {bad_results[-1].get('error')}")


if __name__ == "__main__":
    main()
