# driver_tune.py
import os, argparse, csv, random, time
from typing import Dict, Any, List, Tuple
import numpy as np

try:
    import torch
except Exception:
    torch = None

from environment.rmab_instances import (
    get_rmab_sigmoid, get_scheduling, get_constrained, get_routing
)
from environment.multi_action import MultiActionRMAB

from algos.repo_bridge import linear_solver_approx
from algos.dpmd_experiment_rf import run_dpmd_only
from algos.dpmd_rf import DPMDConfig


# ----------------------------
# Global reseeding utilities
# ----------------------------
def reseed_all(seed: int) -> None:
    """Reseed Python, NumPy, and (optionally) PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# ----------------------------
# Environment builders
# ----------------------------
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


import sys, io, contextlib

def make_env_and_solver(kind: str, n_arms: int, n_actions: int,
                        budget: int, horizon: int, seed: int):
    """Build fresh env + solver without spamming stdout."""
    reseed_all(seed)

    # Temporarily silence any prints from environment creation
    with contextlib.redirect_stdout(io.StringIO()):
        env_i, t_env_i = build_env(kind, n_arms, n_actions, budget, horizon)

    for e in (env_i, t_env_i):
        if hasattr(e, "seed") and callable(getattr(e, "seed")):
            try:
                e.seed(seed)
            except TypeError:
                pass
        if hasattr(e, "reset_rng") and callable(getattr(e, "reset_rng")):
            try:
                e.reset_rng(seed)
            except TypeError:
                pass

    lin_i = linear_solver_approx(env_i)
    return env_i, t_env_i, lin_i



# ----------------------------
# Search space helpers
# ----------------------------
def _sample_lr(rng: random.Random) -> float:
    # log-uniform over [1e-4, 1e-3]
    e = rng.uniform(-4.0, -3.0)
    return 10.0 ** e

def _sample_int(rng: random.Random, choices: List[int]) -> int:
    return rng.choice(choices)

def _sample_float(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)

def sample_hparams(rng: random.Random) -> Dict[str, Any]:
    return {
        "lr": _sample_lr(rng),                               # 1e-4 .. 1e-3
        "num_particles": _sample_int(rng, [6, 8, 10, 12]),   # K (candidates per state)
        "lambda_temp": _sample_float(rng, 0.3, 1.2),         # λ (MD temperature)
        "kappa_exec": _sample_float(rng, 10.0, 30.0),        # execution noise (vMF)
        "kappa_smooth": _sample_float(rng, 10.0, 30.0),      # smoothing kernel (vMF) for targets
        "M_smooth": _sample_int(rng, [6, 8, 10, 12]),        # # actor samples for V^b_κ
        "J_smooth": _sample_int(rng, [1, 2, 3]),             # # vMF perturbations per actor sample
    }


# ----------------------------
# One evaluation for a given config
# ----------------------------
def run_once(env,
             linear_solver,
             seed: int, horizon: int, budget: int,
             n_episodes_eval: int, hp: Dict[str, Any], train_updates: int, warmup_steps: int,
             batch_size: int = 64, gamma: float = 0.99, tau: float = 0.005,
             delay_update: int = 2, reward_scale: float = 0.2) -> Dict[str, Any]:
    """
    Build a DPMDConfig from sampled hyperparams and run a single training+eval.
    Note: matches current DPMDConfig and run_dpmd_only signatures.
    """
    cfg = DPMDConfig(
        gamma=gamma,
        lr=float(hp["lr"]),
        tau=tau,
        delay_update=delay_update,
        reward_scale=reward_scale,
        num_particles=int(hp["num_particles"]),
        lambda_temp=float(hp["lambda_temp"]),
        w_clip=50.0,
        kappa_exec=float(hp["kappa_exec"]),
        # Smoothed Bellman operator params
        kappa_smooth=float(hp["kappa_smooth"]),
        M_smooth=int(hp["M_smooth"]),
        J_smooth=int(hp["J_smooth"]),
    )

    rewards = run_dpmd_only(
        env,
        horizon=horizon,
        budget=budget,
        n_episodes_eval=n_episodes_eval,
        seed=seed,
        linear_solver=linear_solver,
        cfg=cfg,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        train_updates=train_updates,
    )

    # Objective = mean per-timestep reward
    avg_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
    return {"avg_reward": avg_reward}


# ----------------------------
# Successive Halving Tuner
# ----------------------------
def successive_halving(
    rmab_kind: str,
    n_arms: int,
    n_actions: int,
    budget: int,
    horizon: int,
    base_seed: int,
    n_episodes_eval: int,
    n_trials: int,
    rungs: List[Dict[str, int]],
    results_csv: str,
) -> None:
    rng = random.Random(base_seed)
    os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)

    # CSV header
    with open(results_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trial_id","rung","seed",
            "lr","num_particles","lambda_temp",
            "kappa_exec","kappa_smooth","M_smooth","J_smooth",
            "warmup_steps","train_updates",
            "avg_reward","elapsed_sec",
        ])

    # Initial pool
    trials = [{"trial_id": i, "hp": sample_hparams(rng)} for i in range(n_trials)]

    best_so_far = -1e9
    for rung_idx, budget_spec in enumerate(rungs):
        warmup = int(budget_spec["warmup"])
        updates = int(budget_spec["updates"])
        print(f"\n[rung {rung_idx}] warmup={warmup} updates={updates} | candidates={len(trials)}")

        # Break any residual order bias across rungs
        rng.shuffle(trials)

        scored: List[Dict[str, Any]] = []
        for t in trials:
            trial_seed = base_seed + t["trial_id"] * 31 + rung_idx

            reseed_all(trial_seed)

            # Fresh env + solver per trial
            env_i, t_env_i, lin_i = make_env_and_solver(
                kind=rmab_kind,
                n_arms=n_arms,
                n_actions=n_actions,
                budget=budget,
                horizon=horizon,
                seed=trial_seed,
            )

            hp = t["hp"]
            t0 = time.time()
            metrics = run_once(
                env_i, lin_i,
                seed=trial_seed, horizon=horizon, budget=budget,
                n_episodes_eval=n_episodes_eval, hp=hp,
                train_updates=updates, warmup_steps=warmup
            )
            elapsed = time.time() - t0

            row = {
                "trial_id": t["trial_id"], "rung": rung_idx, "seed": trial_seed,
                "warmup_steps": warmup, "train_updates": updates,
                **hp,
                **metrics,
                "elapsed_sec": elapsed,
            }
            scored.append(row)

            # write row
            with open(results_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    row["trial_id"], row["rung"], row["seed"],
                    row["lr"], row["num_particles"], row["lambda_temp"],
                    row["kappa_exec"], row["kappa_smooth"], row["M_smooth"], row["J_smooth"],
                    row["warmup_steps"], row["train_updates"],
                    row["avg_reward"], row["elapsed_sec"],
                ])

            best_so_far = max(best_so_far, row["avg_reward"])
            print(f"[rung {rung_idx}] trial {t['trial_id']:03d} | "
                  f"avg_reward={row['avg_reward']:.3f} (best={best_so_far:.3f}) | "
                  f"hp={ {k: row[k] for k in ['lr','num_particles','lambda_temp','kappa_exec','kappa_smooth','M_smooth','J_smooth']} } | "
                  f"{elapsed:.1f}s")

        # Promote top fraction to next rung (keep top 1/η, η≈2)
        if rung_idx < len(rungs) - 1:
            scored.sort(key=lambda r: r["avg_reward"], reverse=True)
            keep = max(1, len(scored) // 2)  # halving
            hp_keys = ["lr","num_particles","lambda_temp","kappa_exec","kappa_smooth","M_smooth","J_smooth"]
            trials = [{"trial_id": r["trial_id"], "hp": {k: r[k] for k in hp_keys}} for r in scored[:keep]]

    print(f"[tune] done | results at {results_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-D','--rmab_type', type=str, default='routing',
                    help='{sigmoid, scheduling, constrained, routing}')
    ap.add_argument('-J','--n_arms', type=int, default=20)
    ap.add_argument('-N','--n_actions', type=int, default=3)
    ap.add_argument('-B','--budget', type=int, default=2)
    ap.add_argument('-H','--horizon', type=int, default=10)
    ap.add_argument('-V','--n_episodes_eval', type=int, default=5)
    ap.add_argument('-s','--seed', type=int, default=0)

    ap.add_argument('--n_trials', type=int, default=16,
                    help='initial number of random configs')
    ap.add_argument('--results_dir', type=str, default='./results')
    ap.add_argument('--results_name', type=str, default='tune')

    # rung budgets (small → larger). Feel free to tweak.
    ap.add_argument('--rung0_updates', type=int, default=400)
    ap.add_argument('--rung0_warmup', type=int, default=400)
    ap.add_argument('--rung1_updates', type=int, default=1200)
    ap.add_argument('--rung1_warmup', type=int, default=800)

    args = ap.parse_args()

    env_tmp, _ = build_env(args.rmab_type, args.n_arms, args.n_actions, args.budget, args.horizon)
    if isinstance(env_tmp, MultiActionRMAB):
        print(f"[info] MultiAction wrapper detected: {env_tmp.link_type}")
    del env_tmp  # avoid accidental reuse

    os.makedirs(args.results_dir, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = os.path.join(args.results_dir, f"{args.results_name}_{ts}.csv")

    rungs = [
        {"updates": args.rung0_updates, "warmup": args.rung0_warmup},
        {"updates": args.rung1_updates, "warmup": args.rung1_warmup},
    ]

    print(f"[tune] starting | trials={args.n_trials} | rungs={rungs} | csv={csv_path}")
    successive_halving(
        rmab_kind=args.rmab_type,
        n_arms=args.n_arms,
        n_actions=args.n_actions,
        budget=args.budget,
        horizon=args.horizon,
        base_seed=args.seed,
        n_episodes_eval=args.n_episodes_eval,
        n_trials=args.n_trials,
        rungs=rungs,
        results_csv=csv_path,
    )


if __name__ == "__main__":
    main()
