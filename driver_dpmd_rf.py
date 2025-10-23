# driver_dpmd_rf.py
"""
Driver for DPMD-RF using tuned defaults.
"""
import os
import sys
import io
import time
import argparse
import contextlib
import numpy as np
import torch

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


def build_env_quiet(kind: str, n_arms: int, n_actions: int, budget: int, horizon: int, seed: int):
    reseed_all(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        env, _ = build_env(kind, n_arms, n_actions, budget, horizon)

    # If the env exposes seeding hooks, use them (tolerate missing/odd signatures)
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
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Driver for DPMD-RF with tuned defaults")

    # Environment config
    p.add_argument('-D','--rmab_type', type=str, default='routing',
                   help='{sigmoid, scheduling, constrained, routing}')
    p.add_argument('-J','--n_arms', type=int, default=20)
    p.add_argument('-N','--n_actions', type=int, default=3)
    p.add_argument('-B','--budget', type=int, default=2)
    p.add_argument('-H','--horizon', type=int, default=10)
    p.add_argument('-V','--n_episodes_eval', type=int, default=5)
    p.add_argument('-s','--seed', type=int, default=0)

    # Training budgets
    p.add_argument('--warmup_steps', type=int, default=800)
    p.add_argument('--train_updates', type=int, default=1200)
    p.add_argument('--batch_size', type=int, default=64)

    # Core RL hyperparameters
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--delay_update', type=int, default=2)
    p.add_argument('--reward_scale', type=float, default=0.2)

    # Tuned DPMD/RFM hyperparameters
    p.add_argument('--lr', type=float, default=3.629223204600222e-04)
    p.add_argument('--num_particles', type=int, default=10)
    p.add_argument('--lambda_temp', type=float, default=0.6907546519084053)
    p.add_argument('--kappa_exec', type=float, default=22.217739468876033)
    p.add_argument('--kappa_smooth', type=float, default=28.260221064757964)
    p.add_argument('--M_smooth', type=int, default=12)
    p.add_argument('--J_smooth', type=int, default=2)
    p.add_argument('--w_clip', type=float, default=50.0)

    args = p.parse_args()

    # Build env and linear solver
    env = build_env_quiet(args.rmab_type, args.n_arms, args.n_actions, args.budget, args.horizon, args.seed)
    linear_solver = linear_solver_approx(env)  # maps coefficients (on sphere) -> combinatorial action

    # Pack config
    cfg = DPMDConfig(
        gamma=args.gamma,
        lr=args.lr,
        tau=args.tau,
        delay_update=args.delay_update,
        reward_scale=args.reward_scale,
        num_particles=args.num_particles,
        lambda_temp=args.lambda_temp,
        w_clip=args.w_clip,
        kappa_exec=args.kappa_exec,
        # Smoothed Bellman operator (target smoothing)
        kappa_smooth=args.kappa_smooth,
        M_smooth=args.M_smooth,
        J_smooth=args.J_smooth,
    )

    t0 = time.time()
    rewards = run_dpmd_only(
        env,
        horizon=args.horizon,
        budget=args.budget,
        n_episodes_eval=args.n_episodes_eval,
        seed=args.seed,
        linear_solver=linear_solver,
        cfg=cfg,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        train_updates=args.train_updates,
    )
    elapsed = time.time() - t0

    avg_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
    print(f"[done] Avg reward: {avg_reward:.6f} over {len(rewards)} steps | elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()
