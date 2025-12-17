# driver_dpmd_rf.py
"""
Driver for DPMD-RF using tuned defaults.
"""
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
    p.add_argument('--warmup_steps', type=int, default=1000,
                   help='steps collected before any gradient updates')
    p.add_argument('--train_updates', type=int, default=2000,
                   help='number of training episodes/updates')
    p.add_argument('--batch_size', type=int, default=64)

    # Core RL hyperparameters (stable across RMAB variants)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--delay_update', type=int, default=2)
    p.add_argument('--reward_scale', type=float, default=1.0,
                   help='scale rewards before critic targets')

    # DPMD / RFM hyperparameters (paper-consistent, stable defaults)
    p.add_argument('--lr', type=float, default=4e-4,
                   help='learning rate for RFM actor (and critics)')
    p.add_argument('--num_particles', type=int, default=12,
                   help='number of candidate actions K sampled per state')

    # Mirror-descent weighting temperature schedule (Eq. 7)
    p.add_argument('--lambda_start', type=float, default=2.0)
    p.add_argument('--lambda_end', type=float, default=0.8)
    p.add_argument('--lambda_steps', type=int, default=10000)
    p.add_argument('--w_clip', type=float, default=4.0,
                   help='max clamp on per-sample mirror-descent weights')

    # von Mises–Fisher execution noise (Alg. 1, Step 7)
    p.add_argument('--kappa_exec', type=float, default=28.0,
                   help='vMF concentration during action execution')

    # Smoothed Bellman operator (Eq. 10)
    p.add_argument('--kappa_smooth', type=float, default=28.0,
                   help='vMF concentration for target smoothing kernel')
    p.add_argument('--M_smooth', type=int, default=16,
                   help='number of target-policy candidates M')
    p.add_argument('--J_smooth', type=int, default=1,
                   help='vMF perturbations per candidate J')

    # RFM integration
    p.add_argument('--flow_steps', type=int, default=36,
                   help='integration steps for RFM sampling on S^{D-1}')

    # Stability knobs (RFM-for-RL implementation details)
    p.add_argument('--q_norm_clip', type=float, default=3.0,
                   help='clip normalized Q before exponentiation')
    p.add_argument('--q_running_beta', type=float, default=0.05,
                   help='EMA coefficient for running Q mean/std')

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
        w_clip=args.w_clip,
        kappa_exec=args.kappa_exec,
        kappa_smooth=args.kappa_smooth,
        M_smooth=args.M_smooth,
        J_smooth=args.J_smooth,
        flow_steps=args.flow_steps,
        lambda_start=args.lambda_start,
        lambda_end=args.lambda_end,
        lambda_steps=args.lambda_steps,
        q_norm_clip=args.q_norm_clip,
        q_running_beta=args.q_running_beta,
    )


    t0 = time.time()
    rewards, learner = run_dpmd_only(
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
