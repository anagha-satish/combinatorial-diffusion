# driver_disease_dpmd.py

import argparse
import datetime
import io
import os
import sys
import time
import contextlib

import numpy as np
import torch

from environment.disease_graph_loader import (
    load_disease_graph_instance,
    create_disease_env,
)
from approximator.batch_graph_approximator import BatchGraphApproximator

from algos.dpmd_experiment_rf import run_dpmd_only
from algos.dpmd_rf import DPMDConfig


class DiseaseDPMDEnv:
    def __init__(self, base_env):
        self.base = base_env

        # Attributes expected by run_dpmd_only
        self.n_arms = base_env.num_nodes          # action dimension
        self.n_actions = 1                        # dummy
        self.budget = base_env.budget
        self.discount_factor = base_env.discount_factor
        self.num_nodes = base_env.num_nodes

    def reset(self):
        status, mask = self.base.reset()
        self._last_mask = mask
        return status.copy()

    def step(self, action_vec):
        status, mask, reward, done = self.base.step(action_vec)
        self._last_mask = mask
        info = {}
        return status.copy(), float(reward), bool(done), info

    # For the approximator
    def observation(self):
        return self.base.observation()

    def allowed_mask(self):
        return self.base.allowed_mask()

    def random_feasible_action(self):
        return self.base.random_feasible_action()

    def project_to_feasible(self, a):
        return self.base.project_to_feasible(a)


# ------------------------------------------------------------------
# Reseed util
# ------------------------------------------------------------------

def reseed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np_seed = seed % (2**32 - 1)
    np.random.seed(np_seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DPMD-RF on disease graphs")

    parser.add_argument('-D', '--std_name', type=str, default='HIV',
                        help='{HIV, Gonorrhea, Chlamydia, Syphilis, Hepatitis}')
    parser.add_argument('-T', '--cc_threshold', type=int, default=300,
                        help='minimum nodes from connected components')
    parser.add_argument('-I', '--inst_idx', type=int, default=0,
                        help='instance index controlling CC sampling')
    parser.add_argument('-B', '--budget', type=int, default=5,
                        help='batch size per step')
    parser.add_argument('-V', '--n_episodes_eval', type=int, default=10,
                        help='episodes to evaluate after training')
    parser.add_argument('-s', '--seed', type=int, default=0)

    # Training budgets
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--train_updates', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=64)

    # Core RL hyperparams
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--delay_update', type=int, default=2)
    parser.add_argument('--reward_scale', type=float, default=1.0)

    # DPMD / RFM hyperparams (same as routing driver)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--num_particles', type=int, default=12)

    parser.add_argument('--lambda_start', type=float, default=2.0)
    parser.add_argument('--lambda_end', type=float, default=0.8)
    parser.add_argument('--lambda_steps', type=int, default=10000)
    parser.add_argument('--w_clip', type=float, default=4.0)

    parser.add_argument('--kappa_exec', type=float, default=28.0)
    parser.add_argument('--kappa_smooth', type=float, default=28.0)
    parser.add_argument('--M_smooth', type=int, default=16)
    parser.add_argument('--J_smooth', type=int, default=1)

    parser.add_argument('--flow_steps', type=int, default=36)

    parser.add_argument('--q_norm_clip', type=float, default=3.0)
    parser.add_argument('--q_running_beta', type=float, default=0.05)

    args = parser.parse_args()

    reseed_all(args.seed)

    print('--------------------------------------------------------')
    print('Load Disease Graph')
    print('--------------------------------------------------------')

    G, covariates, theta_unary, theta_pairwise, statuses = load_disease_graph_instance(
        std_name=args.std_name,
        cc_threshold=args.cc_threshold,
        inst_idx=args.inst_idx,
    )

    print('graph stats')
    print(f'  disease: {args.std_name}')
    print(f'  nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}')
    print(f'  infected: {sum(statuses.values())}/{len(statuses)} '
          f'({100*sum(statuses.values())/len(statuses):.1f}%)')
    print(f'  covariate dim: {len(covariates[0])}')

    print('--------------------------------------------------------')
    print('Create Disease Environment')
    print('--------------------------------------------------------')

    base_env = create_disease_env(
        G, covariates, theta_unary, theta_pairwise,
        budget=args.budget,
        discount_factor=args.gamma,  # same gamma
        rng_seed=args.seed,
    )

    # Wrap for DPMD
    env = DiseaseDPMDEnv(base_env)
    horizon = base_env.n
    print(f'environment: n={base_env.n}, budget={base_env.budget}')

    # Build linear solver for DPMD using BatchGraphApproximator
    approximator = BatchGraphApproximator(env)

    def linear_solver(c: np.ndarray) -> np.ndarray:
        return approximator.solve_from_coeffs(c)

    # DPMD config
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

    print('--------------------------------------------------------')
    print('Train + Evaluate DPMD-RF on disease env')
    print('--------------------------------------------------------')

    t0 = time.time()
    rewards = run_dpmd_only(
        env,
        horizon=horizon,
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

    # Simple scalar metric
    gamma = args.gamma
    disc_returns = []
    steps_per_ep = horizon
    rewards = np.asarray(rewards, dtype=float)
    if rewards.size > 0:
        rewards = rewards.reshape(-1, steps_per_ep)
        discounts = gamma ** np.arange(steps_per_ep)
        disc_returns = (rewards * discounts[None, :]).sum(axis=1)

    if len(disc_returns) > 0:
        mean_disc = float(np.mean(disc_returns))
        std_disc = float(np.std(disc_returns))
    else:
        mean_disc = 0.0
        std_disc = 0.0

    print('--------------------------------------------------------')
    print('Results')
    print('--------------------------------------------------------')
    print(f'{args.std_name} disease testing | n={base_env.n}, budget={args.budget}, gamma={gamma}')
    print(f'  episodes: {len(disc_returns)}, mean discounted return = {mean_disc:.4f}, std = {std_disc:.4f}')
    print(f'  runtime: {elapsed:.2f} seconds')
    print('[done]')


if __name__ == "__main__":
    main()
