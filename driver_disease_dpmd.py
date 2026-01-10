# driver_disease_dpmd.py

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from environment.disease_graph_loader import (
    load_disease_graph_instance,
    create_disease_env,
)
from approximator.batch_graph_approximator import BatchGraphApproximator

from algos.dpmd_experiment_rf_disease_gnn import run_dpmd_rf_disease_gnn
from algos.dpmd_rf_disease_gnn import DPMDGraphConfig

# ------------------------------------------------------------------
# Simple wrapper around BinaryFrontierEnvBatch
# ------------------------------------------------------------------

class DiseaseDPMDEnv:
    """
    Wrap BinaryFrontierEnvBatch so it looks like the env expected
    by run_dpmd_only (obs, reward, done, info) and by the
    BatchGraphApproximator.
    """

    def __init__(self, base_env):
        self.base = base_env

        # Attributes expected by run_dpmd_only / approximator
        self.n_arms = base_env.num_nodes        # action dimension
        self.n_actions = 1                      # dummy
        self.budget = base_env.budget
        self.discount_factor = base_env.discount_factor
        self.num_nodes = base_env.num_nodes

    # ---- gym-like API ----
    def reset(self):
        status, mask = self.base.reset()
        self._last_mask = mask
        return status.copy()

    def step(self, action_vec):
        status, mask, reward, done = self.base.step(action_vec)
        self._last_mask = mask
        info = {}
        return status.copy(), float(reward), bool(done), info

    # ---- properties so evaluation code can read internals ----
    @property
    def tests_done(self):
        return self.base.tests_done

    @property
    def world_X(self):
        return self.base.world_X

    # ---- helpers for BatchGraphApproximator ----
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
    np_seed = seed % (2 ** 32 - 1)
    np.random.seed(np_seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Detection curve evaluation
# ------------------------------------------------------------------

def evaluate_detection_curve(
    learner,
    env,
    linear_solver,
    n_episodes_eval: int = 10,
):
    """
    Evaluate a trained DPMD policy on the disease environment.

    Returns:
        x: [T] mean fraction of population tested
        y: [T] mean fraction of positive cases detected (normalized)
        y_std: [T] std of fraction detected across episodes
    """
    # Number of nodes in the graph
    n = getattr(env, "num_nodes", getattr(env, "n", None))
    if n is None:
        raise AttributeError("Env must have attribute `num_nodes` or `n`.")

    # Per-step testing budget
    B = int(getattr(env, "budget", 1))
    # Enough steps to (in principle) test everyone once
    max_steps = int(np.ceil(n / B)) + 1

    all_tested = []    # [E, T]
    all_detected = []  # [E, T]

    for ep in range(n_episodes_eval):
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs = reset_out[0]
        else:
            obs = reset_out
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        cum_rewards = []
        tested_frac = []
        cum = 0.0

        for step in range(max_steps):
            # Greedy action from trained policy (same pattern as in run_dpmd_only)
            Cs = learner.sample_candidates(obs, K=learner.cfg.num_particles)
            qs = learner.score_actions(obs, Cs)
            j = int(np.argmax(qs))
            c_star = Cs[j]
            action = linear_solver(c_star)

            step_out = env.step(action)

            # Try to interpret env.step() output robustly
            if isinstance(step_out, tuple) and len(step_out) == 4:
                a, b, c, d = step_out
                # Heuristic: if 'b' looks like a mask with same shape as 'a',
                # treat as (obs, mask, reward, done); else (obs, reward, done, info)
                if isinstance(b, np.ndarray) and b.shape == np.asarray(a).shape:
                    next_obs = a
                    reward = c
                    done = d
                else:
                    next_obs = a
                    reward = b
                    done = c
            elif isinstance(step_out, tuple) and len(step_out) == 5:
                # Gymnasium-style: (obs, reward, terminated, truncated, info)
                next_obs, reward, terminated, truncated, _info = step_out
                done = bool(terminated or truncated)
            else:
                # Fallback
                try:
                    next_obs, reward, done = step_out[0], step_out[1], step_out[2]
                except Exception:
                    break

            cum += float(reward)
            obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)

            # Use env.tests_done if exposed; otherwise infer from obs
            if hasattr(env, "tests_done"):
                tests_done = float(env.tests_done)
            else:
                status = obs
                tests_done = float(np.count_nonzero(status > -0.5))

            tested_frac.append(tests_done / float(n))
            cum_rewards.append(cum)

            if done:
                break

        if len(cum_rewards) == 0:
            cum_rewards = [0.0]
            tested_frac = [0.0]

        # Total positives in this world = final cumulative reward
        total_pos = max(cum_rewards[-1], 1.0)
        detected_frac = [cr / total_pos for cr in cum_rewards]

        # Pad to max_steps so we can average across episodes
        while len(tested_frac) < max_steps:
            tested_frac.append(tested_frac[-1])
            detected_frac.append(detected_frac[-1])

        all_tested.append(tested_frac)
        all_detected.append(detected_frac)

    all_tested = np.array(all_tested)     # [E, T]
    all_detected = np.array(all_detected) # [E, T]

    x = all_tested.mean(axis=0)
    y = all_detected.mean(axis=0)
    y_std = all_detected.std(axis=0)

    return x, y, y_std


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

    # Logging
    parser.add_argument(
        '--log_every',
        type=int,
        default=10,
        help='Log training metrics every N episodes',
    )

    # Core RL hyperparams
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--delay_update', type=int, default=2)
    parser.add_argument('--reward_scale', type=float, default=1.0)

    # DPMD / RFM hyperparams
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

    parser.add_argument('--load_graph_from', type=str, default=None,
                        help='Path to cached graph pickle file. If provided, skips graph generation.')

    args = parser.parse_args()

    reseed_all(args.seed)

    print('--------------------------------------------------------')
    print('Load Disease Graph')
    print('--------------------------------------------------------')

    if args.load_graph_from is not None:
        # Load from cache
        print(f"Loading graph from cache: {args.load_graph_from}")
        print(f"  Note: Ignoring cc_threshold={args.cc_threshold}, inst_idx={args.inst_idx}")

        from environment.disease_graph_loader import load_graph_cache
        G, covariates, theta_unary, theta_pairwise, statuses = load_graph_cache(
            cache_path=args.load_graph_from,
            expected_std_name=args.std_name
        )
    else:
        # Standard graph generation
        from environment.disease_graph_loader import save_graph_cache
        G, covariates, theta_unary, theta_pairwise, statuses = load_disease_graph_instance(
            std_name=args.std_name,
            cc_threshold=args.cc_threshold,
            inst_idx=args.inst_idx,
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
            cc_threshold=args.cc_threshold
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
        discount_factor=args.gamma,
        rng_seed=args.seed,
    )

    env = DiseaseDPMDEnv(base_env)
    horizon = base_env.n
    print(f'environment: n={base_env.n}, budget={base_env.budget}')

    # Build linear solver for DPMD using BatchGraphApproximator
    approximator = BatchGraphApproximator(env)

    def linear_solver(c: np.ndarray) -> np.ndarray:
        return approximator.solve_from_coeffs(c)

    # DPMD config
    cfg = DPMDGraphConfig(
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
    # IMPORTANT: run_dpmd_only must return (rewards_array, learner)
    rewards, learner = run_dpmd_rf_disease_gnn(
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
        log_every_n_episodes=args.log_every,
    )
    elapsed = time.time() - t0

    # Detection curve for the trained policy
    x, y, y_std = evaluate_detection_curve(
        learner,
        env,
        linear_solver,
        n_episodes_eval=10,
    )

    # ------------------------------------------------------------------
    # Ensure results directory exists
    # ------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------
    # Unique run tag for sweeps (already correct)
    # ------------------------------------------------------------------
    run_tag = (
        f"{args.std_name}"
        f"_T{args.cc_threshold}"
        f"_B{args.budget}"
        f"_gamma{args.gamma}"
        f"_seed{args.seed}"
        f"_train{args.train_updates}"
        f"_flow{args.flow_steps}"
        f"_K{args.num_particles}"
    )

    # ------------------------------------------------------------------
    # Save evaluation vectors (unique per sweep run)
    # ------------------------------------------------------------------
    npz_path = f"results/eval_results_{run_tag}.npz"
    np.savez(
        npz_path,
        x=x,
        y=y,
        y_std=y_std
    )
    print("Saved eval vectors to:", npz_path)

    # ------------------------------------------------------------------
    # Plot detection curve (unique per sweep run)
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, linestyle="--", color="tab:blue", label="DPMD-RF")
    plt.fill_between(x, y - y_std, y + y_std, color="tab:blue", alpha=0.25)
    plt.axvline(x=0.5, linestyle="--", color="gray", alpha=0.7)
    plt.xlabel("Fraction of population tested")
    plt.ylabel("Fraction of positive cases detected (normalized)")
    plt.title("Policies with discount = 0.99")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()

    png_path = f"results/disease_detection_curve_{run_tag}.png"
    plt.savefig(png_path, dpi=200)
    print("Saved plot to:", png_path)
    plt.close()



    # Discounted return summary (using training-time rewards)
    gamma = args.gamma
    rewards = np.asarray(rewards, dtype=float)
    steps_per_ep = horizon
    if rewards.size > 0:
        rewards = rewards.reshape(-1, steps_per_ep)
        discounts = gamma ** np.arange(steps_per_ep)
        disc_returns = (rewards * discounts[None, :]).sum(axis=1)
    else:
        disc_returns = np.array([])

    if disc_returns.size > 0:
        mean_disc = float(np.mean(disc_returns))
        std_disc = float(np.std(disc_returns))
    else:
        mean_disc = 0.0
        std_disc = 0.0

    print('--------------------------------------------------------')
    print('Results')
    print('--------------------------------------------------------')
    print(f'{args.std_name} disease testing | n={base_env.n}, budget={args.budget}, gamma={gamma}')
    print(f'  episodes: {disc_returns.size}, mean discounted return = {mean_disc:.4f}, std = {std_disc:.4f}')
    print(f'  runtime: {elapsed:.2f} seconds')
    print('[done]')


if __name__ == "__main__":
    main()
