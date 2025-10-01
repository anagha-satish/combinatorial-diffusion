import argparse, numpy as np
from environment.rmab_instances import get_routing
from algos.dpmd_runner import run_dpmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--n_arms', type=int, default=20)
    parser.add_argument('--budget', type=int, default=3)
    args = parser.parse_args()

    rmab, t_rmab = get_routing(args.horizon, args.n_arms, args.budget, rmab_type='multistate')

    total = run_dpmd(rmab, args.horizon, args.budget, seed=args.seed)
    print(f"DPMD total reward = {total:.2f}")
