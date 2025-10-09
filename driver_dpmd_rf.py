import os, argparse, datetime, random
import numpy as np

from environment.rmab_instances import (
    get_rmab_sigmoid, get_scheduling, get_constrained, get_routing
)
from environment.multi_action import MultiActionRMAB

from algos.repo_bridge import linear_solver_approx
from algos.dpmd_experiment_rf import run_dpmd_only


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--horizon', '-H', type=int, default=20)
    parser.add_argument('--rmab_type', '-D', type=str, default='routing',
                        help='{sigmoid, multistate, scheduling, constrained, standard, routing}')
    parser.add_argument('--n_actions', '-N', type=int, default=3)
    parser.add_argument('--n_arms', '-J', type=int, default=5)
    parser.add_argument('--budget', '-B', type=int, default=2)
    parser.add_argument('--n_episodes_eval', '-V', type=int, default=10)
    parser.add_argument('--prefix', '-p', type=str, default='')
    parser.add_argument('--fast_prototyping', '-F', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.rmab_type == 'sigmoid':
        rmab, t_rmab = get_rmab_sigmoid(args.n_arms, args.n_actions, args.budget, args.horizon)
    elif args.rmab_type == 'constrained':
        rmab, t_rmab = get_constrained(args.horizon, args.n_arms, args.budget, rmab_type='multistate')
    elif args.rmab_type == 'scheduling':
        rmab, t_rmab = get_scheduling(args.horizon, args.n_arms, args.budget, rmab_type='multistate')
    elif args.rmab_type == 'routing':
        rmab, t_rmab = get_routing(args.horizon, args.n_arms, args.budget, rmab_type='multistate')
    else:
        raise NotImplementedError(f"{args.rmab_type} not supported in this dpmd-rf driver yet.")

    if isinstance(rmab, MultiActionRMAB):
        print(f"[info] MultiAction wrapper detected: {rmab.link_type}")

    linear_solver = linear_solver_approx(rmab)

    print(f'Running DPMD-RF on {rmab} with J={rmab.n_arms}, H={rmab.horizon}, B={args.budget}')
    os.makedirs('./plots', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    rewards = run_dpmd_only(rmab, t_rmab, args.horizon, args.budget,
                            args.n_episodes_eval, args.seed, linear_solver)

    print('--------------------------------------------------------')
    print('DPMD-RF results')
    print('--------------------------------------------------------')
    print(f'Avg reward: {rewards.mean():.3f} over {len(rewards)} steps')


if __name__ == '__main__':
    main()
