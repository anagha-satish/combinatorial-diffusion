# Combinatorial Diffusion / LSFlow

This code implements **LSFlow**: a latent spherical flow policy for reinforcement learning with combinatorial action spaces. This algorithm is described in the paper [**Latent Spherical Flow Policy for Reinforcement Learning with Combinatorial Actionss**](https://arxiv.org/pdf/2601.22211) by Lingkai Kong, Anagha Satish, Hezi Jiang, Akseli Kangaslahti, Andrew Ma, Wenbo Chen, Mingxiao Song, Lily Xu, and Milind Tambe, which combines expressive generative policies with feasibility guarantees from combinatorial optimization solvers. This work will appear at the International Conference on Machine Learning (ICML 2026) as a Spotlight. 

## Overview

Reinforcement learning with combinatorial actions is difficult because the feasible action set is exponentially large and constrained. Instead of directly parameterizing a policy over feasible structured actions, LSFLOW learns a distribution over latent cost vectors and uses a combinatorial optimizer to map each sample to a feasible action.

LSFlow has three main components:

1. **Solver-induced policy**
A latent vector `c` defines the linear object of a combinatorial optimization problem. The solver returns the action, so feasibility is guaranteed.

2. **Spherical flow policy**
The policy is modeled as a distribution on the sphere using **spherical flow matching**, enabling expressive stochastic sampling directly in the latent cost space.

3. **Smoothed latent-space critic**
To avoid repeated solver calls, the critic is trained directly in latent cost space. LSFlow introduces a **smoothed Bellman operator** using a von Mises-Fisher (vMF) kernel for stable learning.

## Running Benchmark Experiments

We use the benchmarks created by [**Xu et al.**](https://arxiv.org/pdf/2503.01919). Their code is available at (https://github.com/lily-x/combinatorial-rmab). These settings include:

1. Dynamic Scheduling
2. Dynamic Routing
3. Dynamic Assignment
4. Dynamic Interventions

The main benchmark file is `driver_lsflow.py`. The four domains can be run as follows:

1. **Dynamic Scheduling**
```
python driver_lsflow.py --rmab_type scheduling --n_arms 40 --budget 10 --horizon 20
```

2. **Dynamic Routing**
```
python driver_lsflow.py --rmab_type routing --n_arms 40 --budget 10 --horizon 20
```

3. **Dynamic Assignment**
```
python driver_lsflow.py --rmab_type constrained --n_arms 40 --budget 10 --horizon 20
```

4. **Dynamic Interventions**
```
python driver_lsflow.py --rmab_type routing --n_arms 40 --budget 10 --horizon 20 --seed 0
```

Further hyperparameter details can be found in the appendix.

## Running Disease Network Experiments

We evaluate LSFlow on a real-world sexually transmitted infection (STI) testing class. The data can be accessed at (https://www.icpsr.umich.edu/web/ICPSR/studies/22140). 

Running `driver_disease_lsflow.py` will run this disease experiment. 

The following options are available:

*  `-D`, `--std_name`: disease name {`HIV`, `Gonorrheag`, `Syphillis`, `Chlamydia`}, default `HIV`
*  `-T`, `--cc_threshold`: minimum number of nodes from connected components, default `300`
*  `-I`, `--inst_idx`: instance index controlling connected-component sampling, default `0`
*  `-B`, `--budget`: batch size per step, default `5`
*  `-V`, `--n_episodes_eval`: number of episodes for evaluation, default `10`
*  `-s`, `--seed`: random seed, default `0`

Further hyperparameter details can be found in the appendix.

Example commands are as follows

```
python driver_disease_lsflow.py --std_name HIV --budget 5 --seed 0
python driver_disease_lsflow.py --std_name Gonorrhea --budget 5 --seed 0
python driver_disease_lsflow.py --std_name Chlamydia --budget 5 --seed 0
python driver_disease_lsflow.py --std_name Syphilis --budget 5 --seed 0
```

To load a cached graph

```
python driver_disease_lsflow.py --std_name HIV --load_graph_from path/to/cache.pkl
```

## Dependencies

To install dependencies, run:
```
pip install -r requirements.txt
```
