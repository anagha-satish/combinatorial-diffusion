# Combinatorial Mirror-Descent Diffusion

We build off the code for Combinatorial RMBB from the work of [Xu et al.](https://github.com/lily-x/combinatorial-rmab/tree/main) and Online RL for Diffusion Policies from [Ma et al.](https://github.com/mahaitongdae/diffusion_policy_online_rl) 

For a very simple toy problem, run 

old version with jax: python driver_dpmd_jax.py -D routing -H 10 -J 20 -B 3 -s 0 -V 5

new version without jax and with Riemmanian flow: python driver_dpmd.py -D routing -H 10 -J 20 -B 3 -s 0 -V 5
