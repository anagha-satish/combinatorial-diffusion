from __future__ import annotations
import numpy as np
import torch
from torch import Tensor, nn
from typing import Dict, Optional

from .co_layer import act_greedy
from .fenchel_young import fy_loss
from .critic import td_loss_double as td_loss_double_mlp
from .critic import soft_target_update as soft_target_update_mlp
from .critic import hard_target_update as hard_target_update_mlp

from .graph_critic import soft_target_update as soft_target_update_graph
from .graph_critic import hard_target_update as hard_target_update_graph


def train_step_double(
    actor_phi: nn.Module,
    critic,
    target_critic,
    opt_actor,
    opt_critic,
    batch: Dict[str, Tensor],
    k: int,
    m: int,
    tau: float,
    sigma_b: float,
    sigma_f: float,
    gamma: float = 0.99,
    polyak: float = 0.995,
    target_update: str = "soft",   # "soft" or "hard"
    do_hard_update: bool = False,
    delay_actor: int = 1,
    step_idx: int = 1,
    pa_noise: str = "gumbel",
    constraint_solver=None,
    fy_epsilon: float = 1.0,
    fy_num_samples: int = 20,
    # ---- new for disease+GNN ----
    graph_builder=None,           # DiseaseGraphBuilder
    use_graph: bool = False,      # disease path set True
):
    """
    If use_graph=True:
      - batch["s"], batch["s_next"] are status vectors [B,n]
      - actor_phi expects PyG Batch and returns theta [B,n]
      - critic/target_critic expect (PyG Batch, action [B,n]) and return Q [B]
    Else:
      - keep your original flat MLP path
    """
    s, a, r, s_next, done = batch["s"], batch["a"], batch["r"], batch["s_next"], batch["done"]
    mask = batch.get("mask", None)
    mask_next = batch.get("mask_next", mask)

    if use_graph:
        if graph_builder is None:
            raise ValueError("use_graph=True requires graph_builder (DiseaseGraphBuilder).")

        s_np = s.detach().cpu().numpy()
        snp = s_next.detach().cpu().numpy()

        batch_s = graph_builder.batch_from_status_batch(s_np)
        batch_sn = graph_builder.batch_from_status_batch(snp)

        with torch.no_grad():
            theta_next = actor_phi(batch_sn)  # [B,n]
            a_next = act_greedy(theta_next, k=k, mask=mask_next, constraint_solver=constraint_solver)

        q1, q2 = critic(batch_s, a)
        with torch.no_grad():
            tq1, tq2 = target_critic(batch_sn, a_next)
            q_next = torch.minimum(tq1, tq2)
            y = r + float(gamma) * q_next * (1.0 - done)

        loss_c1 = torch.mean((q1 - y) ** 2)
        loss_c2 = torch.mean((q2 - y) ** 2)
        loss_c_tot = loss_c1 + loss_c2

        opt_critic.zero_grad(set_to_none=True)
        loss_c_tot.backward()
        opt_critic.step()

        if target_update == "soft":
            soft_target_update_graph(target_critic.q1, critic.q1, tau=polyak)
            soft_target_update_graph(target_critic.q2, critic.q2, tau=polyak)
        elif target_update == "hard":
            if do_hard_update:
                hard_target_update_graph(target_critic.q1, critic.q1)
                hard_target_update_graph(target_critic.q2, critic.q2)

        loss_a = None
        weights_entropy_proxy = torch.tensor(0.0, device=s.device)
        weights_max_proxy = torch.tensor(0.0, device=s.device)

        if step_idx % delay_actor == 0:
            theta = actor_phi(batch_s)  # [B,n]

            from .co_layer import sample_candidates_from_theta
            A_cands = sample_candidates_from_theta(
                theta, k=k, m=m, sigma_b=sigma_b, mask=mask, constraint_solver=constraint_solver
            )  # (m, B, n)

            Qs = []
            for i in range(m):
                q1_i, q2_i = critic(batch_s, A_cands[i])
                q_i = (q1_i + q2_i) / 2
                Qs.append(q_i)
            Qs = torch.stack(Qs, dim=0)  # (m, B)

            from .target_action import soft_target_action
            a_hat, weights = soft_target_action(A_cands, Qs, tau=tau)

            loss_a = fy_loss(
                theta, a_hat, k=float(k),
                epsilon=fy_epsilon,
                num_samples=fy_num_samples,
                mask=mask,
                constraint_solver=constraint_solver,
            ).mean()


            opt_actor.zero_grad(set_to_none=True)
            loss_a.backward()
            opt_actor.step()

            with torch.no_grad():
                weights_entropy_proxy = (-(weights.clamp_min(1e-12).log() * weights).sum(dim=0).mean())
                weights_max_proxy = weights.max(dim=0).values.mean()

        stats = {
            "loss/critic_total": loss_c_tot.detach(),
            "loss/critic_q1": loss_c1.detach(),
            "loss/critic_q2": loss_c2.detach(),
            "pa/eps": torch.tensor(float(sigma_b), device=s.device),
            "pa/L": torch.tensor(int(m), device=s.device),
            "weights_entropy": weights_entropy_proxy,
            "weights_max": weights_max_proxy,
        }
        if loss_a is not None:
            stats["loss/actor"] = loss_a.detach()
        return stats

    with torch.no_grad():
        theta_next = actor_phi(s_next)
        a_next = act_greedy(theta_next, k=k, mask=mask_next, constraint_solver=constraint_solver)

    loss_c_tot, l1, l2 = td_loss_double_mlp(
        critic, target_critic, gamma,
        s, a, r, s_next, done, a_next
    )
    opt_critic.zero_grad(set_to_none=True)
    loss_c_tot.backward()
    opt_critic.step()

    if target_update == "soft":
        soft_target_update_mlp(target_critic.q1, critic.q1, tau=polyak)
        soft_target_update_mlp(target_critic.q2, critic.q2, tau=polyak)
    elif target_update == "hard":
        if do_hard_update:
            hard_target_update_mlp(target_critic.q1, critic.q1)
            hard_target_update_mlp(target_critic.q2, critic.q2)

    loss_a = None
    weights_entropy_proxy = torch.tensor(0.0, device=s.device)
    weights_max_proxy = torch.tensor(0.0, device=s.device)

    if step_idx % delay_actor == 0:
        theta = actor_phi(s)

        from .co_layer import sample_candidates_from_theta
        A_cands = sample_candidates_from_theta(
            theta, k=k, m=m, sigma_b=sigma_b, mask=mask, constraint_solver=constraint_solver
        )

        Qs = []
        for i in range(m):
            q1_i, q2_i = critic(s, A_cands[i])
            q_i = (q1_i + q2_i) / 2
            Qs.append(q_i)
        Qs = torch.stack(Qs, dim=0)

        from .target_action import soft_target_action
        a_hat, weights = soft_target_action(A_cands, Qs, tau=tau)

        loss_a = fy_loss(theta, a_hat, k=float(k)).mean()

        opt_actor.zero_grad(set_to_none=True)
        loss_a.backward()
        opt_actor.step()

        with torch.no_grad():
            weights_entropy_proxy = (-(weights.clamp_min(1e-12).log() * weights).sum(dim=0).mean())
            weights_max_proxy = weights.max(dim=0).values.mean()

    stats = {
        "loss/critic_total": loss_c_tot.detach(),
        "loss/critic_q1": l1.detach(),
        "loss/critic_q2": l2.detach(),
        "pa/eps": torch.tensor(float(sigma_b), device=s.device),
        "pa/L": torch.tensor(int(m), device=s.device),
        "weights_entropy": weights_entropy_proxy,
        "weights_max": weights_max_proxy,
    }
    if loss_a is not None:
        stats["loss/actor"] = loss_a.detach()
    return stats
