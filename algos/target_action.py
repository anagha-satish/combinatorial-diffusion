from __future__ import annotations
import torch
from torch import Tensor
from typing import Tuple

def _softmax_with_temperature(x: Tensor, tau: float, dim: int) -> Tensor:
    z = x / tau
    z = z - z.max(dim=dim, keepdim=True).values
    exp = torch.exp(z)
    return exp / (exp.sum(dim=dim, keepdim=True) + 1e-12)

def soft_target_action(
    actions: Tensor,
    q_values: Tensor,
    tau: float,
    cand_dim: int = 0,
) -> Tuple[Tensor, Tensor]:
    if actions.shape[cand_dim] <= 0:
        raise ValueError("no candidates in actions")

    m = actions.shape[cand_dim]
    if cand_dim != 0:
        perm = (cand_dim,) + tuple(i for i in range(actions.ndim) if i != cand_dim)
        actions = actions.permute(perm)
    else:
        perm = None

    if q_values.ndim == 1:
        q = q_values.view(m, 1)
    elif q_values.ndim == 2:
        q = q_values

    weights = _softmax_with_temperature(q, tau=tau, dim=0)
    
    if actions.ndim == 2:
        a2 = actions.unsqueeze(1) # (m, 1, d)
        B = 1
        d_flat = actions.shape[-1]
    else:
        B = actions.shape[1]
        d_flat = int(torch.tensor(actions.shape[2:]).prod().item())
        a2 = actions.reshape(m, B, d_flat)

    w3 = weights.unsqueeze(-1) # (m, B, 1)
    a_hat_flat = (w3 * a2).sum(dim=0) # (B, d_flat)

    if actions.ndim == 2:
        a_hat = a_hat_flat.squeeze(0) # (d,)
    else:
        a_hat = a_hat_flat.reshape((B,) + actions.shape[2:]) # (B, ...)

    if q_values.ndim == 1:
        weights_out = weights.squeeze(1) # (m,)
    else:
        weights_out = weights # (m, B)

    return a_hat, weights_out







