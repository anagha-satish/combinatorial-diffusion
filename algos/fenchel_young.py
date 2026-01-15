# algos/fenchel_young.py
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor


def fy_loss(
    theta: Tensor,
    a_hat: Tensor,
    k: float,
    *,
    epsilon: float = 1.0,
    num_samples: int = 20,
    mask: Optional[Tensor] = None,
    constraint_solver: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:

    # Normalize to 2D [B,d]
    was_1d = (theta.ndim == 1)
    theta2 = theta.unsqueeze(0) if was_1d else theta
    a_hat2 = a_hat.unsqueeze(0) if a_hat.ndim == 1 else a_hat

    if theta2.ndim != 2:
        theta2 = theta2.reshape(-1, theta2.shape[-1])
    if a_hat2.ndim != 2:
        a_hat2 = a_hat2.reshape(-1, a_hat2.shape[-1])

    B, d = theta2.shape
    if a_hat2.shape != (B, d):
        # allow a_hat provided as [d] to broadcast to [B,d]
        if a_hat2.shape == (1, d) and B > 1:
            a_hat2 = a_hat2.expand(B, d)
        else:
            raise ValueError(f"a_hat shape {tuple(a_hat2.shape)} != {(B, d)}")

    # Broadcast mask to [B,d] if provided
    mask2 = None
    if mask is not None:
        if mask.ndim == 1:
            if mask.numel() != d:
                raise ValueError(f"mask length {mask.numel()} != d={d}")
            mask2 = mask.view(1, -1).expand(B, d)
        elif mask.ndim == 2:
            if mask.shape != (B, d):
                # allow (1,d) broadcast
                if mask.shape == (1, d) and B > 1:
                    mask2 = mask.expand(B, d)
                else:
                    raise ValueError(f"mask shape {tuple(mask.shape)} != {(B, d)}")
            else:
                mask2 = mask
        else:
            raise ValueError(f"mask must be 1D or 2D, got ndim={mask.ndim}")

    # Sample perturbations: Z ~ N(0, I)
    Z = torch.randn((num_samples, B, d), device=theta2.device, dtype=theta2.dtype)
    theta_eps = theta2.unsqueeze(0) + float(epsilon) * Z  # [S,B,d]

    # Compute perturbed maxima via the CO-layer solver
    from .co_layer import act_greedy

    max_vals = []
    for s in range(num_samples):
        a_eps = act_greedy(
            theta_eps[s],
            k=int(k),
            mask=mask2,
            constraint_solver=constraint_solver,
        )  # [B,d] feasible 0/1 action
        max_vals.append((theta_eps[s] * a_eps).sum(dim=-1))  # [B]

    max_term = torch.stack(max_vals, dim=0).mean(dim=0)  # [B]

    # Linear term: <theta, a_hat>
    lin_term = (theta2 * a_hat2.to(dtype=theta2.dtype)).sum(dim=-1)  # [B]

    loss = max_term - lin_term
    return loss.squeeze(0) if was_1d else loss
