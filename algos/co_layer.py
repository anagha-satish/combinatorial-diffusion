from __future__ import annotations
import torch
from torch import Tensor
from typing import Optional, Tuple, Callable

INF = 1e9

def _ensure_2d(x: Tensor) -> Tuple[Tensor, bool]:
    """
    reshape to (B, d)
    """
    if x.ndim == 1:
        return x.unsqueeze(0), True
    elif x.ndim == 2:
        return x, False
    else:
        B = int(torch.tensor(x.shape[:-1]).prod().item())
        return x.reshape(B, x.shape[-1]), False

def _apply_mask(scores: Tensor, mask: Optional[Tensor]) -> Tensor:
    """
    mask: 1 for can choose, 0 for cannot choose(-INF)
    """
    if mask is None:
        return scores
    
    m = mask.to(dtype=scores.dtype)
    return scores * m + (1.0 - m) * (-INF)

## TODO: add constrains begin
@torch.no_grad()
def solve_from_theta(
    theta: Tensor,
    k: int,
    mask: Optional[Tensor] = None,
    constraint_solver: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """
    a = argmax_{a∈{0,1}^d, sum(a)=k} <theta, a>
    """
    theta2d, added = _ensure_2d(theta)  # (B, d)

    if constraint_solver is not None:
        B, d = theta2d.shape
        actions = []
        mask2d = None if mask is None else mask.reshape(theta2d.shape)

        for b in range(B):
            theta_b = theta2d[b]

            if mask2d is not None:
                m_b = mask2d[b].to(dtype=theta_b.dtype)
                theta_b_eff = theta_b * m_b + (1.0 - m_b) * (-INF)
            else:
                theta_b_eff = theta_b

            a_b = constraint_solver(theta_b_eff)  # shape: (d,)

            a_b = a_b.to(theta_b.device, dtype=theta_b.dtype)

            actions.append(a_b)

        action = torch.stack(actions, dim=0)  # (B, d)

        if added and theta.ndim == 1:
            return action.squeeze(0)
        return action

    scores = _apply_mask(theta2d, mask if mask is None else mask.reshape(theta2d.shape))

    B, d = scores.shape
    vals, idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=False)
    action = torch.zeros_like(scores)
    action.scatter_(dim=1, index=idx, src=torch.ones_like(vals))

    if mask is not None:
        action = action * mask.reshape(scores.shape).to(dtype=action.dtype)

    if added and theta.ndim == 1:
        return action.squeeze(0)

    return action
## TODO: add constrains end

@torch.no_grad()
def solve_from_eta(
    eta: Tensor,
    k: int,
    mask: Optional[Tensor] = None,
    constraint_solver: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """
    add perturbation of theta
    """
    return solve_from_theta(eta, k=k, mask=mask, constraint_solver=constraint_solver)

@torch.no_grad()
def sample_candidates_from_theta(
    theta: Tensor,
    k: int,
    m: int,
    sigma_b: float,
    mask: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
    constraint_solver: Optional[Callable[[Tensor], Tensor]] = None
) -> Tensor:
    """
    eta_i = θ + Normal(0, sigma_b), a'_i = f(eta_i, s)
    """
    theta2d, added = _ensure_2d(theta)
    B, d = theta2d.shape
    noise = torch.randn((m, B, d), device=theta2d.device, generator=generator) * sigma_b
    etas = theta2d.unsqueeze(0) + noise

    actions = []
    mask2d = None if mask is None else mask.reshape(theta2d.shape)

    for i in range(m):
        a_i = solve_from_eta(etas[i], k=k, mask=mask2d, constraint_solver=constraint_solver)
        actions.append(a_i)
    A = torch.stack(actions, dim=0)  # (m, B, d)

    if added and theta.ndim == 1:
        return A.squeeze(1)  # (m, d)

    return A  # (m, B, d)


@torch.no_grad()
def act_greedy(
    theta: Tensor,
    k: int,
    mask: Optional[Tensor] = None,
    constraint_solver: Optional[Callable[[Tensor], Tensor]] = None
) -> Tensor:
    return solve_from_theta(theta, k=k, mask=mask, constraint_solver=constraint_solver)


@torch.no_grad()
def act_with_noise(
    theta: Tensor,
    k: int,
    sigma_f: float,
    mask: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
    constraint_solver: Optional[Callable[[Tensor], Tensor]] = None
) -> Tensor:
    theta2d, added = _ensure_2d(theta)
    noise = torch.randn(
        theta2d.shape, device=theta2d.device, dtype=theta2d.dtype, generator=generator
    ) * sigma_f

    eta = theta2d + noise
    a = solve_from_eta(
        eta,
        k=k,
        mask=mask if mask is None else mask.reshape(theta2d.shape),
        constraint_solver=constraint_solver
    )

    if added and theta.ndim == 1:
        return a.squeeze(0)
    return a

## TODO: add constrains begin
def make_constraint_solver_from_approximator(approximator, device=None) -> Callable[[Tensor], Tensor]:

    def constraint_solver(theta_1d: Tensor) -> Tensor:
        # theta_1d: shape (d,)
        c = theta_1d.detach().cpu().numpy().astype(float)
        x_np = approximator.solve_from_coeffs(c)  # 0/1 np array, shape (d,)
        x_t = torch.from_numpy(x_np).to(device=device, dtype=theta_1d.dtype)
        return x_t

    return constraint_solver
## TODO: add constrains end