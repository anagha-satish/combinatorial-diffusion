from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import Callable, Optional

from .co_layer import act_with_noise, act_greedy, sample_candidates_from_theta

class ActorPhi(nn.Module):
    """
    Actor phi_w: state -> theta(score)
    """
    def __init__(self, s_dim: int, d: int, hidden: int = 256):
        super().__init__()
        self.s_dim = s_dim
        self.d = d
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d),
        )

    def forward(self, s: Tensor) -> Tensor:
        return self.net(s)
    
    @torch.no_grad()
    def act_for_env(self, s: Tensor, k: int, sigma_f: float, mask: Tensor | None = None, constraint_solver: Optional[Callable[[Tensor], Tensor]] = None) -> Tensor:
        """
        action with noise
        """
        theta = self.forward(s)
        return act_with_noise(theta, k=k, sigma_f=sigma_f, mask=mask, constraint_solver=constraint_solver)
    
    @torch.no_grad()
    def act_greedy(self, s: Tensor, k: int, mask: Tensor | None = None, constraint_solver: Optional[Callable[[Tensor], Tensor]] = None) -> Tensor:
        """
        a = f(theta, s)
        """
        theta = self.forward(s)
        return act_greedy(theta, k=k, mask=mask, constraint_solver=constraint_solver)

    @torch.no_grad()
    def sample_candidates(self, s: Tensor, k: int, m: int, sigma_b: float, mask: Tensor | None = None, constraint_solver: Optional[Callable[[Tensor], Tensor]] = None) -> Tensor:
        """
        get m 个 (a'_i)
        """
        theta = self.forward(s)
        return sample_candidates_from_theta(theta, k=k, m=m, sigma_b=sigma_b, mask=mask, constraint_solver=constraint_solver)



