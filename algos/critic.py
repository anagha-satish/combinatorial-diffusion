from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import Tuple

class Critic(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, hidden: int = 256):
        super().__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, s: Tensor, a: Tensor) -> Tensor:
        s = s.reshape(s.shape[0], -1)
        a = a.reshape(a.shape[0], -1)
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)  # (B,)

class DoubleCritic(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = Critic(s_dim, a_dim, hidden)
        self.q2 = Critic(s_dim, a_dim, hidden)
    def forward(self, s: Tensor, a: Tensor) -> Tuple[Tensor, Tensor]:
        return self.q1(s, a), self.q2(s, a)

@torch.no_grad()
def soft_target_update(target: nn.Module, online: nn.Module, tau: float = 0.995):
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(tau).add_(p_o.data, alpha=1 - tau)
        
@torch.no_grad()
def hard_target_update(target: nn.Module, online: nn.Module):
    target.load_state_dict(online.state_dict())

def td_loss(
    critic: Critic,
    target_critic: Critic,
    gamma: float,
    s: Tensor, a: Tensor, r: Tensor, s_next: Tensor, done: Tensor,
    a_next: Tensor,
) -> Tensor:
    q = critic(s, a)
    with torch.no_grad():
        q_next = target_critic(s_next, a_next)
        y = r + gamma * q_next * (1.0 - done)
    return torch.mean((q - y) ** 2)

def td_loss_double(
    critic: DoubleCritic,
    target_critic: DoubleCritic,
    gamma: float,
    s: Tensor, a: Tensor, r: Tensor, s_next: Tensor, done: Tensor,
    a_next: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    q1, q2 = critic(s, a)  # (B,), (B,)
    with torch.no_grad():
        q1_next, q2_next = target_critic(s_next, a_next)
        q_next = torch.minimum(q1_next, q2_next)
        y = r + gamma * q_next * (1.0 - done)  # (B,)
    l1 = torch.mean((q1 - y) ** 2)
    l2 = torch.mean((q2 - y) ** 2)
    return (l1 + l2), l1, l2

