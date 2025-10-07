# algos/dpmd.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, NamedTuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor


from models.ddpm.service import ddpm_service as actor_service


class Experience(NamedTuple):
    obs:    np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_obs: np.ndarray
    done: np.ndarray


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )
    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


def _to_tensor(x, device, dtype=torch.float32) -> Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)

def _coerce_width(x: Tensor, target_F: int) -> Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() >= 3:
        x = x.view(x.shape[0], -1)
    F = x.shape[-1]
    if F == target_F:
        return x
    if F > target_F:
        return x[:, :target_F].contiguous()
    pad = torch.zeros(x.shape[0], target_F - F, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)

def _soft_update(src: nn.Module, tgt: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p_t, p_s in zip(tgt.parameters(), src.parameters()):
            p_t.data.mul_(1.0 - tau).add_(p_s.data, alpha=tau)


@dataclass
class DPMDConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    tau: float = 0.005
    delay_update: int = 2
    reward_scale: float = 0.2
    num_particles: int = 8
    target_J: int = 4
    lambda_temp: float = 1.0
    w_clip: Optional[float] = 50.0


class DPMD:
    """
    DPMD with a DDPM/RSM actor in coefficient space (matches paper Eqs. 3–5).
    - Actor lives in models/ddpm/service.py (singleton 'actor_service').
    - This class owns two Q-nets (+ targets) and trains them with TD.
    - Actor update delegates to actor_service.update(obs, c0, w).
    """
    def __init__(self, obs_dim: int, act_dim: int,
                 device: Optional[torch.device] = None,
                 cfg: DPMDConfig = DPMDConfig()):
        self.cfg = cfg
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # critics
        self.q1 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.q2 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.tq1 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.tq2 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())
        self.q_optim = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()),
                                  lr=self.cfg.lr)

        self.step = 0

    @torch.no_grad()
    def _target_v(self, next_obs: Tensor) -> Tensor:
        next_obs = _coerce_width(next_obs, self.obs_dim)
        B = next_obs.shape[0]

        cand_np = actor_service.sample(next_obs.detach().cpu().numpy(), K=self.cfg.target_J)
        cand = _to_tensor(cand_np, self.device)
        J = cand.shape[1]

        rep_obs = next_obs.unsqueeze(1).expand(B, J, self.obs_dim).reshape(B * J, self.obs_dim)
        flat_cand = cand.reshape(B * J, self.act_dim)
        q1 = self.tq1(rep_obs, flat_cand).view(B, J)
        q2 = self.tq2(rep_obs, flat_cand).view(B, J)
        return torch.minimum(q1, q2).mean(dim=1)

    def update(self, batch: Experience) -> Dict[str, float]:
        obs = _to_tensor(batch.obs, self.device).view(len(batch.obs), -1)
        act = _to_tensor(batch.action, self.device).view(len(batch.action), -1)
        rew = _to_tensor(batch.reward, self.device).view(-1)
        nxt = _to_tensor(batch.next_obs, self.device).view(len(batch.next_obs), -1)
        done = _to_tensor(batch.done, self.device).view(-1)

        obs = _coerce_width(obs, self.obs_dim)
        nxt = _coerce_width(nxt, self.obs_dim)

        rew = rew * self.cfg.reward_scale

        # ----- Critic target -----
        with torch.no_grad():
            vhat = self._target_v(nxt)
            y = rew + (1.0 - done) * self.cfg.gamma * vhat

        # ----- Critic step -----
        self.q_optim.zero_grad(set_to_none=True)
        q1_pred = self.q1(obs, act)
        q2_pred = self.q2(obs, act)
        q1_loss = torch.mean((q1_pred - y) ** 2)
        q2_loss = torch.mean((q2_pred - y) ** 2)
        (q1_loss + q2_loss).backward()
        self.q_optim.step()

        # ----- Actor step (DDPM/RSM + mirror-descent weights) -----
        with torch.no_grad():
            q_min = torch.minimum(self.q1(obs, act), self.q2(obs, act))
            q_center = (q_min - q_min.mean()) / (q_min.std() + 1e-6)
            w = torch.exp(q_center / self.cfg.lambda_temp)
            if self.cfg.w_clip is not None:
                w = torch.clamp(w, max=float(self.cfg.w_clip))

        policy_loss = actor_service.update(
            obs_np=obs.detach().cpu().numpy(),
            c0_np=act.detach().cpu().numpy(),
            w_np=w.detach().cpu().numpy()
        )

        # ----- Target soft-update -----
        if (self.step % self.cfg.delay_update) == 0:
            _soft_update(self.q1, self.tq1, self.cfg.tau)
            _soft_update(self.q2, self.tq2, self.cfg.tau)

        self.step += 1

        with torch.no_grad():
            qm = float(q_min.mean().detach().cpu())
            qs = float(q_min.std().detach().cpu())

        return {
            "q1_loss": float(q1_loss.detach().cpu()),
            "q2_loss": float(q2_loss.detach().cpu()),
            "policy_loss": float(policy_loss),
            "q_mean": qm,
            "q_std": qs,
        }

    @torch.no_grad()
    def score_actions(self, obs: np.ndarray, C: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1),
                            device=self.device)
        c = torch.as_tensor(C, device=self.device, dtype=torch.float32)
        o = _coerce_width(o, self.obs_dim).repeat(c.shape[0], 1)
        q = torch.minimum(self.q1(o, c), self.q2(o, c))
        return q.detach().cpu().numpy()

    @torch.no_grad()
    def sample_candidates(self, obs: np.ndarray, K: int) -> np.ndarray:
        o = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return actor_service.sample(obs_np=o, K=K)[0]
