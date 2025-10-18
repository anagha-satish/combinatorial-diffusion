# algos/dpmd_rf.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, NamedTuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from models.rfm.service import rfm_service

# -----------------------------
# Experience container
# -----------------------------
class Experience(NamedTuple):
    obs:    np.ndarray      # [B, F]
    action: np.ndarray      # [B, D]  (coefficients on S^{D-1})
    reward: np.ndarray      # [B]
    next_obs: np.ndarray    # [B, F]
    done: np.ndarray        # [B]

# -----------------------------
# Critic
# -----------------------------
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

# -----------------------------
# Utils
# -----------------------------
def _to_tensor(x, device, dtype=torch.float32) -> Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)

def _coerce_width(x: Tensor, target_F: int) -> Tensor:
    """Ensure x has last-dim = target_F by slicing or right-padding with zeros."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() >= 3:
        x = x.view(x.shape[0], -1)
    F = x.shape[-1]
    if F == target_F:
        return x
    if F > target_F:
        return x[:, :target_F].contiguous()
    # F < target_F -> pad
    pad = torch.zeros(x.shape[0], target_F - F, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)

@dataclass
class DPMDConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    tau: float = 0.005
    delay_update: int = 2
    reward_scale: float = 0.2
    num_particles: int = 8 # K: base policy candidates per state
    target_J: int = 4       # J: noisy copies per base for target-V
    lambda_temp: float = 1.0 # temperature in exp(Q / lambda)
    w_clip: Optional[float] = 50.0 # optional cap on weights to avoid blowups
    kappa_exec: float = 20.0 # vMF kappa for execution noise
    kappa_target: float = 20.0 # vMF kappa for target smoothing
    J_target_noise: int = 2    # how many noisy draws per base candidate

class DPMD:
    """DPMD with RFM policy (the actor lives in rfm_service)."""
    def __init__(self, obs_dim: int, act_dim: int,
                 device: Optional[torch.device] = None,
                 cfg: DPMDConfig = DPMDConfig()):
        self.cfg = cfg
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # critics + target critics
        self.q1 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.q2 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.tq1 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.tq2 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())

        # joint optimizer for both online critics
        self.q_optim = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()),
                                  lr=self.cfg.lr)

        self.step = 0


    @torch.no_grad()
    def _target_v(self, next_obs: Tensor) -> Tensor:
        """
        Smoothed Bellman: for each s', sample base candidates from policy,
        then for each candidate draw J noisy copies via vMF, and average min-Q.
        """
        next_obs = _coerce_width(next_obs, self.obs_dim)
        B = next_obs.shape[0]

        # Add vMF noise around policy candidates: K bases × J noisy copies
        cand = rfm_service.sample(next_obs.detach().cpu().numpy(),
                                  K=self.cfg.num_particles, steps=30,
                                  kappa=self.cfg.kappa_target,
                                  J_noise=self.cfg.J_target_noise) 
        cand_t = _to_tensor(cand, self.device)        
        Kc = cand_t.shape[1]

        rep_obs = next_obs.unsqueeze(1).expand(B, Kc, self.obs_dim).reshape(B * Kc, self.obs_dim)
        flat_cand = cand_t.reshape(B * Kc, self.act_dim)

        # Evaluate target critics and take min (TD3)
        q1 = self.tq1(rep_obs, flat_cand).view(B, Kc)
        q2 = self.tq2(rep_obs, flat_cand).view(B, Kc)
        return torch.minimum(q1, q2).mean(dim=1)


    def update(self, batch: Experience) -> Dict[str, float]:
        """" Main training step for DPMD.
        1. Critic update via Bellman error
        2. Actor update via RFM loss with weights from target critics
        3. Slow target network update """""
        obs = _to_tensor(batch.obs, self.device).view(len(batch.obs), -1)
        act = _to_tensor(batch.action, self.device)
        rew = _to_tensor(batch.reward, self.device).view(-1)
        nxt = _to_tensor(batch.next_obs, self.device).view(len(batch.next_obs), -1)
        done = _to_tensor(batch.done, self.device).view(-1)

        obs = _coerce_width(obs, self.obs_dim)
        nxt = _coerce_width(nxt, self.obs_dim)

        rew = rew * self.cfg.reward_scale

        # -------- target value --------
        with torch.no_grad():
            vhat = self._target_v(nxt)
            y = rew + (1.0 - done) * self.cfg.gamma * vhat

        # -------- critic step --------
        self.q_optim.zero_grad(set_to_none=True)
        q1_pred = self.q1(obs, act)
        q2_pred = self.q2(obs, act)
        q1_loss = torch.mean((q1_pred - y) ** 2)
        q2_loss = torch.mean((q2_pred - y) ** 2)
        (q1_loss + q2_loss).backward()
        self.q_optim.step()

        # -------- actor weights --------
        with torch.no_grad():
            # online min-Q 
            q_min_online = torch.minimum(self.q1(obs, act), self.q2(obs, act))

            # target critics for mirror-descent energy
            q_min_target = torch.minimum(self.tq1(obs, act), self.tq2(obs, act))

            # init EMA
            if not hasattr(self, "_w_mean"):
                self._w_mean = q_min_target.mean().item()
                self._w_std  = q_min_target.std().item() + 1e-6
            # EMA update
            self._w_mean = 0.995 * self._w_mean + 0.01 * q_min_target.mean().item()
            self._w_std  = 0.995 * self._w_std  + 0.01 * (q_min_target.std().item() + 1e-6)

            # normalize target-Qs for weights
            q_center_w = (q_min_target - self._w_mean) / self._w_std
            w = torch.exp(q_center_w / self.cfg.lambda_temp)
            if self.cfg.w_clip is not None:
                w = torch.clamp(w, max=float(self.cfg.w_clip))

        # one RFM update step
        policy_loss = rfm_service.update(obs.detach().cpu().numpy(),
                                        act.detach().cpu().numpy(),
                                        w.detach().cpu().numpy())

        # -------- slow target update --------
        if (self.step % self.cfg.delay_update) == 0:
            with torch.no_grad():
                for p_t, p in zip(self.tq1.parameters(), self.q1.parameters()):
                    p_t.mul_(1.0 - self.cfg.tau).add_(p, alpha=self.cfg.tau)
                for p_t, p in zip(self.tq2.parameters(), self.q2.parameters()):
                    p_t.mul_(1.0 - self.cfg.tau).add_(p, alpha=self.cfg.tau)

        self.step += 1

        # return dict 
        return {
            "q1_loss": float(q1_loss.detach().cpu()),
            "q2_loss": float(q2_loss.detach().cpu()),
            "policy_loss": float(policy_loss),
            "q_mean": float(q_min_online.mean().detach().cpu()),
            "q_std": float(q_min_online.std().detach().cpu()),
            "qw_mean": float(q_min_target.mean().detach().cpu()),
            "qw_std": float(q_min_target.std().detach().cpu()),
        }


    @torch.no_grad()
    def score_actions(self, obs: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Score a batch of candidate actions C for a single observation obs."""
        o = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1),
                            device=self.device)
        c = torch.as_tensor(C, device=self.device, dtype=torch.float32)
        o_rep = o.repeat(c.shape[0], 1)
        q = torch.minimum(self.q1(o_rep, c), self.q2(o_rep, c))
        return q.detach().cpu().numpy()

    @torch.no_grad()
    def sample_candidates(self, obs: np.ndarray, K: int) -> np.ndarray:
        """Sample K candidate actions from the current actor."""
        o = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return rfm_service.sample(obs_np=o, K=K, steps=30, kappa=None, J_noise=1)[0]
