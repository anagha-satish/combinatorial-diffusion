# algos/dpmd_rf.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, NamedTuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from models.rfm.service import rfm_service

# -----------------------------
# Replay experience tuple
# -----------------------------
class Experience(NamedTuple):
    obs:         np.ndarray  # [B, F]
    action:      np.ndarray  # [B, D]  executed coefficient \tilde c
    reward:      np.ndarray  # [B]
    next_obs:    np.ndarray  # [B, F]
    done:        np.ndarray  # [B]     (0/1)
    action_star: np.ndarray  # [B, D]  greedy c* sampled under π_old
    policy_id:   np.ndarray  # [B]     integer policy version that produced the sample

# -----------------------------
# Twin critic (min over Q1,Q2)
# -----------------------------
class QNet(nn.Module):
    """Approximates Q_e(s, c) on S^{D-1} with a simple MLP."""
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
# Small utils
# -----------------------------
def _to_tensor(x, device, dtype=torch.float32) -> Tensor:
    if isinstance(x, np.ndarray):
        if not x.flags.writeable:
            x = np.array(x, copy=True)
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        return torch.tensor(x, device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)

def _fit_width(x: Tensor, target_F: int) -> Tensor:
    """Ensure last-dim == target_F by slice or right-pad with zeros."""
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

# -----------------------------
# Config (hyperparams)
# -----------------------------
@dataclass
class DPMDConfig:
    # RL
    gamma: float = 0.99
    lr: float = 3e-4
    tau: float = 0.005          # target soft-update coeff
    delay_update: int = 2       # update targets every N steps
    reward_scale: float = 0.2

    # Actor sampling / candidate eval
    num_particles: int = 8      # K candidates per state

    # Mirror-descent temperature for weights
    lambda_temp: float = 0.7
    w_clip: Optional[float] = 50.0

    # Execution noise (on-sphere)
    kappa_exec: float = 20.0    # vMF κ for executed coefficients

    # Smoothed Bellman operator
    kappa_smooth: float = 20.0  # vMF κ for target smoothing
    M_smooth: int = 8           # #policy samples c' ~ π_target(·|s')
    J_smooth: int = 1           # #vMF perturbations per c'

class DPMD:
    """
    DPMD with RFM policy on S^{D-1}.
      • Critics train to smoothed Bellman targets
      • Actor trains via MD-weighted RFM using target critics
    """
    def __init__(self, obs_dim: int, act_dim: int,
                 device: Optional[torch.device] = None,
                 cfg: DPMDConfig = DPMDConfig()):
        self.cfg = cfg
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Critics and targets (twin Q; use min)
        self.q1 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.q2 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.tq1 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.tq2 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())

        self.q_optim = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.cfg.lr
        )

        # MD weight baseline (EMA of target-critic values)
        self._w_mean: Optional[float] = None
        self._w_std: Optional[float] = None
        self.step = 0
        self.policy_version = 0

        # Keep target actor aligned initially
        if hasattr(rfm_service, "sync_target_from_current"):
            rfm_service.sync_target_from_current()

    # -----------------------------
    # Actor helpers
    # -----------------------------
    @torch.no_grad()
    def _sample_candidates(self, obs: Tensor, K: int, *, use_target: bool = False) -> Tensor:
        """Draw K coefficients from current (or target) policy on S^{D-1}."""
        obs = _fit_width(obs, self.obs_dim)
        if use_target and hasattr(rfm_service, "sample_target"):
            C_np = rfm_service.sample_target(obs_np=obs.detach().cpu().numpy(),
                                             K=K, steps=30, kappa=None, J_noise=1)
        else:
            C_np = rfm_service.sample(obs_np=obs.detach().cpu().numpy(),
                                      K=K, steps=30, kappa=None, J_noise=1)
        return _to_tensor(C_np, self.device)  # [B, K, D]

    @torch.no_grad()
    def _greedy_from_candidates(self, obs: Tensor, C: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Pick argmax_i min(Q1,Q2)(s, c_i) among candidates C. Also report target critics at argmax."""
        obs = _fit_width(obs, self.obs_dim)           # [B,F]
        B, K, D = C.shape
        rep_obs = obs.unsqueeze(1).expand(B, K, self.obs_dim).reshape(B * K, self.obs_dim)
        flat_C = C.reshape(B * K, D)

        # Online min-Q
        q1 = self.q1(rep_obs, flat_C).view(B, K)
        q2 = self.q2(rep_obs, flat_C).view(B, K)
        q_min = torch.minimum(q1, q2)

        idx = torch.argmax(q_min, dim=1)              # [B]
        a_star = C[torch.arange(B, device=self.device), idx]  # [B,D]
        q_star_online = q_min[torch.arange(B, device=self.device), idx]

        # Target critics on the same chosen candidates
        tq1 = self.tq1(rep_obs, flat_C).view(B, K)
        tq2 = self.tq2(rep_obs, flat_C).view(B, K)
        tmin = torch.minimum(tq1, tq2)
        q_star_target = tmin[torch.arange(B, device=self.device), idx]
        return a_star, q_star_online, q_star_target

    # -----------------------------
    # Smoothed Bellman target V^b_κ(s')
    # -----------------------------
    @torch.no_grad()
    def _smoothed_value(self, next_obs: Tensor) -> Tensor:
        """
        Monte-Carlo approximation of V^b_κ(s'):
        """
        next_obs = _fit_width(next_obs, self.obs_dim)  # [B,F]
        B = next_obs.shape[0]
        M = max(1, int(self.cfg.M_smooth))
        J = max(1, int(self.cfg.J_smooth))
        kappa = float(self.cfg.kappa_smooth)

        Cprime = self._sample_candidates(next_obs, M, use_target=True)  # [B,M,D]

        # Perturb each c' with vMF on the sphere (J times)
        perturbed = []
        for m in range(M):
            cm = Cprime[:, m]  # [B,D]
            cm_tilde = rfm_service.perturb(cm.detach().cpu().numpy(), kappa=kappa, J=J)  # [B,J,D]
            perturbed.append(_to_tensor(cm_tilde, self.device))
        Chat = torch.stack(perturbed, dim=1)  # [B,M,J,D]

        rep_obs = next_obs.view(B, 1, 1, self.obs_dim).expand(B, M, J, self.obs_dim).reshape(B*M*J, self.obs_dim)
        flat_c  = Chat.reshape(B * M * J, self.act_dim)

        q1 = self.tq1(rep_obs, flat_c).view(B, M, J)
        q2 = self.tq2(rep_obs, flat_c).view(B, M, J)
        return torch.minimum(q1, q2).mean(dim=(1, 2))  # [B]

    # -----------------------------
    # Main update
    # -----------------------------
    def update(self, batch: Experience) -> Dict[str, float]:
        # Parse batch
        obs      = _fit_width(_to_tensor(batch.obs,      self.device), self.obs_dim)
        act_exec = _to_tensor(batch.action,  self.device)
        rew      = _to_tensor(batch.reward,  self.device).view(-1)
        nxt      = _fit_width(_to_tensor(batch.next_obs, self.device), self.obs_dim)
        done     = _to_tensor(batch.done,    self.device).view(-1)

        # Optional reward scaling
        rew = rew * self.cfg.reward_scale

        # 1) Smoothed target
        with torch.no_grad():
            Vb = self._smoothed_value(nxt)
            y  = rew + (1.0 - done) * self.cfg.gamma * Vb

        # 2) Critic updates
        self.q_optim.zero_grad(set_to_none=True)
        q1_loss = torch.mean((self.q1(obs, act_exec) - y) ** 2)
        q2_loss = torch.mean((self.q2(obs, act_exec) - y) ** 2)
        (q1_loss + q2_loss).backward()
        self.q_optim.step()

        # 3) MD weights from target critics on (s_old, c1_old=c*)
        with torch.no_grad():
            s_old  = _fit_width(_to_tensor(batch.obs,         self.device), self.obs_dim)
            c1_old = _to_tensor(batch.action_star, self.device)
            tq1 = self.tq1(s_old, c1_old)
            tq2 = self.tq2(s_old, c1_old)
            q_min_t = torch.minimum(tq1, tq2)  # [B]
            # True MD weight
            lam = max(self.cfg.lambda_temp, 1e-6)
            w = torch.exp(q_min_t / lam)
            if self.cfg.w_clip is not None:
                w = torch.clamp(w, max=float(self.cfg.w_clip))

        # 4) Train RFM actor
        policy_loss = rfm_service.update(
            s_old.detach().cpu().numpy(),
            c1_old.detach().cpu().numpy(),
            w.detach().cpu().numpy(),
        )

        # 5) Soft-update target critics and target actor
        if (self.step % self.cfg.delay_update) == 0:
            with torch.no_grad():
                for p_t, p in zip(self.tq1.parameters(), self.q1.parameters()):
                    p_t.mul_(1.0 - self.cfg.tau).add_(p, alpha=self.cfg.tau)
                for p_t, p in zip(self.tq2.parameters(), self.q2.parameters()):
                    p_t.mul_(1.0 - self.cfg.tau).add_(p, alpha=self.cfg.tau)
            if hasattr(rfm_service, "soft_update_target"):
                rfm_service.soft_update_target(self.cfg.tau)
            elif hasattr(rfm_service, "sync_target_from_current"):
                rfm_service.sync_target_from_current()

        self.step += 1

        # Logging
        with torch.no_grad():
            C = self._sample_candidates(nxt, self.cfg.num_particles, use_target=False)
            _, q_star_online, q_star_target = self._greedy_from_candidates(nxt, C)

        return {
            "q1_loss": float(q1_loss.detach().cpu()),
            "q2_loss": float(q2_loss.detach().cpu()),
            "policy_loss": float(policy_loss),
            "q_mean_next_online": float(q_star_online.mean().detach().cpu()),
            "q_std_next_online": float(q_star_online.std().detach().cpu()),
            "q_mean_next_target": float(q_star_target.mean().detach().cpu()),
            "q_std_next_target": float(q_star_target.std().detach().cpu()),
        }


    # -----------------------------
    # Public driver helpers
    # -----------------------------
    @torch.no_grad()
    def score_actions(self, obs: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Return min(Q1,Q2)(s, c_i) for candidate set C (used to pick i*)."""
        o = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), device=self.device)
        c = torch.as_tensor(C, device=self.device, dtype=torch.float32)
        o_rep = o.repeat(c.shape[0], 1)
        q = torch.minimum(self.q1(o_rep, c), self.q2(o_rep, c))
        return q.detach().cpu().numpy()

    @torch.no_grad()
    def sample_candidates(self, obs: np.ndarray, K: int) -> np.ndarray:
        """Draw K coefficient candidates from current actor πθ(·|s)."""
        o = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return rfm_service.sample(obs_np=o, K=K, steps=30, kappa=None, J_noise=1)[0]

__all__ = ["DPMD", "DPMDConfig", "Experience"]
