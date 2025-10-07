# models/ddpm/service.py
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from .policy import ScoreNet

def _flatten_obs(x: np.ndarray | Tensor) -> Tensor:
    t = torch.as_tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x.to(dtype=torch.float32)
    if t.dim() == 1: t = t.unsqueeze(0)
    if t.dim() >= 3: t = t.view(t.shape[0], -1)
    return t

class NoiseSchedule:
    def __init__(self, T: int, mode: str = "cosine"):
        self.T = T
        if mode == "linear":
            beta = torch.linspace(1e-4, 0.02, T)
        else:
            # cosine schedule (approx)
            steps = torch.arange(T+1, dtype=torch.float32)
            s = 0.008
            f = torch.cos(( (steps/T + s) / (1+s) ) * math.pi/2 )**2
            a_bar = f / f[0]
            beta = torch.clamp(1 - (a_bar[1:] / a_bar[:-1]), 1e-6, 0.999)
        alpha = 1.0 - beta
        a_bar = torch.cumprod(alpha, dim=0)
        self.alpha = alpha
        self.a_bar = a_bar
        self.sigma = torch.sqrt(1.0 - a_bar)

import math

class _DDPMService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: ScoreNet | None = None
        self.optim: optim.Optimizer | None = None
        self.obs_dim: int | None = None
        self.act_dim: int | None = None
        self.lr: float = 1e-4
        self.seed: int = 0
        self.T: int = 50
        self.sched: NoiseSchedule | None = None

    def init(self, obs_dim: int, act_dim: int, lr: float = 1e-4, seed: int = 0, T: int = 50, force: bool = False):
        if (self.model is not None and not force
                and self.obs_dim == obs_dim and self.act_dim == act_dim and self.T == T):
            return
        self.lr, self.seed, self.T = float(lr), int(seed), int(T)
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)
        self.model = ScoreNet(self.obs_dim, self.act_dim).to(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self.sched = NoiseSchedule(self.T, mode="cosine")

    # ---------- RSM LOSS (Eq. 5) ----------
    def update(self, obs_np: np.ndarray, c0_np: np.ndarray, w_np: np.ndarray) -> float:
        assert self.model is not None and self.optim is not None and self.sched is not None
        obs = _flatten_obs(obs_np).to(self.device)                # [B, F]
        c0  = torch.as_tensor(c0_np, device=self.device, dtype=torch.float32)
        w   = torch.as_tensor(w_np,  device=self.device, dtype=torch.float32)
        if c0.dim() == 1: c0 = c0.unsqueeze(0)
        if w.dim()  == 0: w  = w.unsqueeze(0)
        # clamp/pad obs to fixed F
        if obs.shape[1] != self.obs_dim:
            obs = obs[:, :self.obs_dim] if obs.shape[1] > self.obs_dim else torch.nn.functional.pad(
                obs, (0, self.obs_dim - obs.shape[1]))
        if c0.shape[1] != self.act_dim:
            raise ValueError(f"c0 dim {c0.shape[1]} != act_dim {self.act_dim}")

        B = c0.shape[0]
        T = self.T
        # sample t ∼ Uniform{1..T}
        t_idx = torch.randint(1, T+1, (B,), device=self.device)
        a_bar_t = self.sched.a_bar[t_idx-1].view(B,1)
        sigma_t = self.sched.sigma[t_idx-1].view(B,1)

        # sample c_t ~ N(sqrt(a_bar_t)*c0, (1-a_bar_t)I) (Eq. 3)
        eps = torch.randn_like(c0)
        c_t = torch.sqrt(a_bar_t) * c0 + sigma_t * eps

        # target conditional score
        # note: sigma_t^2 = 1 - a_bar_t
        target = -(c_t - torch.sqrt(a_bar_t) * c0) / (sigma_t**2 + 1e-8)

        self.optim.zero_grad(set_to_none=True)
        s_pred = self.model(obs, c_t, t_idx, T)
        loss_vec = (s_pred - target).pow(2).sum(dim=-1)
        # mirror-descent weighting
        loss = torch.mean(w * loss_vec)
        loss.backward()
        self.optim.step()
        return float(loss.detach().cpu())

    # ---------- SAMPLING (K candidates) ----------
    @torch.no_grad()
    def sample(self, obs_np: np.ndarray, K: int) -> np.ndarray:
        assert self.model is not None and self.sched is not None
        obs = _flatten_obs(obs_np).to(self.device) 
        if obs.shape[1] != self.obs_dim:
            obs = obs[:, :self.obs_dim] if obs.shape[1] > self.obs_dim else torch.nn.functional.pad(
                obs, (0, self.obs_dim - obs.shape[1]))

        B, D, T = obs.shape[0], self.act_dim, self.T
        c_t = torch.randn(B*K, D, device=self.device)
        obs_rep = obs.repeat_interleave(K, dim=0)

        # update using score net
        for t in range(T, 0, -1):
            t_idx = torch.full((B*K,), t, device=self.device, dtype=torch.long)
            a_bar_t = self.sched.a_bar[t-1].view(1,1)
            sigma_t = self.sched.sigma[t-1].view(1,1)
            s_pred = self.model(obs_rep, c_t, t_idx, T)

            # recover epsilon
            eps_hat = -sigma_t * s_pred
            
            c0_hat = (c_t - sigma_t * eps_hat) / torch.sqrt(a_bar_t + 1e-8)

            if t > 1:
                a_bar_prev = self.sched.a_bar[t-2].view(1,1)
                sigma_prev = torch.sqrt(1.0 - a_bar_prev)
                c_t = torch.sqrt(a_bar_prev) * c0_hat + sigma_prev * eps_hat
            else:
                c_t = c0_hat

        return c_t.view(B, K, D).detach().cpu().numpy()

ddpm_service = _DDPMService()
