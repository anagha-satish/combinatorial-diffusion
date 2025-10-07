# models/ddpm/policy.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch import Tensor

# ---------- sinusoidal t-emb ----------
class TimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, dim),
        )
    def forward(self, t_idx: Tensor, T: int) -> Tensor:
        # t_idx: [B] integers in [1..T]
        device = t_idx.device
        half = self.dim // 2
        t = t_idx.float() / float(T)
        freqs = torch.exp(torch.linspace(math.log(1e-4), math.log(1.0), half, device=device))
        ang = t.view(-1,1) * freqs.view(1,-1) * 2*math.pi
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if self.dim % 2 == 1:  # pad if odd
            emb = torch.cat([emb, torch.zeros(emb.shape[0],1, device=device)], dim=-1)
        return self.mlp(emb)

# ---------- simple MLP score net sθ(c_t ; s, t) ----------
class ScoreNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hid=(256,256), tdim=128):
        super().__init__()
        self.tok = TimeEmbed(tdim)
        in_dim = obs_dim + act_dim + tdim
        layers = []
        prev = in_dim
        for h in hid:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        layers += [nn.Linear(prev, act_dim)]
        self.net = nn.Sequential(*layers)
        self.obs_dim, self.act_dim, self.tdim = obs_dim, act_dim, tdim

    def forward(self, obs: Tensor, c_t: Tensor, t_idx: Tensor, T: int) -> Tensor:
        if obs.dim() == 1:  obs = obs.unsqueeze(0)
        if c_t.dim() == 1:  c_t = c_t.unsqueeze(0)
        if t_idx.dim() == 0: t_idx = t_idx.unsqueeze(0)
        t_emb = self.tok(t_idx, T)
        x = torch.cat([obs, c_t, t_emb], dim=-1)
        return self.net(x)
