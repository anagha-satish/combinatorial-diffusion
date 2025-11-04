# models/rfm/policy.py
import torch
import torch.nn as nn
from torch import Tensor

# ---------- S^D geometry helpers ----------
def normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Project any vector(s) to the unit sphere."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def sphere_logmap(p: Tensor, x: Tensor, eps: float = 1e-8) -> Tensor:  # Log_p(x)
    """Riemannian log map on the sphere: a tangent vector at p."""
    # <p,x> = cos θ (clamped for stability)
    dot = (p * x).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    v = x - dot * p                                   # x - cosθ * p
    theta = torch.arccos(dot)                         # θ = arccos(<p,x>)
    nv = v.norm(dim=-1, keepdim=True).clamp_min(eps)  # ||x - cosθ p|| = sinθ
    return v * (theta / nv)                           # (θ/sinθ) * (x - cosθ p)

def sphere_expmap(p: Tensor, v: Tensor, eps: float = 1e-8) -> Tensor:  # Exp_p(v)
    """Exponential map: move from p along tangent v by distance ||v||."""
    nv = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    return normalize(p * torch.cos(nv) + v * (torch.sin(nv) / nv))

def geodesic_velocity(p: Tensor, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """(dc/dt at time t, c_t) along the geodesic from p to x on S^{D-1}."""
    xi = sphere_logmap(p, x)  # Log_p(x)
    xi_norm = xi.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    t = t.view(-1, 1)
    c_t = sphere_expmap(p, t * xi)    # geodesic interp
    a = -(torch.sin(t * xi_norm) * xi_norm) * p
    b = torch.cos(t * xi_norm) * xi
    dcdt = a + b
    return dcdt - (dcdt * c_t).sum(dim=-1, keepdim=True) * c_t, c_t  # project back to tangent


# ---------- Time embedding ----------
class TimeEmbed(nn.Module):
    """
    Sinusoidal Fourier features for t in [0,1] with a tiny MLP head.
    Output dim == embed_dim.
    """
    def __init__(self, embed_dim: int = 32, L: int = 16):
        """
        embed_dim: final output dim (after MLP)
        L: number of harmonics (produces 2*L raw features: sin, cos)
        """
        super().__init__()
        self.L = int(L)
        self.embed_dim = int(embed_dim)
        freqs = 2.0 ** torch.arange(self.L) * torch.pi
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Sequential(
            nn.Linear(2 * self.L, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """
        t: [B,1] or [B] in [0,1]
        returns: [B, embed_dim]
        """
        if t.dim() == 1:
            t = t.view(-1, 1)
        # [B, L]
        x = t.to(dtype=self.freqs.dtype) * self.freqs.view(1, -1)
        pe = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # [B, 2L]
        return self.proj(pe)


# ---------- RFM policy ----------
class RFMPolicy(nn.Module):
    """u_theta(obs, z_t, t) -> tangent vector on S^{D-1}."""
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        # time embedding (32-d output, 16 harmonics) -> adjust if you prefer
        self.tok = TimeEmbed(embed_dim=32, L=16)
        in_dim = obs_dim + act_dim + 32

        layers: list[nn.Module] = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.SiLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, act_dim)]
        self.net = nn.Sequential(*layers)
        self.act_dim = act_dim

    def forward(self, obs: Tensor, z_t: Tensor, t: Tensor) -> Tensor:
        """If z_t is batched, make obs and t match that batch size."""
        if z_t.dim() == 2:
            B = z_t.shape[0]
            if obs.dim() == 1:
                obs = obs.unsqueeze(0).expand(B, -1)
            if t.dim() == 0:
                t = t.expand(B)
            elif t.dim() == 1 and t.shape[0] != B:
                t = t[:1].expand(B)
        else:
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            if t.dim() == 0:
                t = t.unsqueeze(0)

        # embed time
        if t.dim() == 1:
            t = t.view(-1, 1)
        tfeat = self.tok(t)

        u = self.net(torch.cat([obs, z_t, tfeat], dim=-1))
        # project to tangent at z_t (stay on S^{D-1})
        return u - (u * z_t).sum(dim=-1, keepdim=True) * z_t

    @torch.no_grad()
    def sample(self, obs: Tensor, K: int, steps: int = 30) -> Tensor:
        """
        Return [B, K, D] samples on the sphere using a 2nd-order (Heun) integrator.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B, D = obs.shape[0], self.act_dim
        z = normalize(torch.randn(B * K, D, device=obs.device, dtype=obs.dtype))
        obs_rep = obs.repeat_interleave(K, dim=0)
        dt = 1.0 / steps
        t_grid = torch.linspace(0, 1, steps+1, device=obs.device, dtype=obs.dtype)
        for i in range(steps):
            # k1 at (z, t_i)
            t_i, t_ip1 = t_grid[i], t_grid[i+1]
            u_i = self.forward(obs_rep, z, t_i)  # tangent velocity

            # provisional Euler step
            z_tilde = sphere_expmap(z, dt * u_i)

            # k2 at (z_tilde, t_{i+1})
            u_ip1 = self.forward(obs_rep, z_tilde, t_ip1)

            # Heun average (RK2)
            u_heun = 0.5 * (u_i + u_ip1)
            z = sphere_expmap(z, dt * u_heun)

        return z.view(B, K, D)

    # ----------- stable core loss -----------
    def _rfm_loss_core(self, obs: Tensor, z1: Tensor, w: Tensor, *,
                       t: Tensor | None = None,
                       t_eps: float = 0.1) -> Tensor:
        """
        Weighted Riemannian flow-matching loss on S^{D-1} for a batch of endpoints z1.
        Uses antithetic sampling for t if t is not provided (variance reduction).
        """
        # ensure batched
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if z1.dim() == 1:
            z1 = z1.unsqueeze(0)
        if w.dim() == 0:
            w = w.unsqueeze(0)

        B = z1.shape[0]

        # sample z0 ~ Uniform(S^{D-1})
        z0 = normalize(torch.randn_like(z1))

        # sample t ~ U[t_eps, 1 - t_eps]
        if t is None:
            if B >= 2:
                half = (B + 1) // 2
                t_half = torch.rand(half, device=z1.device, dtype=z1.dtype)
                t_full = torch.cat([t_half, 1.0 - t_half[:B - half]], dim=0)
                t = t_full
            else:
                t = torch.rand(B, device=z1.device, dtype=z1.dtype)
        else:
            if t.dim() == 0:
                t = t.unsqueeze(0)
        t = t.clamp(t_eps, 1.0 - t_eps)

        # geodesic interpolation
        xi_01 = sphere_logmap(z0, z1)                  # Log_{z0}(z1)
        z_t   = sphere_expmap(z0, t.view(-1, 1) * xi_01)

        # MD target velocity (geodesic to z1 from z_t over remaining time)
        log_ct_z1 = sphere_logmap(z_t, z1)             # Log_{c_t}(z1)
        proj = log_ct_z1 - (log_ct_z1 * z_t).sum(dim=-1, keepdim=True) * z_t
        inv_1mt = 1.0 / (1.0 - t).view(-1, 1)          # bounded by t_eps above
        u_star = proj * inv_1mt

        # model velocity
        u = self.forward(obs, z_t, t)

        err = (u - u_star).pow(2).sum(dim=-1)
        return (w * err).mean()

    def rfm_loss(self, obs: Tensor, z1: Tensor, w: Tensor) -> Tensor:
        """Backward-compatible single-endpoint loss (per sample weights)."""
        return self._rfm_loss_core(obs, z1, w)
