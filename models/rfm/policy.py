# models/rfm/policy.py
import torch
import torch.nn as nn
from torch import Tensor

# ---------- S^D geometry helpers ----------
def normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def sphere_logmap(p: Tensor, x: Tensor, eps: float = 1e-8) -> Tensor: # Log_c
    dot = (p * x).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6) # <p,x> = cos theta
    v = x - dot * p # x - costheta * p
    theta = torch.arccos(dot) # theta = arccos(<p,x>)
    nv = v.norm(dim=-1, keepdim=True).clamp_min(eps) #||x-(costheta)p|| = sintheta
    return v * (theta / nv) # (theta/sintheta) * (x - costheta * p)

def sphere_expmap(p: Tensor, v: Tensor, eps: float = 1e-8) -> Tensor: #Exp_c
    nv = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    return normalize(p * torch.cos(nv) + v * (torch.sin(nv) / nv))

def geodesic_velocity(p: Tensor, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """Velocity and point along the geodesic z(t) from p to x."""
    xi = sphere_logmap(p, x) #Log_c
    xi_norm = xi.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    t = t.view(-1, 1)
    c_t = sphere_expmap(p, t * xi)   #c_t = Exp_c(t * Log_c0(c1))
    a = -(torch.sin(t * xi_norm) * xi_norm) * p
    b = torch.cos(t * xi_norm) * xi
    dcdt = a + b #derivative of c_t, closed form
    return dcdt - (dcdt * c_t).sum(dim=-1, keepdim=True) * c_t, c_t # projects back to sphere


# ---------- RFM policy ----------
class RFMPolicy(nn.Module):
    """u_theta(obs, z_t, t) -> tangent vector on S^{D-1}."""
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        in_dim = obs_dim + act_dim + 1
        layers: list[nn.Module] = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, act_dim)]
        self.net = nn.Sequential(*layers)
        self.act_dim = act_dim

    def forward(self, obs: Tensor, z_t: Tensor, t: Tensor) -> Tensor:
        # If z_t is batched, make obs and t match that batch size, fixing previous issue of unsqueezing obs
        if z_t.dim() == 2:
            B = z_t.shape[0]
            if obs.dim() == 1:
                obs = obs.unsqueeze(0).expand(B, -1)
            if t.dim() == 0:
                t = t.repeat(B)
            if t.dim() == 1 and t.shape[0] != B:
                t = t[:1].repeat(B)
        else:
            if obs.dim() == 1:  obs = obs.unsqueeze(0)
            if t.dim() == 0:    t   = t.unsqueeze(0)

        if t.dim() == 1:
            t = t.view(-1, 1)
        u = self.net(torch.cat([obs, z_t, t], dim=-1))
        # project to tangent at z_t (stay on S^{D-1})
        return u - (u * z_t).sum(dim=-1, keepdim=True) * z_t


    @torch.no_grad()
    def sample(self, obs: Tensor, K: int, steps: int = 30) -> Tensor:
        """Return [B, K, D] samples on the sphere."""
        if obs.dim() == 1: obs = obs.unsqueeze(0)
        B, D = obs.shape[0], self.act_dim
        z = normalize(torch.randn(B * K, D, device=obs.device))
        obs_rep = obs.repeat_interleave(K, dim=0)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B * K,), i * dt, device=obs.device)
            u = self.forward(obs_rep, z, t)
            z = sphere_expmap(z, dt * u)
        return z.view(B, K, D)

    def rfm_loss(self, obs: Tensor, z0: Tensor, z1: Tensor, w: Tensor) -> Tensor:
        """
        Weighted Riemannian Flow Matching loss using a supplied base point z0
        (from previous iteration), and target endpoint z1.
        No normalization is applied to z1 here (caller controls that).
        """
        if obs.dim() == 1: obs = obs.unsqueeze(0)
        if z0.dim() == 1:  z0  = z0.unsqueeze(0)
        if z1.dim() == 1:  z1  = z1.unsqueeze(0)
        if w.dim() == 0:   w   = w.unsqueeze(0)

        # ensure base point is on S^{D-1} (samples from sample() already are; this is safety)
        z0 = normalize(z0)

        B, _ = z1.shape
        t = torch.rand(B, device=z1.device)  # t ~ U(0,1)

        with torch.no_grad():
            u_star, z_t = geodesic_velocity(z0, z1, t)  # target tangent field on path

        u = self.forward(obs, z_t, t)
        err = (u - u_star).pow(2).sum(dim=-1)
        return (w * err).mean()
