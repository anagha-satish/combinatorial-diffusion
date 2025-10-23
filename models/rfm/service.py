# models/rfm/service.py
from __future__ import annotations
import numpy as np
import torch
import torch.optim as optim
from torch import Tensor
from .policy import RFMPolicy, normalize

def _flatten_obs(x: np.ndarray | Tensor) -> Tensor:
    t = torch.as_tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x.to(dtype=torch.float32)
    if t.dim() == 1: t = t.unsqueeze(0)
    if t.dim() >= 3: t = t.view(t.shape[0], -1)
    return t

# Helper: approximate von Mises–Fisher sampling on the unit sphere
@torch.no_grad()
def _vmf_perturb(mu: Tensor, kappa: float, J: int) -> Tensor:
    """
    Approximate vMF(μ, κ) samples on S^{D-1}
    """
    assert mu.dim() == 2
    B, D = mu.shape
    mu = normalize(mu)
    eps = torch.randn(B, J, D, device=mu.device, dtype=mu.dtype)
    # project noise to tangent at μ
    dot = (eps * mu.unsqueeze(1)).sum(-1, keepdim=True)
    eps_tan = eps - dot * mu.unsqueeze(1)
    sigma = (1.0 / (float(kappa) + 1e-6)) ** 0.5
    z = mu.unsqueeze(1) + sigma * eps_tan
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    return z  # [B, J, D]

# Core class: wraps the Riemannian Flow Matching policy network
class _RFMService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: RFMPolicy | None = None
        self.model_target: RFMPolicy | None = None
        self.optim: optim.Optimizer | None = None
        self.obs_dim: int | None = None
        self.act_dim: int | None = None
        self.lr: float | None = None
        self.seed: int = 0

    def init(self, obs_dim: int, act_dim: int, lr: float, seed: int = 0, force: bool = False):
        self.lr, self.seed = float(lr), int(seed)
        # Set seeds for reproducibility
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        # Save shape info
        self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)
        # Instantiate Riemannian Flow policy
        self.model = RFMPolicy(self.obs_dim, self.act_dim).to(self.device)
        self.model_target = RFMPolicy(self.obs_dim, self.act_dim).to(self.device)
        self.model_target.load_state_dict(self.model.state_dict())
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)

    # One gradient update step for the actor.
    def update(self, obs_np: np.ndarray, z1_np: np.ndarray, w_np: np.ndarray) -> float:
        assert self.model is not None and self.optim is not None

        obs = _flatten_obs(obs_np).to(self.device)  # [B, F]
        z1  = torch.as_tensor(z1_np, device=self.device, dtype=torch.float32)
        w   = torch.as_tensor(w_np,  device=self.device, dtype=torch.float32)
        if z1.dim() == 1: z1 = z1.unsqueeze(0)
        if w.dim()  == 0: w  = w.unsqueeze(0)

        if obs.shape[1] != self.obs_dim:
            obs = obs[:, :self.obs_dim] if obs.shape[1] > self.obs_dim else torch.nn.functional.pad(
                obs, (0, self.obs_dim - obs.shape[1]))
        if z1.shape[1] != self.act_dim:
            raise ValueError(f"z1 dim {z1.shape[1]} != act_dim {self.act_dim}")

        # Compute RFM loss and apply gradient step
        self.optim.zero_grad(set_to_none=True)
        loss = self.model.rfm_loss(obs, z1, w)
        loss.backward()
        self.optim.step()
        return float(loss.detach().cpu())

    # Generate samples (coefficients c) from the learned flow
    @torch.no_grad()
    def sample(self, obs_np: np.ndarray, K: int, steps: int = 30, *, kappa: float | None = None, J_noise: int = 1) -> np.ndarray:
        assert self.model is not None
        obs = _flatten_obs(obs_np).to(self.device)
        if obs.shape[1] != self.obs_dim:
            obs = obs[:, :self.obs_dim]
        Z = self.model.sample(obs, K=K, steps=steps)  # [B,K,D]
        if (kappa is None) or (J_noise <= 0) or (float(kappa) <= 0):
            return Z.detach().cpu().numpy()
        B, K_, D = Z.shape
        mu = Z.reshape(B * K_, D)
        Z_noisy = _vmf_perturb(mu, kappa=float(kappa), J=int(J_noise))  # [B*K,J,D]
        Z_noisy = Z_noisy.reshape(B, K_ * int(J_noise), D)
        return Z_noisy.detach().cpu().numpy()

    @torch.no_grad()
    def sample_target(self, obs_np: np.ndarray, K: int, steps: int = 30, *, kappa: float | None = None, J_noise: int = 1) -> np.ndarray:
        """Sample from the target actor."""
        assert self.model_target is not None
        obs = _flatten_obs(obs_np).to(self.device)
        if obs.shape[1] != self.obs_dim:
            obs = obs[:, :self.obs_dim]
        Z = self.model_target.sample(obs, K=K, steps=steps)
        if (kappa is None) or (J_noise <= 0) or (float(kappa) <= 0):
            return Z.detach().cpu().numpy()
        B, K_, D = Z.shape
        mu = Z.reshape(B * K_, D)
        Z_noisy = _vmf_perturb(mu, kappa=float(kappa), J=int(J_noise))
        Z_noisy = Z_noisy.reshape(B, K_ * int(J_noise), D)
        return Z_noisy.detach().cpu().numpy()

    @torch.no_grad()
    def sync_target_from_current(self) -> None:
        if self.model is None or self.model_target is None: return
        self.model_target.load_state_dict(self.model.state_dict())

    @torch.no_grad()
    def soft_update_target(self, tau: float) -> None:
        if self.model is None or self.model_target is None: return
        with torch.no_grad():
            for p_t, p in zip(self.model_target.parameters(), self.model.parameters()):
                p_t.mul_(1.0 - tau).add_(p, alpha=tau)

    @torch.no_grad()
    def perturb(self, mu_np: np.ndarray, kappa: float, J: int = 1) -> np.ndarray:
        """
        Draw J vMF-like samples around each mean direction μ on S^{D-1}.
        """
        mu = torch.as_tensor(mu_np, device=self.device, dtype=torch.float32)
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)
        mu = normalize(mu)
        Z = _vmf_perturb(mu, kappa=float(kappa), J=int(J))  # [B, J, D]
        return Z.detach().cpu().numpy()

rfm_service = _RFMService()
