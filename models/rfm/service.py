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

class _RFMService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: RFMPolicy | None = None
        self.optim: optim.Optimizer | None = None
        self.obs_dim: int | None = None
        self.act_dim: int | None = None
        self.lr: float = 1e-4
        self.seed: int = 0
        self._prev_z: Tensor | None = None  # carries last batch of z

    def init(self, obs_dim: int, act_dim: int, lr: float = 1e-4, seed: int = 0, force: bool = False):
        if (self.model is not None and not force
            and self.obs_dim == obs_dim and self.act_dim == act_dim):
            return
        self.lr, self.seed = float(lr), int(seed)
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)
        self.model = RFMPolicy(self.obs_dim, self.act_dim).to(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        self._prev_z = None

    def update(self, obs_np: np.ndarray, z1_np: np.ndarray, w_np: np.ndarray) -> float:
        """Use previous-iteration z as base point (z0)."""
        assert self.model is not None and self.optim is not None
        obs = _flatten_obs(obs_np).to(self.device)
        z1  = torch.as_tensor(z1_np, device=self.device, dtype=torch.float32)
        w   = torch.as_tensor(w_np,  device=self.device, dtype=torch.float32)
        if z1.dim() == 1: z1 = z1.unsqueeze(0)
        if w.dim() == 0:  w  = w.unsqueeze(0)

        if obs.shape[1] != self.obs_dim:
            obs = obs[:, :self.obs_dim] if obs.shape[1] > self.obs_dim else torch.nn.functional.pad(
                obs, (0, self.obs_dim - obs.shape[1]))
        if z1.shape[1] != self.act_dim:
            raise ValueError(f"z1 dim {z1.shape[1]} != act_dim {self.act_dim}")

        if self._prev_z is None:
            raise RuntimeError("No previous z stored; initialize self._prev_z before first update.")
        z0 = self._prev_z.to(self.device)

        self.optim.zero_grad(set_to_none=True)
        loss = self.model.rfm_loss(obs, z0, z1, w)
        loss.backward()
        self.optim.step()

        self._prev_z = z1.detach().cpu()
        return float(loss.detach().cpu())

    @torch.no_grad()
    def sample(self, obs_np: np.ndarray, K: int, steps: int = 30) -> np.ndarray:
        assert self.model is not None
        obs = _flatten_obs(obs_np).to(self.device)
        if obs.shape[1] != self.obs_dim:
            obs = obs[:, :self.obs_dim] if obs.shape[1] > self.obs_dim else torch.nn.functional.pad(
                obs, (0, self.obs_dim - obs.shape[1]))
        Z = self.model.sample(obs, K=K, steps=steps)
        return Z.detach().cpu().numpy()


rfm_service = _RFMService()
