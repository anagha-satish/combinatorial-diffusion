# models/rfm/service_gnn.py
from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim
from torch import Tensor
from torch_geometric.data import Batch

from .policy_gnn import RFMPolicyGNN, normalize


# Helper: approximate von Mises–Fisher sampling on the unit sphere
@torch.no_grad()
def _vmf_perturb_torch(mu: Tensor, kappa: float, J: int) -> Tensor:
    """
    Simple Gaussian-in-tangent approximation to vMF around mean direction mu.
    Returns [B, J, D].
    """
    assert mu.dim() == 2
    B, D = mu.shape
    mu = normalize(mu)
    eps = torch.randn(B, J, D, device=mu.device, dtype=mu.dtype)
    dot = (eps * mu.unsqueeze(1)).sum(-1, keepdim=True)
    eps_tan = eps - dot * mu.unsqueeze(1)
    sigma = (1.0 / (float(kappa) + 1e-6)) ** 0.5
    z = mu.unsqueeze(1) + sigma * eps_tan
    return normalize(z)


class _RFMServiceGNN:
    """
    Mirror of service.py, but:
      - obs is a torch_geometric Batch
      - model is RFMPolicyGNN
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: RFMPolicyGNN | None = None
        self.optim: optim.Optimizer | None = None

        self.node_base_dim: int | None = None
        self.edge_in_dim: int | None = None
        self.act_dim: int | None = None
        self.lr: float | None = None
        self.seed: int = 0

    def init(
        self,
        *,
        node_base_dim: int,
        edge_in_dim: int,
        act_dim: int,
        lr: float,
        seed: int = 0,
        force: bool = False,
        hidden: int = 128,
        layers: int = 3,
        time_dim: int = 32,
    ):
        """
        node_base_dim: dim of batch.x (e.g. 3 for [unary2 + status1])
        edge_in_dim:   dim of batch.edge_attr (e.g. 4)
        act_dim:       n_nodes (sphere dimension)
        """
        self.lr, self.seed = float(lr), int(seed)
        if force or (self.model is None):
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            self.node_base_dim = int(node_base_dim)
            self.edge_in_dim = int(edge_in_dim)
            self.act_dim = int(act_dim)

            self.model = RFMPolicyGNN(
                node_base_dim=self.node_base_dim,
                edge_in_dim=self.edge_in_dim,
                act_dim=self.act_dim,
                hidden=int(hidden),
                layers=int(layers),
                time_dim=int(time_dim),
            ).to(self.device)

            self.optim = optim.Adam(self.model.parameters(), lr=self.lr)

    def update(self, batch: Batch, z1: Tensor | np.ndarray, w: Tensor | np.ndarray) -> float:
        """
        One gradient update step for the actor.
        batch: Batch with B graphs
        z1: [B,D] endpoints on the sphere (torch or np)
        w:  [B] per-sample weights (torch or np)
        """
        assert self.model is not None and self.optim is not None

        batch = batch.to(self.device)

        z1_t = torch.as_tensor(z1, device=self.device, dtype=torch.float32)
        w_t = torch.as_tensor(w, device=self.device, dtype=torch.float32)
        if z1_t.dim() == 1:
            z1_t = z1_t.unsqueeze(0)
        if w_t.dim() == 0:
            w_t = w_t.unsqueeze(0)

        self.optim.zero_grad(set_to_none=True)
        loss = self.model.rfm_loss(batch, z1_t, w_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optim.step()
        return float(loss.detach().cpu())

    # -------- sampling / targets / noise --------
    @torch.no_grad()
    def sample(
        self,
        batch: Batch,
        K: int,
        steps: int = 30,
        *,
        kappa: float | None = None,
        J_noise: int = 1,
    ) -> np.ndarray:
        assert self.model is not None
        batch = batch.to(self.device)
        Z = self.model.sample(batch, K=K, steps=steps)  # [B,K,D]

        if (kappa is None) or (J_noise <= 0) or (float(kappa) <= 0):
            return Z.detach().cpu().numpy()

        B, K_, D = Z.shape
        mu = Z.reshape(B * K_, D)
        Z_noisy = _vmf_perturb_torch(mu, kappa=float(kappa), J=int(J_noise))  # [B*K,J,D]
        Z_noisy = Z_noisy.reshape(B, K_ * int(J_noise), D)
        return Z_noisy.detach().cpu().numpy()


    @torch.no_grad()
    def perturb(self, mu_np: np.ndarray, kappa: float, J: int = 1) -> np.ndarray:
        """
        Draw J vMF-like samples around each mean direction μ on S^{D-1}.
        Accepts numpy array, returns numpy array.
        """
        mu = torch.as_tensor(mu_np, device=self.device, dtype=torch.float32)
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)
        Z = _vmf_perturb_torch(mu, float(kappa), int(J))  # [B,J,D]
        return Z.detach().cpu().numpy()


rfm_service_gnn = _RFMServiceGNN()
