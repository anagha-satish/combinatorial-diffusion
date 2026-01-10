# models/rfm/policy_gnn.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv


# ---------- S^D geometry helpers ----------
def normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Project any vector(s) to the unit sphere."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def sphere_logmap(p: Tensor, x: Tensor, eps: float = 1e-8) -> Tensor:  # Log_p(x)
    """Riemannian log map on the sphere: a tangent vector at p."""
    dot = (p * x).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    v = x - dot * p
    theta = torch.arccos(dot)
    nv = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    return v * (theta / nv)


def sphere_expmap(p: Tensor, v: Tensor, eps: float = 1e-8) -> Tensor:  # Exp_p(v)
    """Exponential map: move from p along tangent v by distance ||v||."""
    nv = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    return normalize(p * torch.cos(nv) + v * (torch.sin(nv) / nv))


# ---------- Time embedding ----------
class TimeEmbed(nn.Module):
    """
    Sinusoidal Fourier features for t in [0,1] with a tiny MLP head.
    Output dim == embed_dim.
    """

    def __init__(self, embed_dim: int = 32, L: int = 16):
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
        if t.dim() == 1:
            t = t.view(-1, 1)
        x = t.to(dtype=self.freqs.dtype) * self.freqs.view(1, -1)  # [B,L]
        pe = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)        # [B,2L]
        return self.proj(pe)                                       # [B,embed_dim]


# ---------- Helpers for graph batching ----------
def _repeat_batch(batch: Batch, K: int) -> Batch:
    """Repeat a Batch of B graphs into a Batch of (B*K) graphs."""
    if K <= 1:
        return batch
    data_list = batch.to_data_list()
    rep = [g for g in data_list for _ in range(int(K))]
    return Batch.from_data_list(rep).to(batch.x.device)


def _node_local_index(batch: Batch) -> Tensor:
    """
    Return per-node local index within its graph: [num_nodes_total].
    Requires batch.ptr exists (it does for PyG Batch).
    """
    # batch.ptr: [B+1], offsets into the concatenated node list
    idx = torch.arange(batch.num_nodes, device=batch.x.device)
    return idx - batch.ptr[batch.batch]


# ---------- RFM GNN policy ----------
class RFMPolicyGNN(nn.Module):
    """
    Enhanced GNN velocity field with:
      - Deeper architecture (6 layers default)
      - Wider hidden (256 default)
      - Residual connections
      - Layer normalization

    u_theta(batch_graph, z_t, t) -> tangent vector on S^{D-1}, where D = n_nodes.
    """

    def __init__(
        self,
        node_base_dim: int,
        edge_in_dim: int,
        act_dim: int,
        hidden: int = 256,
        layers: int = 6,
        time_dim: int = 32,
    ):
        super().__init__()
        self.act_dim = int(act_dim)
        self.node_base_dim = int(node_base_dim)
        self.edge_in_dim = int(edge_in_dim)
        self.hidden = int(hidden)
        self.num_layers = int(layers)

        self.tok = TimeEmbed(embed_dim=time_dim, L=16)

        # Edge MLP to match GINE edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        # Node input: base + c_scalar + tfeat
        node_in = node_base_dim + 1 + time_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_in, hidden),
            nn.SiLU(),
        )

        # GNN layers with residual connections
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=hidden))
            self.norms.append(nn.LayerNorm(hidden))

        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),  # per-node scalar velocity
        )

    def forward(self, batch: Batch, z_t: Tensor, t: Tensor) -> Tensor:
        """
        batch: Batch with B graphs
               batch.x: [B*n_nodes, node_base_dim]
        z_t:  [B, D] or [B*K, D] (D == n_nodes)
        t: scalar tensor or [B] / [B*K]
        returns: [B, D] tangent vector at z_t
        """
        assert z_t.dim() == 2, "z_t must be [B, D]"
        B, D = z_t.shape
        assert D == self.act_dim, f"z_t dim {D} != act_dim {self.act_dim}"

        device = batch.x.device

        # t -> [B,1]
        if t.dim() == 0:
            t = t.expand(B)
        if t.dim() == 1:
            t = t.view(-1, 1)
        tfeat = self.tok(t)  # [B, time_dim]

        # broadcast tfeat to nodes
        tfeat_nodes = tfeat[batch.batch]  # [num_nodes_total, time_dim]

        # get local node index to pick c_t per node
        local_idx = _node_local_index(batch)  # [num_nodes_total]
        c_scalar = z_t[batch.batch, local_idx].unsqueeze(-1)  # [num_nodes_total, 1]

        x = torch.cat([batch.x, c_scalar, tfeat_nodes], dim=-1)  # [num_nodes_total, node_in]

        # edge features
        e = self.edge_mlp(batch.edge_attr)  # [num_edges_total, hidden]

        # Project input to hidden dimension
        h = self.input_proj(x)  # [num_nodes_total, hidden]

        # Apply GNN layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, batch.edge_index, e)  # [num_nodes_total, hidden]
            h_new = norm(h_new)  # Layer normalization
            h = h + F.silu(h_new)  # Residual connection

        v_node = self.out(h).squeeze(-1)  # [num_nodes_total]

        # reshape node scalars -> [B, D] (assumes fixed n_nodes per graph)
        # This is valid for your setting ("graph topology the same, only status and c changes").
        v = v_node.view(B, D)

        # project to tangent at z_t
        return v - (v * z_t).sum(dim=-1, keepdim=True) * z_t

    @torch.no_grad()
    def sample(self, batch: Batch, K: int, steps: int = 30) -> Tensor:
        """
        Return [B, K, D] samples on the sphere using a 2nd-order (Heun) integrator.
        """
        assert K >= 1
        B = int(batch.num_graphs)
        D = self.act_dim

        # Repeat graphs K times -> Batch with B*K graphs
        batch_rep = _repeat_batch(batch, K)
        BK = int(batch_rep.num_graphs)

        z = normalize(torch.randn(BK, D, device=batch.x.device, dtype=batch.x.dtype))

        dt = 1.0 / float(steps)
        t_grid = torch.linspace(0, 1, steps + 1, device=batch.x.device, dtype=batch.x.dtype)

        for i in range(steps):
            t_i, t_ip1 = t_grid[i], t_grid[i + 1]
            u_i = self.forward(batch_rep, z, t_i)          # [BK,D]
            z_tilde = sphere_expmap(z, dt * u_i)           # [BK,D]
            u_ip1 = self.forward(batch_rep, z_tilde, t_ip1)
            u_heun = 0.5 * (u_i + u_ip1)
            z = sphere_expmap(z, dt * u_heun)

        return z.view(B, K, D)

    # ----------- stable core loss -----------
    def _rfm_loss_core(
        self,
        batch: Batch,
        z1: Tensor,
        w: Tensor,
        *,
        t: Tensor | None = None,
        t_eps: float = 0.1,
    ) -> Tensor:
        """
        Weighted Riemannian flow-matching loss on S^{D-1} for a batch of endpoints z1.
        """
        if z1.dim() == 1:
            z1 = z1.unsqueeze(0)
        if w.dim() == 0:
            w = w.unsqueeze(0)

        B, D = z1.shape
        assert D == self.act_dim
        assert int(batch.num_graphs) == B, "batch.num_graphs must match z1 batch size"

        # sample z0 ~ Uniform(S^{D-1})
        z0 = normalize(torch.randn_like(z1))

        # sample t ~ U[t_eps, 1 - t_eps] with antithetic pairs
        if t is None:
            if B >= 2:
                half = (B + 1) // 2
                t_half = torch.rand(half, device=z1.device, dtype=z1.dtype)
                t_full = torch.cat([t_half, 1.0 - t_half[: B - half]], dim=0)
                t = t_full
            else:
                t = torch.rand(B, device=z1.device, dtype=z1.dtype)
        else:
            if t.dim() == 0:
                t = t.unsqueeze(0)
        t = t.clamp(t_eps, 1.0 - t_eps)

        # geodesic interpolation
        xi_01 = sphere_logmap(z0, z1)                  # Log_{z0}(z1)
        z_t = sphere_expmap(z0, t.view(-1, 1) * xi_01)

        # target velocity: Log_{z_t}(z1) / (1-t)
        log_ct_z1 = sphere_logmap(z_t, z1)
        proj = log_ct_z1 - (log_ct_z1 * z_t).sum(dim=-1, keepdim=True) * z_t
        inv_1mt = 1.0 / (1.0 - t).view(-1, 1)
        u_star = proj * inv_1mt

        # model velocity via GNN
        u = self.forward(batch, z_t, t)

        err = (u - u_star).pow(2).sum(dim=-1)
        return (w * err).mean()

    def rfm_loss(self, batch: Batch, z1: Tensor, w: Tensor) -> Tensor:
        return self._rfm_loss_core(batch, z1, w)
