from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv


class GraphActorPhi(nn.Module):
    """
    Graph actor: (Batch of disease graphs) -> per-node scores theta

    Input:
      batch: torch_geometric.data.Batch
        - batch.x:         [num_nodes_total, node_in_dim]   (e.g., 3 = unary(2)+status(1))
        - batch.edge_index [2, num_edges_total]
        - batch.edge_attr  [num_edges_total, edge_in_dim]   (e.g., 4)

    Output:
      theta: [B, n]  (B graphs; each graph has n nodes)
        - per-node logit/score used by your existing co_layer (top-k / noisy top-k / candidate sampling)
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden: int = 256,
        layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_in_dim = int(node_in_dim)
        self.edge_in_dim = int(edge_in_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
        )

        self.node_proj = nn.Sequential(
            nn.Linear(self.node_in_dim, self.hidden),
            nn.ReLU(),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden, self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.hidden),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=self.hidden))
            self.norms.append(nn.LayerNorm(self.hidden))

        self.head = nn.Linear(self.hidden, 1)

    def forward(self, batch: Batch) -> Tensor:
        """
        Returns:
          theta: [B, n] (float32)
        """
        if not isinstance(batch, Batch):
            raise TypeError(f"GraphActorPhi expects a PyG Batch, got {type(batch)}")

        x = batch.x
        if x.dim() != 2 or x.shape[1] != self.node_in_dim:
            raise ValueError(f"batch.x must be [N, {self.node_in_dim}], got {tuple(x.shape)}")

        e = batch.edge_attr
        if e.dim() != 2 or e.shape[1] != self.edge_in_dim:
            raise ValueError(f"batch.edge_attr must be [E, {self.edge_in_dim}], got {tuple(e.shape)}")

        h = self.node_proj(x) # [N, hidden]
        e_h = self.edge_mlp(e) # [E, hidden]

        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, batch.edge_index, e_h)  # [N, hidden]
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            if self.dropout > 0:
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # residual

        node_score = self.head(h).squeeze(-1)  # [N]


        B = int(batch.num_graphs)
        n = int(batch.num_nodes // max(B, 1))
        if B * n != int(batch.num_nodes):
            theta_list = []
            for b in range(B):
                start = int(batch.ptr[b])
                end = int(batch.ptr[b + 1])
                theta_list.append(node_score[start:end])
            max_n = max(t.numel() for t in theta_list)
            out = node_score.new_zeros((B, max_n))
            for b, t in enumerate(theta_list):
                out[b, : t.numel()] = t
            return out
        else:
            return node_score.view(B, n)

    @torch.no_grad()
    def act_greedy(
        self,
        batch: Batch,
        k: int,
        mask: Optional[Tensor] = None,
        constraint_solver=None,
    ) -> Tensor:
        """
        Convenience wrapper so you can call actor.act_greedy(batch, k, ...)
        using your existing co_layer.
        Returns: [B, n] 0/1 action
        """
        from .co_layer import act_greedy as _act_greedy

        theta = self.forward(batch)  # [B, n]
        return _act_greedy(theta, k=k, mask=mask, constraint_solver=constraint_solver)

    @torch.no_grad()
    def act_with_noise(
        self,
        batch: Batch,
        k: int,
        sigma_f: float,
        mask: Optional[Tensor] = None,
        constraint_solver=None,
    ) -> Tensor:
        """
        Convenience wrapper for noisy action sampling.
        Returns: [B, n] 0/1 action
        """
        from .co_layer import act_with_noise as _act_with_noise

        theta = self.forward(batch)  # [B, n]
        return _act_with_noise(theta, k=k, sigma_f=sigma_f, mask=mask, constraint_solver=constraint_solver)

    @torch.no_grad()
    def sample_candidates(
        self,
        batch: Batch,
        k: int,
        m: int,
        sigma_b: float,
        mask: Optional[Tensor] = None,
        constraint_solver=None,
    ) -> Tensor:
        """
        Convenience wrapper returning candidates generated from theta:
          returns [m, B, n] 0/1 candidate actions
        """
        from .co_layer import sample_candidates_from_theta as _sample

        theta = self.forward(batch)  # [B, n]
        return _sample(theta, k=k, m=m, sigma_b=sigma_b, mask=mask, constraint_solver=constraint_solver)


__all__ = ["GraphActorPhi"]