from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, global_mean_pool


def _node_local_index(batch: Batch) -> Tensor:
    """
    Per-node local index within its graph: [num_nodes_total].
    Assumes graphs are packed in Batch with .ptr and .batch fields.
    """
    idx = torch.arange(batch.num_nodes, device=batch.x.device)
    return idx - batch.ptr[batch.batch]


class GraphCriticNet(nn.Module):
    """
    Q(s_graph, a_binary) network.

    Inputs:
      - batch: PyG Batch (B graphs)
          batch.x:         [N_total, node_in_dim]   (e.g., 3 = unary(2)+status(1))
          batch.edge_index [2, E_total]
          batch.edge_attr  [E_total, edge_in_dim]   (e.g., 4)
      - a: [B, n] binary (0/1) action vector

    Output:
      - q: [B] scalar Q values
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden: int = 256,
        layers: int = 3,
        dropout: float = 0.0,
        pool: str = "mean",  # "mean" or "attn"
    ):
        super().__init__()
        self.node_in_dim = int(node_in_dim)
        self.edge_in_dim = int(edge_in_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.pool = str(pool)

        self.node_proj = nn.Sequential(
            nn.Linear(self.node_in_dim + 1, self.hidden),
            nn.ReLU(),
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
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

        if self.pool == "attn":
            self.pool_attn = nn.Sequential(
                nn.Linear(self.hidden, self.hidden),
                nn.Tanh(),
                nn.Linear(self.hidden, 1),
            )
        else:
            self.pool_attn = None

        self.head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

    def forward(self, batch: Batch, a: Tensor) -> Tensor:
        if not isinstance(batch, Batch):
            raise TypeError(f"GraphCriticNet expects PyG Batch, got {type(batch)}")

        x0 = batch.x
        e0 = batch.edge_attr

        if x0.dim() != 2 or x0.shape[1] != self.node_in_dim:
            raise ValueError(f"batch.x must be [N, {self.node_in_dim}], got {tuple(x0.shape)}")
        if e0.dim() != 2 or e0.shape[1] != self.edge_in_dim:
            raise ValueError(f"batch.edge_attr must be [E, {self.edge_in_dim}], got {tuple(e0.shape)}")

        if a.dim() != 2:
            raise ValueError(f"a must be [B,n], got {tuple(a.shape)}")

        B = int(batch.num_graphs)
        if int(a.shape[0]) != B:
            raise ValueError(f"a batch dim mismatch: a.shape[0]={int(a.shape[0])} vs num_graphs={B}")

        local_idx = _node_local_index(batch)  # [N_total]
        a_scalar = a[batch.batch, local_idx].to(dtype=x0.dtype).unsqueeze(-1)  # [N_total,1]

        x = torch.cat([x0, a_scalar], dim=-1)  # [N_total, node_in_dim+1]
        h = self.node_proj(x)                  # [N_total, hidden]
        e = self.edge_mlp(e0)                  # [E_total, hidden]

        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, batch.edge_index, e)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            if self.dropout > 0:
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # residual

        if self.pool == "mean":
            g = global_mean_pool(h, batch.batch)  # [B, hidden]
        elif self.pool == "attn":
            attn_scores = self.pool_attn(h)  # [N_total, 1]
            g = h.new_zeros((B, self.hidden))
            for b in range(B):
                mask = batch.batch == b
                if mask.any():
                    hb = h[mask]  # [Nb, hidden]
                    ab = attn_scores[mask]  # [Nb, 1]
                    w = torch.softmax(ab.squeeze(-1), dim=0).unsqueeze(-1)  # [Nb,1]
                    g[b] = (hb * w).sum(dim=0)
        else:
            raise ValueError(f"unknown pool={self.pool}")

        q = self.head(g).squeeze(-1)  # [B]
        return q


class DoubleGraphCritic(nn.Module):
    """
    Twin critics for TD3-style min(Q1,Q2).
    """
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden: int = 256,
        layers: int = 3,
        dropout: float = 0.0,
        pool: str = "mean",
    ):
        super().__init__()
        self.q1 = GraphCriticNet(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden=hidden,
            layers=layers,
            dropout=dropout,
            pool=pool,
        )
        self.q2 = GraphCriticNet(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden=hidden,
            layers=layers,
            dropout=dropout,
            pool=pool,
        )

    def forward(self, batch: Batch, a: Tensor) -> Tuple[Tensor, Tensor]:
        return self.q1(batch, a), self.q2(batch, a)


@torch.no_grad()
def soft_target_update(target: nn.Module, online: nn.Module, tau: float = 0.995):
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(tau).add_(p_o.data, alpha=1 - tau)


@torch.no_grad()
def hard_target_update(target: nn.Module, online: nn.Module):
    target.load_state_dict(online.state_dict())


__all__ = [
    "GraphCriticNet",
    "DoubleGraphCritic",
    "soft_target_update",
    "hard_target_update",
]