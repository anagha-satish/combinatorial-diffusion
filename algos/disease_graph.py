from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import Data, Batch


def _as_status_1d(x) -> np.ndarray:
    """Convert env obs/status to float32 1D array."""
    if isinstance(x, tuple):
        x = x[0]
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x


def build_static_graph_from_env(env) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build static graph tensors from disease env.

    Returns:
        node_cov:     [n, 2] float32
        edge_index:   [2, E_dir] int64   (directed: includes both directions)
        edge_attr:    [E_dir, 4] float32 (pairwise factor flattened)
    """
    if not hasattr(env, "base") or not hasattr(env.base, "get_status_and_factors"):
        raise AttributeError(
            "Expected disease env to have env.base.get_status_and_factors(). "
            "If your env differs, adapt this function accordingly."
        )

    status, unary_factors, pairwise_factors = env.base.get_status_and_factors()
    n = len(status)

    node_cov = np.array(
        [np.asarray(f).flatten() for f in unary_factors.values()],
        dtype=np.float32,
    )
    if node_cov.shape != (n, 2):
        raise ValueError(f"expected node_cov [n,2], got {node_cov.shape}")

    edge_list = [
        [min([int(Xidx[1:]) for Xidx in uv]), max([int(Xidx[1:]) for Xidx in uv])]
        for uv in pairwise_factors.keys()
    ]
    edge_index = np.asarray(edge_list, dtype=np.int64).T

    edge_attr = np.asarray(
        [np.asarray(tbl).flatten() for tbl in pairwise_factors.values()],
        dtype=np.float32,
    )
    if edge_attr.ndim != 2 or edge_attr.shape[1] != 4:
        raise ValueError(f"expected edge_attr second dim 4, got {edge_attr.shape}")

    edge_index_dir = np.concatenate([edge_index, edge_index[[1, 0], :]], axis=1)  # [2, E_dir]
    edge_attr_dir = np.concatenate([edge_attr, edge_attr], axis=0)                # [E_dir, 4]

    return node_cov, edge_index_dir, edge_attr_dir


@dataclass
class DiseaseGraphBuilder:
    """
    Build PyG Data/Batch for disease environment:
      - static: node_cov [n,2], edge_index [2,E], edge_attr [E,4]
      - dynamic: status [n] -> node feature x = [node_cov, status]
    """
    node_cov: Tensor           # [n,2]
    edge_index: Tensor         # [2,E]
    edge_attr: Tensor          # [E,4]
    device: torch.device

    @property
    def n(self) -> int:
        return int(self.node_cov.shape[0])

    @classmethod
    def from_env(
        cls,
        env,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "DiseaseGraphBuilder":
        node_cov, edge_index, edge_attr = build_static_graph_from_env(env)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        node_cov_t = torch.tensor(node_cov, dtype=torch.float32, device=device)
        edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32, device=device)

        return cls(
            node_cov=node_cov_t,
            edge_index=edge_index_t,
            edge_attr=edge_attr_t,
            device=device,
        )

    def data_from_status(self, status: Union[np.ndarray, Tensor]) -> Data:
        """
        status: [n] float array/tensor
        returns: PyG Data with x [n,3], edge_index, edge_attr
        """
        if isinstance(status, torch.Tensor):
            s = status.to(device=self.device, dtype=torch.float32).reshape(-1)
            s = s[: self.n]
            status_feat = s.view(self.n, 1)
        else:
            s = _as_status_1d(status)
            if s.shape[0] != self.n:
                s = s[: self.n]
                if s.shape[0] != self.n:
                    raise ValueError(f"status dim mismatch: got {s.shape[0]} expected {self.n}")
            status_feat = torch.tensor(s, dtype=torch.float32, device=self.device).view(self.n, 1)

        x = torch.cat([self.node_cov, status_feat], dim=1)  # [n, 3]
        return Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr)

    def batch_from_status_batch(
        self,
        status_batch: Union[np.ndarray, Tensor, List[np.ndarray], List[Tensor]],
    ) -> Batch:
        """
        status_batch:
          - np.ndarray [B,n] or torch.Tensor [B,n]
          - or list of length B, each [n]
        returns: PyG Batch with B graphs
        """
        datas: List[Data] = []

        if isinstance(status_batch, torch.Tensor):
            sb = status_batch.to(device=self.device, dtype=torch.float32)
            if sb.ndim == 1:
                sb = sb.view(1, -1)
            for i in range(sb.shape[0]):
                datas.append(self.data_from_status(sb[i]))
        elif isinstance(status_batch, np.ndarray):
            sb = np.asarray(status_batch, dtype=np.float32)
            if sb.ndim == 1:
                sb = sb.reshape(1, -1)
            for i in range(sb.shape[0]):
                datas.append(self.data_from_status(sb[i]))
        else:
            # list/iterable
            for s in status_batch:
                datas.append(self.data_from_status(s))

        return Batch.from_data_list(datas).to(self.device)

    def batch_from_env_obs(self, obs_or_status_list: Union[np.ndarray, List]) -> Batch:
        """
        Convenience: if you have raw env obs (maybe tuples), this normalizes to status.
        """
        if isinstance(obs_or_status_list, np.ndarray):
            return self.batch_from_status_batch(obs_or_status_list)
        statuses = [_as_status_1d(x) for x in obs_or_status_list]
        return self.batch_from_status_batch(statuses)


__all__ = [
    "DiseaseGraphBuilder",
    "build_static_graph_from_env",
]