# algos/dpmd_rf_disease_gnn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, NamedTuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv

from models.rfm.service_gnn import rfm_service_gnn


# -----------------------------
# Replay experience tuple
# -----------------------------
class Experience(NamedTuple):
    obs:         np.ndarray  # [B, n]  status
    action:      np.ndarray  # [B, n]  executed coefficient c_exec (stored in replay)
    reward:      np.ndarray  # [B]
    next_obs:    np.ndarray  # [B, n]  next status
    done:        np.ndarray  # [B]
    action_star: np.ndarray  # [B, n]  greedy center coefficient c*
    policy_id:   np.ndarray  # [B]


# -----------------------------
# Config
# -----------------------------
@dataclass
class DPMDGraphConfig:
    # RL
    gamma: float = 0.99
    lr: float = 4e-4
    tau: float = 0.005
    delay_update: int = 2
    reward_scale: float = 1.0

    # Actor sampling / candidate eval
    num_particles: int = 12

    # Mirror-descent temperature for weights
    w_clip: Optional[float] = 4.0
    lambda_start: float = 2.0
    lambda_end:   float = 0.8
    lambda_steps: int   = 10_000

    # Execution noise (on-sphere)
    kappa_exec: float = 28.0

    # Smoothed Bellman operator
    kappa_smooth: float = 28.0
    M_smooth: int = 16
    J_smooth: int = 1

    # running statistics for Q normalization
    q_running_beta: float = 0.05
    q_norm_clip: float = 3.0

    # temperature learning (log_alpha)
    alpha_lr: float = 3e-2
    delay_alpha_update: int = 180
    target_entropy: float = 0.0

    # diffusion/flow steps
    flow_steps: int = 36

    # ---- graph feature dims ----
    node_in_dim: int = 3        # [unary(2), status(1)]
    edge_in_dim: int = 4        # pairwise factor flattened

    # ---- critic GNN ----
    q_hidden: int = 128
    q_layers: int = 3


# -----------------------------
# Utils
# -----------------------------
def _to_tensor(x, device, dtype=torch.float32) -> Tensor:
    if isinstance(x, np.ndarray):
        if not x.flags.writeable:
            x = np.array(x, copy=True)
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        return torch.tensor(x, device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _repeat_batch(batch: Batch, K: int) -> Batch:
    """Repeat a Batch of B graphs into a Batch of (B*K) graphs."""
    if K <= 1:
        return batch
    data_list = batch.to_data_list()
    rep = [g for g in data_list for _ in range(int(K))]
    return Batch.from_data_list(rep).to(batch.x.device)


def _node_local_index(batch: Batch) -> Tensor:
    """Per-node local index within its graph: [num_nodes_total]."""
    idx = torch.arange(batch.num_nodes, device=batch.x.device)
    return idx - batch.ptr[batch.batch]


# -----------------------------
# Critic Q(batch_graph, c): GNN
# -----------------------------
class GraphQNet(nn.Module):
    """
    Enhanced Q(s,c) GNN with:
      - Deeper architecture (6 layers default)
      - Wider hidden (256 default)
      - Residual connections
      - Attention-based pooling
    """
    def __init__(self, node_base_dim: int, edge_in_dim: int, hidden: int = 256, layers: int = 6):
        super().__init__()
        self.node_base_dim = int(node_base_dim)
        self.edge_in_dim = int(edge_in_dim)
        self.hidden = int(hidden)
        self.num_layers = int(layers)

        # Edge MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        # Input projection: node_in → hidden
        node_in = self.node_base_dim + 1  # + c_scalar
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

        # Attention pooling
        self.pool_attn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        # Head
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: Batch, c: Tensor) -> Tensor:
        """
        batch: Batch with B graphs
        c:     [B, n_nodes] coefficient vector
        returns: [B] Q values
        """
        assert c.dim() == 2, "c must be [B, D]"
        B = int(batch.num_graphs)
        D = int(c.shape[1])

        # broadcast c to nodes via (batch_id, local_node_id)
        local_idx = _node_local_index(batch)                          # [num_nodes_total]
        c_scalar = c[batch.batch, local_idx].unsqueeze(-1)            # [num_nodes_total,1]

        x = torch.cat([batch.x, c_scalar], dim=-1)                    # [num_nodes_total, node_in]
        e = self.edge_mlp(batch.edge_attr)                            # [num_edges_total, hidden]

        # Project input to hidden dimension
        h = self.input_proj(x)                                        # [num_nodes_total, hidden]

        # Apply GNN layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, batch.edge_index, e)                      # [num_nodes_total, hidden]
            h_new = norm(h_new)                                       # Layer normalization
            h = h + F.silu(h_new)                                     # Residual connection

        # Attention-based pooling
        attn_scores = self.pool_attn(h)                               # [num_nodes_total, 1]
        attn_weights = torch.softmax(attn_scores, dim=0)              # Softmax over all nodes in batch

        # Weight by batch assignment
        pooled = torch.zeros(B, self.hidden, device=h.device, dtype=h.dtype)
        for b in range(B):
            mask = (batch.batch == b)
            if mask.any():
                h_b = h[mask]                                         # [num_nodes_in_graph_b, hidden]
                attn_b = attn_weights[mask]                           # [num_nodes_in_graph_b, 1]
                attn_b = attn_b / (attn_b.sum() + 1e-8)              # Renormalize within graph
                pooled[b] = (h_b * attn_b).sum(dim=0)                # [hidden]

        q = self.head(pooled).squeeze(-1)                             # [B]
        assert q.shape[0] == B
        return q


# -----------------------------
# Disease DPMD-RF with:
#   - actor: RFMPolicyGNN via rfm_service_gnn
#   - critics: twin GraphQNet on (Batch, c)
# -----------------------------
class DPMDGraphDisease:
    def __init__(
        self,
        n_nodes: int,
        node_covariates: np.ndarray,  # [n,2]
        edge_index: np.ndarray,       # [2, E_dir]
        edge_attr: np.ndarray,        # [E_dir, 4]
        *,
        device: Optional[torch.device] = None,
        cfg: DPMDGraphConfig = DPMDGraphConfig(),
    ):
        self.cfg = cfg
        self.n = int(n_nodes)
        self.act_dim = self.n

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # static graph tensors
        assert node_covariates.shape == (self.n, 2)
        self.node_cov = torch.tensor(node_covariates, dtype=torch.float32, device=self.device)  # [n,2]
        self.edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)        # [2,E]
        self.edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=self.device)       # [E,4]

        # critics + targets
        self.q1 = GraphQNet(cfg.node_in_dim, cfg.edge_in_dim, hidden=cfg.q_hidden, layers=cfg.q_layers).to(self.device)
        self.q2 = GraphQNet(cfg.node_in_dim, cfg.edge_in_dim, hidden=cfg.q_hidden, layers=cfg.q_layers).to(self.device)
        self.tq1 = GraphQNet(cfg.node_in_dim, cfg.edge_in_dim, hidden=cfg.q_hidden, layers=cfg.q_layers).to(self.device)
        self.tq2 = GraphQNet(cfg.node_in_dim, cfg.edge_in_dim, hidden=cfg.q_hidden, layers=cfg.q_layers).to(self.device)
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())

        self.q_optim = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=cfg.lr
        )

        # MD stats / policy version
        self.step = 0
        self.policy_version = 0
        self._q_mean = 0.0
        self._q_std = 1.0

        # temperature log_alpha
        self.log_alpha = nn.Parameter(torch.tensor(-0.5, device=self.device))
        self.alpha_optim = optim.Adam([self.log_alpha], lr=cfg.alpha_lr)


    # ------------------------------------------------------------------
    # Build Batch from status
    # ------------------------------------------------------------------
    def _data_from_status(self, status: np.ndarray) -> Data:
        status_feat = torch.tensor(status.astype(np.float32), device=self.device).view(self.n, 1)
        x = torch.cat([self.node_cov, status_feat], dim=1)  # [n,3]
        return Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr)

    def batch_from_status_batch(self, status_batch: np.ndarray) -> Batch:
        datas = [self._data_from_status(s) for s in status_batch]
        return Batch.from_data_list(datas).to(self.device)

    # ------------------------------------------------------------------
    # Actor sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _sample_candidates(self, batch: Batch, K: int) -> Tensor:
        C_np = rfm_service_gnn.sample(
            batch=batch,
            K=K,
            steps=self.cfg.flow_steps,
            kappa=None,
            J_noise=1,
        )
        return _to_tensor(C_np, self.device)  # [B,K,D]

    @torch.no_grad()
    def sample_candidates(self, obs_status: np.ndarray, K: int) -> np.ndarray:
        batch = self.batch_from_status_batch(obs_status.reshape(1, -1))
        C = self._sample_candidates(batch, K=K)[0]  # [K,D]
        return C.detach().cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def score_actions(self, obs_status: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        obs_status: [n]
        C: [K,D]
        returns q: [K]
        """
        batch = self.batch_from_status_batch(obs_status.reshape(1, -1))  # 1 graph
        K = int(C.shape[0])
        batchK = _repeat_batch(batch, K)                                  # K graphs
        c = _to_tensor(C, self.device)                                    # [K,D]
        q = torch.minimum(self.q1(batchK, c), self.q2(batchK, c))          # [K]
        return q.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # DPMD-RF core pieces
    # ------------------------------------------------------------------
    def _current_lambda(self) -> float:
        s = min(1.0, self.step / max(1, self.cfg.lambda_steps))
        return float((1.0 - s) * self.cfg.lambda_start + s * self.cfg.lambda_end)

    @torch.no_grad()
    def _weights_no_smooth(self, batch_old: Batch, c1: Tensor, lam: float) -> Tensor:
        q = torch.minimum(self.tq1(batch_old, c1), self.tq2(batch_old, c1))  # [B]

        q_mean_batch = q.mean()
        q_std_batch = q.std() + 1e-6

        self._q_mean = (1 - self.cfg.q_running_beta) * self._q_mean + self.cfg.q_running_beta * float(q_mean_batch)
        self._q_std  = (1 - self.cfg.q_running_beta) * self._q_std  + self.cfg.q_running_beta * float(q_std_batch)

        q_norm_ema = (q - self._q_mean) / (self._q_std + 1e-6)
        q_norm_batch = (q - q_mean_batch) / q_std_batch
        q_norm = 0.5 * q_norm_ema + 0.5 * q_norm_batch
        q_norm = torch.clamp(q_norm, -self.cfg.q_norm_clip, self.cfg.q_norm_clip)

        alpha = torch.exp(self.log_alpha).detach()
        lam = max(float(lam), 1e-6)
        logits = (alpha * q_norm) / lam
        w = torch.exp(logits)

        if self.cfg.w_clip is not None:
            w = torch.clamp(w, max=float(self.cfg.w_clip))
        return w  # [B]

    @torch.no_grad()
    def _smoothed_value(self, batch_next: Batch) -> Tensor:
        """
        Smoothed V(s') computed via sampling actor + vMF noise and target critics.
        returns: [B]
        """
        B = int(batch_next.num_graphs)
        M = max(1, int(self.cfg.M_smooth))
        J = max(1, int(self.cfg.J_smooth))
        kappa = float(self.cfg.kappa_smooth)

        # Cprime: [B,M,D]
        Cprime = self._sample_candidates(batch_next, M)
        cm = Cprime.reshape(B * M, self.act_dim)  # [B*M,D]

        # vMF noise around each candidate
        cm_tilde = rfm_service_gnn.perturb(cm.detach().cpu().numpy(), kappa=kappa, J=J)  # [B*M,J,D]
        Chat = _to_tensor(cm_tilde, self.device).reshape(B, M, J, self.act_dim)

        # Evaluate target critics on (B*M*J) repeated graphs
        batch_rep = _repeat_batch(batch_next, M * J)                 # [B*M*J graphs]
        flat_c = Chat.reshape(B * M * J, self.act_dim)               # [B*M*J,D]

        q1 = self.tq1(batch_rep, flat_c).view(B, M, J)
        q2 = self.tq2(batch_rep, flat_c).view(B, M, J)
        qmin = torch.minimum(q1, q2)

        qflat = qmin.view(B, -1)  # [B, M*J]
        ktrim = int(0.2 * qflat.shape[1])
        vals, _ = torch.sort(qflat, dim=1, descending=True)
        Vb = vals[:, :-ktrim].mean(dim=1) if ktrim > 0 else vals.mean(dim=1)
        return Vb  # [B]

    # -----------------------------------------------------
    # Pretrain critics (myopic): y=r
    # -----------------------------------------------------
    def pretrain_critics_step(self, batch: Experience, huber_delta: float = 5.0) -> float:
        s = _to_tensor(batch.obs, self.device)               # [B,n]
        c_clean = _to_tensor(batch.action_star, self.device) # [B,n]
        y = _to_tensor(batch.reward, self.device).view(-1)   # [B]

        batch_graph = self.batch_from_status_batch(s.detach().cpu().numpy())

        def huber(a, b, delta=huber_delta):
            x = a - b
            ax = torch.abs(x)
            return torch.where(ax < delta, 0.5*x*x, delta*(ax - 0.5*delta)).mean()

        self.q_optim.zero_grad(set_to_none=True)
        q1_loss = huber(self.q1(batch_graph, c_clean), y)
        q2_loss = huber(self.q2(batch_graph, c_clean), y)
        (q1_loss + q2_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            10.0
        )
        self.q_optim.step()

        with torch.no_grad():
            self.tq1.load_state_dict(self.q1.state_dict())
            self.tq2.load_state_dict(self.q2.state_dict())

        return float((q1_loss + q2_loss).detach().cpu())

    # -----------------------------------------------------
    # Main update
    # -----------------------------------------------------
    def update(self, batch: Experience) -> Dict[str, float]:
        s      = _to_tensor(batch.obs,      self.device)              # [B,n]
        c_star = _to_tensor(batch.action_star, self.device)           # [B,n]
        rew    = _to_tensor(batch.reward,  self.device).view(-1) * float(self.cfg.reward_scale)
        s_next = _to_tensor(batch.next_obs, self.device)              # [B,n]
        done   = _to_tensor(batch.done,    self.device).view(-1)

        batch_s = self.batch_from_status_batch(s.detach().cpu().numpy())
        batch_sn = self.batch_from_status_batch(s_next.detach().cpu().numpy())

        # 1) Smoothed target
        with torch.no_grad():
            Vb = self._smoothed_value(batch_sn)  # [B]
            y = rew + (1.0 - done) * float(self.cfg.gamma) * Vb

        # 2) Critic update
        def huber(a, b, delta=5.0):
            x = a - b
            ax = torch.abs(x)
            return torch.where(ax < delta, 0.5*x*x, delta*(ax - 0.5*delta)).mean()

        self.q_optim.zero_grad(set_to_none=True)
        q1_loss = huber(self.q1(batch_s, c_star), y)
        q2_loss = huber(self.q2(batch_s, c_star), y)
        (q1_loss + q2_loss).backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
        self.q_optim.step()

        # 3) Actor update: sample c1 from actor on s, weight with target critics
        with torch.no_grad():
            C1 = self._sample_candidates(batch_s, K=1)  # [B,1,D]
            c1 = C1[:, 0, :]                                             # [B,D]
            w = self._weights_no_smooth(batch_s, c1, lam=float(self._current_lambda()))  # [B]

        policy_loss = rfm_service_gnn.update(batch_s, c1, w)

        # 4.5) delayed temperature update (keep your old heuristic)
        if (self.step % max(1, int(self.cfg.delay_alpha_update))) == 0:
            self.alpha_optim.zero_grad(set_to_none=True)
            approx_entropy = 0.5 * self.act_dim * torch.log(
                torch.tensor(2.0 * np.pi * np.e, device=self.device, dtype=torch.float32)
                * (0.1 * torch.exp(self.log_alpha)).pow(2)
            )
            alpha_loss = -self.log_alpha * (-approx_entropy.detach() + float(self.cfg.target_entropy))
            alpha_loss.backward()
            self.alpha_optim.step()

        # 5) soft-update target critics
        if (self.step % int(self.cfg.delay_update)) == 0:
            with torch.no_grad():
                for p_t, p in zip(self.tq1.parameters(), self.q1.parameters()):
                    p_t.mul_(1.0 - self.cfg.tau).add_(p, alpha=self.cfg.tau)
                for p_t, p in zip(self.tq2.parameters(), self.q2.parameters()):
                    p_t.mul_(1.0 - self.cfg.tau).add_(p, alpha=self.cfg.tau)

        self.step += 1

        return {
            "q1_loss": float(q1_loss.detach().cpu()),
            "q2_loss": float(q2_loss.detach().cpu()),
            "policy_loss": float(policy_loss),
        }


__all__ = ["DPMDGraphDisease", "DPMDGraphConfig", "Experience"]
