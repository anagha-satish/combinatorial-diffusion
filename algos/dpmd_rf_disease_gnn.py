# algos/dpmd_rf_disease_gnn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, NamedTuple, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.data import Data, Batch

from models.rfm.service import rfm_service


# -----------------------------
# Replay experience tuple
# -----------------------------
class Experience(NamedTuple):
    obs:         np.ndarray  # [B, F]   (GNN pooled embedding z(s))
    action:      np.ndarray  # [B, D]   executed coefficient c_exec (stored in replay)
    reward:      np.ndarray  # [B]
    next_obs:    np.ndarray  # [B, F]   next embedding z(s')
    done:        np.ndarray  # [B]
    action_star: np.ndarray  # [B, D]   greedy center coefficient c*
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

    # ---- GNN encoder ----
    node_in_dim: int = 3        # [unary(2), status(1)]
    edge_in_dim: int = 4        # pairwise factor flattened
    gnn_hidden: int = 64
    gnn_layers: int = 2
    emb_dim: int = 64           # pooled z(s) dim; keep = gnn_hidden for simplicity


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

def _fit_width(x: Tensor, target_F: int) -> Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() >= 3:
        x = x.view(x.shape[0], -1)
    F = x.shape[-1]
    if F == target_F:
        return x
    if F > target_F:
        return x[:, :target_F].contiguous()
    pad = torch.zeros(x.shape[0], target_F - F, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


# -----------------------------
# GNN state encoder z(s)
# -----------------------------
class GraphStateEncoder(nn.Module):
    """
    Encodes (graph structure + unary/pairwise params + current status) into a pooled embedding z(s).
    """
    def __init__(self, node_in_dim: int, edge_in_dim: int, hidden: int, layers: int = 2):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.convs = nn.ModuleList()
        for i in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden if i > 0 else node_in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=hidden))
        self.out = nn.Identity()

    def forward(self, batch: Batch) -> Tensor:
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        e = self.edge_mlp(edge_attr)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, e))
        z = global_mean_pool(x, batch_idx)  # [B, hidden]
        return self.out(z)


# -----------------------------
# Critic Q(z(s), c): MLP on concat
# -----------------------------
class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


# -----------------------------
# Disease DPMD-RF with:
#   - GNN encoder producing z(s)
#   - diffusion actor via rfm_service on z(s)
#   - twin MLP critics on (z(s), c)
# -----------------------------
class DPMDGraphDisease:
    def __init__(
        self,
        n_nodes: int,
        node_covariates: np.ndarray,    # [n,2] unary factors flattened
        edge_index: np.ndarray,         # [2, E_dir]
        edge_attr: np.ndarray,          # [E_dir, 4]
        *,
        device: Optional[torch.device] = None,
        cfg: DPMDGraphConfig = DPMDGraphConfig(),
    ):
        self.cfg = cfg
        self.n = int(n_nodes)
        self.act_dim = self.n
        self.obs_dim = int(cfg.emb_dim)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- static graph tensors on device ---
        assert node_covariates.shape == (self.n, 2)
        self.node_cov = torch.tensor(node_covariates, dtype=torch.float32, device=self.device)  # [n,2]
        self.edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)       # [2,E]
        self.edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=self.device)      # [E,4]

        # --- encoder ---
        self.encoder = GraphStateEncoder(
            node_in_dim=cfg.node_in_dim,
            edge_in_dim=cfg.edge_in_dim,
            hidden=cfg.gnn_hidden,
            layers=cfg.gnn_layers,
        ).to(self.device)

        # If emb_dim != gnn_hidden, add a projection
        self.proj = nn.Linear(cfg.gnn_hidden, cfg.emb_dim).to(self.device) if cfg.gnn_hidden != cfg.emb_dim else nn.Identity()

        # --- critics + targets ---
        self.q1 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.q2 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.tq1 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.tq2 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())

        # critic optimizer ALSO trains encoder
        self.q_optim = optim.Adam(
            list(self.encoder.parameters()) + list(self.proj.parameters()) +
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

        # keep target actor aligned initially
        if hasattr(rfm_service, "sync_target_from_current"):
            rfm_service.sync_target_from_current()

    # ------------------------------------------------------------------
    # Graph -> embedding
    # ------------------------------------------------------------------
    def _data_from_status(self, status: np.ndarray) -> Data:
        status_feat = torch.tensor(status.astype(np.float32), device=self.device).view(self.n, 1)
        x = torch.cat([self.node_cov, status_feat], dim=1)  # [n,3]
        return Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr)

    def encode_status_batch(self, status_batch: np.ndarray) -> Tensor:
        """
        status_batch: [B,n] float32
        returns: z [B, emb_dim]
        """
        datas = [self._data_from_status(s) for s in status_batch]
        batch = Batch.from_data_list(datas).to(self.device)
        z = self.encoder(batch)
        z = self.proj(z)
        return z

    def encode_status(self, status: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = self.encode_status_batch(status.reshape(1, -1)).detach().cpu().numpy()[0]
        return z.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Diffusion policy on z(s)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _sample_candidates(self, z: Tensor, K: int, *, use_target: bool = False) -> Tensor:
        z = _fit_width(z, self.obs_dim)
        if use_target and hasattr(rfm_service, "sample_target"):
            C_np = rfm_service.sample_target(
                obs_np=z.detach().cpu().numpy(),
                K=K,
                steps=self.cfg.flow_steps,
                kappa=None,
                J_noise=1
            )
        else:
            C_np = rfm_service.sample(
                obs_np=z.detach().cpu().numpy(),
                K=K,
                steps=self.cfg.flow_steps,
                kappa=None,
                J_noise=1
            )
        return _to_tensor(C_np, self.device)  # [B,K,D]

    @torch.no_grad()
    def sample_candidates(self, obs_status: np.ndarray, K: int, *, use_target: bool = False) -> np.ndarray:
        z = torch.tensor(self.encode_status(obs_status).reshape(1, -1), device=self.device, dtype=torch.float32)
        C = self._sample_candidates(z, K=K, use_target=use_target)[0]  # [K,D]
        return C.detach().cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def score_actions(self, obs_status: np.ndarray, C: np.ndarray) -> np.ndarray:
        z = torch.tensor(self.encode_status(obs_status).reshape(1, -1), device=self.device, dtype=torch.float32)
        c = torch.tensor(C, device=self.device, dtype=torch.float32)
        z_rep = z.repeat(c.shape[0], 1)
        q = torch.minimum(self.q1(z_rep, c), self.q2(z_rep, c))
        return q.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # DPMD-RF core pieces
    # ------------------------------------------------------------------
    def _current_lambda(self) -> float:
        s = min(1.0, self.step / max(1, self.cfg.lambda_steps))
        return float((1.0 - s) * self.cfg.lambda_start + s * self.cfg.lambda_end)

    @torch.no_grad()
    def _weights_no_smooth(self, z_old: Tensor, c1: Tensor, lam: float) -> Tensor:
        tq1 = self.tq1(z_old, c1)
        tq2 = self.tq2(z_old, c1)
        q = torch.minimum(tq1, tq2)

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
        return w

    @torch.no_grad()
    def _smoothed_value(self, z_next: Tensor) -> Tensor:
        z_next = _fit_width(z_next, self.obs_dim)
        B = z_next.shape[0]
        M = max(1, int(self.cfg.M_smooth))
        J = max(1, int(self.cfg.J_smooth))
        kappa = float(self.cfg.kappa_smooth)

        Cprime = self._sample_candidates(z_next, M, use_target=True)  # [B,M,D]
        cm = Cprime.reshape(B * M, self.act_dim)
        cm_tilde = rfm_service.perturb(cm.detach().cpu().numpy(), kappa=kappa, J=J)  # [B*M,J,D]
        Chat = _to_tensor(cm_tilde, self.device).reshape(B, M, J, self.act_dim)

        rep_z = z_next.view(B, 1, 1, self.obs_dim).expand(B, M, J, self.obs_dim).reshape(B*M*J, self.obs_dim)
        flat_c = Chat.reshape(B*M*J, self.act_dim)

        q1 = self.tq1(rep_z, flat_c).view(B, M, J)
        q2 = self.tq2(rep_z, flat_c).view(B, M, J)
        qmin = torch.minimum(q1, q2)

        qflat = qmin.view(B, -1)
        ktrim = int(0.2 * qflat.shape[1])
        vals, _ = torch.sort(qflat, dim=1, descending=True)
        Vb = vals[:, :-ktrim].mean(dim=1) if ktrim > 0 else vals.mean(dim=1)
        return Vb

    # -----------------------------------------------------
    # Pretrain critics (myopic): y=r
    # -----------------------------------------------------
    def pretrain_critics_step(self, batch: Experience, huber_delta: float = 5.0) -> float:
        z = _fit_width(_to_tensor(batch.obs, self.device), self.obs_dim)
        c_clean = _to_tensor(batch.action_star, self.device)
        y = _to_tensor(batch.reward, self.device).view(-1)

        def huber(a, b, delta=huber_delta):
            x = a - b
            ax = torch.abs(x)
            return torch.where(ax < delta, 0.5*x*x, delta*(ax - 0.5*delta)).mean()

        self.q_optim.zero_grad(set_to_none=True)
        q1_loss = huber(self.q1(z, c_clean), y)
        q2_loss = huber(self.q2(z, c_clean), y)
        (q1_loss + q2_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.proj.parameters()) +
            list(self.q1.parameters()) + list(self.q2.parameters()),
            10.0
        )
        self.q_optim.step()

        # hard sync targets during pretrain
        with torch.no_grad():
            self.tq1.load_state_dict(self.q1.state_dict())
            self.tq2.load_state_dict(self.q2.state_dict())

        return float((q1_loss + q2_loss).detach().cpu())

    # -----------------------------------------------------
    # Main update
    # -----------------------------------------------------
    def update(self, batch: Experience) -> Dict[str, float]:
        z      = _fit_width(_to_tensor(batch.obs,      self.device), self.obs_dim)
        c_star = _to_tensor(batch.action_star, self.device)
        rew    = _to_tensor(batch.reward,  self.device).view(-1) * float(self.cfg.reward_scale)
        z_next = _fit_width(_to_tensor(batch.next_obs, self.device), self.obs_dim)
        done   = _to_tensor(batch.done,    self.device).view(-1)

        # 1) Smoothed target
        with torch.no_grad():
            Vb = self._smoothed_value(z_next)
            y = rew + (1.0 - done) * float(self.cfg.gamma) * Vb

        # 2) Critic update (encoder is trained here)
        def huber(a, b, delta=5.0):
            x = a - b
            ax = torch.abs(x)
            return torch.where(ax < delta, 0.5*x*x, delta*(ax - 0.5*delta)).mean()

        self.q_optim.zero_grad(set_to_none=True)
        q1_loss = huber(self.q1(z, c_star), y)
        q2_loss = huber(self.q2(z, c_star), y)
        (q1_loss + q2_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.proj.parameters()) +
            list(self.q1.parameters()) + list(self.q2.parameters()),
            10.0
        )
        self.q_optim.step()

        # 3) Actor update: sample c1 from target actor on z, weight with target critics, update diffusion model
        with torch.no_grad():
            C1 = self._sample_candidates(z, K=1, use_target=True)  # [B,1,D]
            c1 = C1[:, 0, :]                                       # [B,D]
            w = self._weights_no_smooth(z, c1, lam=float(self._current_lambda()))

        policy_loss = rfm_service.update(
            z.detach().cpu().numpy(),
            c1.detach().cpu().numpy(),
            w.detach().cpu().numpy(),
        )

        # 4.5) delayed temperature update
        if (self.step % max(1, int(self.cfg.delay_alpha_update))) == 0:
            self.alpha_optim.zero_grad(set_to_none=True)
            approx_entropy = 0.5 * self.act_dim * torch.log(
                torch.tensor(2.0 * np.pi * np.e, device=self.device, dtype=torch.float32)
                * (0.1 * torch.exp(self.log_alpha)).pow(2)
            )
            alpha_loss = -self.log_alpha * (-approx_entropy.detach() + float(self.cfg.target_entropy))
            alpha_loss.backward()
            self.alpha_optim.step()

        # 5) soft-update target critics and target actor
        if (self.step % int(self.cfg.delay_update)) == 0:
            with torch.no_grad():
                for p_t, p in zip(self.tq1.parameters(), self.q1.parameters()):
                    p_t.mul_(1.0 - self.cfg.tau).add_(p, alpha=self.cfg.tau)
                for p_t, p in zip(self.tq2.parameters(), self.q2.parameters()):
                    p_t.mul_(1.0 - self.cfg.tau).add_(p, alpha=self.cfg.tau)
            if hasattr(rfm_service, "soft_update_target"):
                rfm_service.soft_update_target(self.cfg.tau)
            elif hasattr(rfm_service, "sync_target_from_current"):
                rfm_service.sync_target_from_current()

        self.step += 1

        return {
            "q1_loss": float(q1_loss.detach().cpu()),
            "q2_loss": float(q2_loss.detach().cpu()),
            "policy_loss": float(policy_loss),
        }


__all__ = ["DPMDGraphDisease", "DPMDGraphConfig", "Experience"]
