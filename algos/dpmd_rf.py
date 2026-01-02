# algos/dpmd_rf.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, NamedTuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from models.rfm.service import rfm_service

# -----------------------------
# Replay experience tuple
# -----------------------------
class Experience(NamedTuple):
    obs:         np.ndarray  # [B, F]
    action:      np.ndarray  # [B, D]  executed coefficient \tilde c
    reward:      np.ndarray  # [B]
    next_obs:    np.ndarray  # [B, F]
    done:        np.ndarray  # [B]     (0/1)
    action_star: np.ndarray  # [B, D]  greedy c* sampled under π_old
    policy_id:   np.ndarray  # [B]     integer policy version that produced the sample

# -----------------------------
# Twin critic (min over Q1,Q2)
# -----------------------------
class QNet(nn.Module):
    """Approximates Q_e(s, c) on S^{D-1} with a simple MLP."""
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
# Small utils
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
    """Ensure last-dim == target_F by slice or right-pad with zeros."""
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
# Config (hyperparams)
# -----------------------------
@dataclass
class DPMDConfig:
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

    # temperature schedule for per-state softmax weights
    lambda_start: float = 2.0
    lambda_end:   float = 0.8
    lambda_steps: int   = 10_000   # updates over which to anneal


    # Execution noise (on-sphere)
    kappa_exec: float = 28.0

    # Smoothed Bellman operator
    kappa_smooth: float = 28.0
    M_smooth: int = 16
    J_smooth: int = 1

    # running statistics for Q normalization: EMA coeff
    q_running_beta: float = 0.05
    q_norm_clip: float = 3          # clamp normalized Q to [-q_norm_clip, q_norm_clip]
    # temperature learning (log_alpha) to scale normalized Q in weights
    alpha_lr: float = 3e-2
    delay_alpha_update: int = 180
    target_entropy: float = 0.0

    flow_steps: int = 36           # RFM flow integration steps

class DPMD:
    """
    DPMD with RFM policy on S^{D-1}.
      • Critics train to smoothed Bellman targets
      • Actor trains via MD-weighted RFM using target critics
    """
    def __init__(self, obs_dim: int, act_dim: int,
                 device: Optional[torch.device] = None,
                 cfg: DPMDConfig = DPMDConfig()):
        self.cfg = cfg
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert self.act_dim > 0 and self.obs_dim > 0, "Bad dims"

        # Critics and targets (twin Q; use min)
        self.q1 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.q2 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.tq1 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.tq2 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())

        self.q_optim = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.cfg.lr
        )

        # MD weight baseline (EMA of target-critic values)
        self.step = 0
        self.policy_version = 0
        self._q_mean: float = 0.0
        self._q_std: float = 1.0

        # Learnable temperature log_alpha
        self.log_alpha = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)

        # Start cooler so MD weights aren't too peaky early
        with torch.no_grad():
            self.log_alpha.copy_(torch.tensor(-0.5, device=self.device))  # temp ≈ 1.105


        # Keep target actor aligned initially
        if hasattr(rfm_service, "sync_target_from_current"):
            rfm_service.sync_target_from_current()

    # -----------------------------------------------------
    # Pretraining (myopic)
    # -----------------------------------------------------
    def pretrain_critics_step(self, batch: Experience, *, huber_delta: float = 5.0) -> float:
        """One supervised step for critics using *myopic* targets (gamma=0): y = r."""
        obs      = _fit_width(_to_tensor(batch.obs,      self.device), self.obs_dim)
        c_clean = _to_tensor(batch.action_star, self.device)

        rew      = _to_tensor(batch.reward,  self.device).view(-1)  # target

        def huber(a, b, delta=huber_delta):
            x = a - b
            ax = torch.abs(x)
            return torch.where(ax < delta, 0.5*x*x, delta*(ax - 0.5*delta)).mean()

        self.q_optim.zero_grad(set_to_none=True)
        q1_loss = huber(self.q1(obs, c_clean), rew)
        q2_loss = huber(self.q2(obs, c_clean), rew)
        (q1_loss + q2_loss).backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
        self.q_optim.step()

        # keep targets aligned early on
        with torch.no_grad():
            for p_t, p in zip(self.tq1.parameters(), self.q1.parameters()):
                p_t.copy_(p)
            for p_t, p in zip(self.tq2.parameters(), self.q2.parameters()):
                p_t.copy_(p)

        return float((q1_loss + q2_loss).detach().cpu())


    # -----------------------------
    # Actor helpers
    # -----------------------------
    @torch.no_grad()
    def _sample_candidates(self, obs: Tensor, K: int, *, use_target: bool = False) -> Tensor:
        """Draw K coefficients from current (or target) policy on S^{D-1}."""
        obs = _fit_width(obs, self.obs_dim)
        if use_target and hasattr(rfm_service, "sample_target"):
            C_np = rfm_service.sample_target(obs_np=obs.detach().cpu().numpy(),
                                             K=K, steps=self.cfg.flow_steps, kappa=None, J_noise=1)
        else:
            C_np = rfm_service.sample(obs_np=obs.detach().cpu().numpy(),
                                      K=K, steps=self.cfg.flow_steps, kappa=None, J_noise=1)
        return _to_tensor(C_np, self.device)  # [B, K, D]

    @torch.no_grad()
    def _greedy_from_candidates(self, obs: Tensor, C: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Pick argmax_i min(Q1,Q2)(s, c_i) among candidates C. Also report target critics at argmax."""
        obs = _fit_width(obs, self.obs_dim)           # [B,F]
        B, K, D = C.shape
        rep_obs = obs.unsqueeze(1).expand(B, K, self.obs_dim).reshape(B * K, self.obs_dim)
        flat_C = C.reshape(B * K, D)

        # Online min-Q
        q1 = self.q1(rep_obs, flat_C).view(B, K)
        q2 = self.q2(rep_obs, flat_C).view(B, K)
        q_min = torch.minimum(q1, q2)

        idx = torch.argmax(q_min, dim=1)              # [B]
        a_star = C[torch.arange(B, device=self.device), idx]  # [B,D]
        q_star_online = q_min[torch.arange(B, device=self.device), idx]

        # Target critics on the same chosen candidates
        tq1 = self.tq1(rep_obs, flat_C).view(B, K)
        tq2 = self.tq2(rep_obs, flat_C).view(B, K)
        tmin = torch.minimum(tq1, tq2)
        q_star_target = tmin[torch.arange(B, device=self.device), idx]
        return a_star, q_star_online, q_star_target
    
    def _current_lambda(self) -> float:
        s = min(1.0, self.step / max(1, self.cfg.lambda_steps))
        return float((1.0 - s) * self.cfg.lambda_start + s * self.cfg.lambda_end)

    @torch.no_grad()
    def _weights_no_smooth(
        self,
        s_old: Tensor,   # [B, F]
        c1: Tensor,      # [B, D]
        *,
        lam: float,
        w_clip: Optional[float] = None,
    ) -> Tensor:
        """
        MD weights w(s,c1) using target critics evaluated at c1 (NO vMF perturbation),
        but KEEP the original EMA + batch normalization for stability.
        """
        # 1) Score with target critics at c1 (no smoothing)
        tq1 = self.tq1(s_old, c1)
        tq2 = self.tq2(s_old, c1)
        q = torch.minimum(tq1, tq2)  # [B], raw values

        # 2) Update running stats on RAW q (important!)
        q_mean_batch = q.mean()
        q_std_batch  = q.std() + 1e-6

        self._q_mean = (1 - self.cfg.q_running_beta) * self._q_mean + self.cfg.q_running_beta * float(q_mean_batch)
        self._q_std  = (1 - self.cfg.q_running_beta) * self._q_std  + self.cfg.q_running_beta * float(q_std_batch)

        # 3) Normalize (EMA z-score and batch z-score), then blend+clip (same spirit as your code)
        q_norm_ema   = (q - self._q_mean) / (self._q_std + 1e-6)
        q_norm_batch = (q - q_mean_batch) / q_std_batch

        q_norm = 0.5 * q_norm_ema + 0.5 * q_norm_batch
        q_norm = torch.clamp(q_norm, -self.cfg.q_norm_clip, self.cfg.q_norm_clip)

        # 4) Convert to weights
        alpha = torch.exp(self.log_alpha).detach()
        lam = max(float(lam), 1e-6)
        logits = (alpha * q_norm) / lam

        w = torch.exp(logits)
        if w_clip is not None:
            w = torch.clamp(w, max=float(w_clip))
        return w


    # -----------------------------
    # Smoothed Bellman target V^b_κ(s')
    # -----------------------------
    @torch.no_grad()
    def _smoothed_value(self, next_obs: Tensor) -> Tensor:
        next_obs = _fit_width(next_obs, self.obs_dim)  # [B,F]
        B = next_obs.shape[0]
        M = max(1, int(self.cfg.M_smooth))
        J = max(1, int(self.cfg.J_smooth))
        kappa = float(self.cfg.kappa_smooth)

        # Candidates from target actor
        Cprime = self._sample_candidates(next_obs, M, use_target=True)  # [B,M,D]

        # vMF perturbations for smoothing
        cm = Cprime.reshape(B * M, self.act_dim)
        cm_tilde = rfm_service.perturb(cm.detach().cpu().numpy(), kappa=kappa, J=J)  # [B*M, J, D]
        Chat = _to_tensor(cm_tilde, self.device).reshape(B, M, J, self.act_dim)


        rep_obs = next_obs.view(B, 1, 1, self.obs_dim)\
                        .expand(B, M, J, self.obs_dim)\
                        .reshape(B*M*J, self.obs_dim)
        flat_c  = Chat.reshape(B*M*J, self.act_dim)

        q1 = self.tq1(rep_obs, flat_c).view(B, M, J)
        q2 = self.tq2(rep_obs, flat_c).view(B, M, J)
        qmin = torch.minimum(q1, q2)                                    # [B,M,J]

        qflat = qmin.view(B, -1)                 # [B, M*J]
        k = int(0.2 * qflat.shape[1])           # trim 20%
        vals, _ = torch.sort(qflat, dim=1, descending=True)
        Vb = vals[:, :-k].mean(dim=1) if k>0 else vals.mean(dim=1)

        return Vb



    # -----------------------------
    # Main update
    # -----------------------------
    def update(self, batch: Experience) -> Dict[str, float]:
        # Parse batch
        obs      = _fit_width(_to_tensor(batch.obs,      self.device), self.obs_dim)
        c_clean = _to_tensor(batch.action_star, self.device)  # clean center c
        rew      = _to_tensor(batch.reward,  self.device).view(-1)
        nxt      = _fit_width(_to_tensor(batch.next_obs, self.device), self.obs_dim)
        done     = _to_tensor(batch.done,    self.device).view(-1)

        # Optional reward scaling
        rew = rew * self.cfg.reward_scale

        # 1) Smoothed target
        with torch.no_grad():
            Vb = self._smoothed_value(nxt)
            y  = rew + (1.0 - done) * self.cfg.gamma * Vb

        # 2) Critic updates
        def huber(a, b, delta=2.0):
            x = a - b
            absx = torch.abs(x)
            return torch.where(absx < delta, 0.5*x*x, delta*(absx - 0.5*delta)).mean()

        self.q_optim.zero_grad(set_to_none=True)
        q1_loss = huber(self.q1(obs, c_clean), y, delta=5.0)
        q2_loss = huber(self.q2(obs, c_clean), y, delta=5.0)
            
        (q1_loss + q2_loss).backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters())+list(self.q2.parameters()), 10.0)
        self.q_optim.step()

        # 3) Actor update: sample states from replay, sample c from target actor compute MD weights from target critics (no smoothing), then update online actor.”
        with torch.no_grad():
            s_old = _fit_width(_to_tensor(batch.obs, self.device), self.obs_dim)      # [B, F]
            # sample c1 ~ pi_old(.|s) using target actor
            C1 = self._sample_candidates(s_old, K=1, use_target=True)  # [B,1,D]
            c1 = C1[:, 0, :]                                           # [B,D]
                       # [B, D]
            # weights on \hat c using target critics
            w = self._weights_no_smooth(
                s_old, c1,
                lam=float(self._current_lambda()),
                w_clip=self.cfg.w_clip
            )


        # Train RFM actor on (s, c1, w)
        policy_loss = rfm_service.update(
            s_old.detach().cpu().numpy(),      # [B, F]
            c1.detach().cpu().numpy(),         # [B, D]  (π_old sample from replay)
            w.detach().cpu().numpy(),          # [B]
        )

        # 4.5) Delayed temperature (log_alpha) update
        # Approximate entropy of a Gaussian with variance (0.1 * exp(log_alpha))^2 per dim
        if (self.step % max(1, int(self.cfg.delay_alpha_update))) == 0:
            self.alpha_optim.zero_grad(set_to_none=True)
            approx_entropy = 0.5 * self.act_dim * torch.log(
                torch.tensor(2.0 * np.pi * np.e, device=self.device, dtype=torch.float32)
                * (0.1 * torch.exp(self.log_alpha)).pow(2)
            )
            # Loss: -log_alpha * (-entropy + target_entropy)
            alpha_loss = -self.log_alpha * (-approx_entropy.detach() + float(self.cfg.target_entropy))
            alpha_loss.backward()
            self.alpha_optim.step()


        # 5) Soft-update target critics and target actor
        if (self.step % self.cfg.delay_update) == 0:
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

        # Logging
        with torch.no_grad():
            C = self._sample_candidates(nxt, self.cfg.num_particles, use_target=False)
            _, q_star_online, q_star_target = self._greedy_from_candidates(nxt, C)

        return {
            "q1_loss": float(q1_loss.detach().cpu()),
            "q2_loss": float(q2_loss.detach().cpu()),
            "policy_loss": float(policy_loss),
            "q_mean_next_online": float(q_star_online.mean().detach().cpu()),
            "q_std_next_online": float(q_star_online.std().detach().cpu()),
            "q_mean_next_target": float(q_star_target.mean().detach().cpu()),
            "q_std_next_target": float(q_star_target.std().detach().cpu()),
        }



    # -----------------------------
    # Public driver helpers
    # -----------------------------
    @torch.no_grad()
    def score_actions(self, obs: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Return min(Q1,Q2)(s, c_i) for candidate set C (used to pick i*)."""
        o = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), device=self.device)
        c = torch.as_tensor(C, device=self.device, dtype=torch.float32)
        o_rep = o.repeat(c.shape[0], 1)
        q = torch.minimum(self.q1(o_rep, c), self.q2(o_rep, c))
        return q.detach().cpu().numpy()

    @torch.no_grad()
    def sample_candidates(self, obs: np.ndarray, K: int) -> np.ndarray:
        """Draw K coefficient candidates from current actor πθ(·|s)."""
        o = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return rfm_service.sample(obs_np=o, K=K, steps=self.cfg.flow_steps, kappa=None, J_noise=1)[0]

__all__ = ["DPMD", "DPMDConfig", "Experience"]