from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


@dataclass
class NodeSetEncoderConfig:
    flat_key: str = "flat"
    node_key: str = "node_x"

    flat_dim: int = 39
    node_feat_dim: int = 8

    R: int = 4
    O: int = 4

    present_idx: int = 7

    flat_hidden: int = 128
    node_hidden: int = 128
    fused_hidden: int = 256
    out_dim: int = 256

    dropout: float = 0.0
    eta_uncertainty: bool = False


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ETAHead(nn.Module):
    def __init__(self, node_hidden: int, hidden: int = 128, dropout: float = 0.0, eta_uncertainty: bool = False):
        super().__init__()
        self.eta_uncertainty = bool(eta_uncertainty)
        if not self.eta_uncertainty:
            # Keep legacy parameter names for backward checkpoint compatibility.
            self.mlp = nn.Sequential(
                nn.Linear(node_hidden * 2, hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden, 1),
            )
            self.shared = None
            self.mu_head = None
            self.log_var_head = None
        else:
            self.mlp = None
            self.shared = nn.Sequential(
                nn.Linear(node_hidden * 2, hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )
            self.mu_head = nn.Linear(hidden, 1)
            self.log_var_head = nn.Linear(hidden, 1)

    def forward(self, rider_h: torch.Tensor, order_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """
        rider_h: [B, R, H]
        order_h: [B, O, H]
        returns: [B, R, O]
        """
        B, R, H = rider_h.shape
        _, O, _ = order_h.shape

        rr = rider_h.unsqueeze(2).expand(B, R, O, H)
        oo = order_h.unsqueeze(1).expand(B, R, O, H)
        pair = torch.cat([rr, oo], dim=-1)  # [B, R, O, 2H]
        if not self.eta_uncertainty:
            eta = self.mlp(pair).squeeze(-1)    # [B, R, O]
            return eta, None

        h = self.shared(pair)
        mu = self.mu_head(h).squeeze(-1)
        raw_log_var = self.log_var_head(h).squeeze(-1)
        log_var = torch.clamp(raw_log_var, min=-10.0, max=10.0)
        return mu, log_var


class NodeSetJointEncoder(nn.Module):
    def __init__(self, cfg: NodeSetEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.flat_mlp = MLP(cfg.flat_dim, cfg.flat_hidden, cfg.flat_hidden, dropout=cfg.dropout)
        self.node_mlp = MLP(cfg.node_feat_dim, cfg.node_hidden, cfg.node_hidden, dropout=cfg.dropout)
        self.eta_head = ETAHead(
            cfg.node_hidden,
            hidden=128,
            dropout=cfg.dropout,
            eta_uncertainty=bool(cfg.eta_uncertainty),
        )
        self.last_eta_log_var: torch.Tensor | None = None

        fused_in = cfg.flat_hidden + cfg.node_hidden + (cfg.R * cfg.O)

        self.proj = nn.Sequential(
            nn.Linear(fused_in, cfg.fused_hidden),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(cfg.fused_hidden, cfg.out_dim),
            nn.ReLU(),
        )

    @staticmethod
    def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg

        flat = obs[cfg.flat_key].float()
        flat = self._ensure_batch(flat)              # [B, flat_dim]
        h_flat = self.flat_mlp(flat)                 # [B, flat_hidden]

        node_x = obs[cfg.node_key].float()
        if node_x.dim() == 2:
            node_x = node_x.unsqueeze(0)             # [B, N, F]
        B, N, _ = node_x.shape

        h_nodes = self.node_mlp(node_x)              # [B, N, H]

        present = node_x[:, :, cfg.present_idx]      # [B, N]
        mask = (present > 0.5).float().unsqueeze(-1) # [B, N, 1]

        h_nodes_masked = h_nodes * mask
        denom = mask.sum(dim=1).clamp(min=1.0)       # [B, 1]
        g = h_nodes_masked.sum(dim=1) / denom        # [B, H]

        R = cfg.R
        O = cfg.O
        assert N >= R + O, f"node_x N={N} must be >= R+O={R+O}"

        rider_h = h_nodes[:, 0:R, :]                 # [B, R, H]
        order_h = h_nodes[:, R:R+O, :]               # [B, O, H]
        eta_mu, eta_log_var = self.eta_head(rider_h, order_h)        # [B, R, O]
        self.last_eta_log_var = eta_log_var

        eta_flat = eta_mu.reshape(B, R * O)             # [B, R*O]
        fused = torch.cat([h_flat, g, eta_flat], dim=-1)
        features = self.proj(fused)                  # [B, out_dim]
        return features, eta_mu


class SB3NodeSetFeaturesExtractor(BaseFeaturesExtractor):
    """
    Dict obs keys:
      - flat:   (flat_dim,)
      - node_x: (R+O, node_feat_dim)
    """

    def __init__(self, observation_space, cfg: NodeSetEncoderConfig = None, **kwargs):
        if cfg is None:
            flat_space = observation_space.spaces.get("flat", None)
            node_space = observation_space.spaces.get("node_x", None)

            flat_dim = int(flat_space.shape[0]) if flat_space is not None else int(kwargs.get("flat_dim", 39))
            node_feat_dim = int(node_space.shape[1]) if node_space is not None else int(kwargs.get("node_feat_dim", 8))

            n_nodes = int(node_space.shape[0]) if node_space is not None else int(kwargs.get("n_nodes", 8))
            R = int(kwargs.get("R", 4))
            O = int(kwargs.get("O", max(1, n_nodes - R)))

            cfg = NodeSetEncoderConfig(
                flat_dim=flat_dim,
                node_feat_dim=node_feat_dim,
                R=R,
                O=O,
                present_idx=int(kwargs.get("present_idx", 7)),
                flat_hidden=int(kwargs.get("flat_hidden", 128)),
                node_hidden=int(kwargs.get("node_hidden", 128)),
                fused_hidden=int(kwargs.get("fused_hidden", 256)),
                out_dim=int(kwargs.get("out_dim", 256)),
                dropout=float(kwargs.get("dropout", 0.0)),
                eta_uncertainty=bool(kwargs.get("eta_uncertainty", False)),
            )

        super().__init__(observation_space, features_dim=cfg.out_dim)
        self.encoder = NodeSetJointEncoder(cfg)

        # for debugging / Phase 2 loss plumbing
        self.last_eta: torch.Tensor | None = None
        self.last_eta_log_var: torch.Tensor | None = None
        self.last_eta_sigma: torch.Tensor | None = None

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        features, eta = self.encoder(observations)
        self.last_eta = eta
        self.last_eta_log_var = getattr(self.encoder, "last_eta_log_var", None)
        if self.last_eta_log_var is None:
            self.last_eta_sigma = None
        else:
            self.last_eta_sigma = torch.exp(0.5 * self.last_eta_log_var).clamp(min=1e-6)
        return features
