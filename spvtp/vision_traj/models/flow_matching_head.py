"""Transformer flow-matching velocity head for action chunk prediction.

Same architecture as :class:`DiffusionHead`, but the time input is a
continuous scalar ``t ∈ [0, 1]`` instead of a discrete diffusion timestep.

Training convention (rectified-flow / SD3 style):
    x_t = (1 - t) * x_0 + t * ε,    ε ~ N(0, I)
    target velocity: v*(x_t, t) = ε - x_0

Inference: integrate ``dx = v(x, t) dt`` from ``t = 1`` (noise) to
``t = 0`` (data) with forward Euler.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FlowMatchingHead(nn.Module):
    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        embed_dim: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        dropout: float = 0.1,
        time_min_period: float = 4e-3,
        time_max_period: float = 4.0,
    ) -> None:
        super().__init__()
        if time_min_period <= 0:
            raise ValueError(f"time_min_period must be positive, got {time_min_period}")
        if time_max_period <= time_min_period:
            raise ValueError(
                "time_max_period must be greater than time_min_period "
                f"(got {time_max_period} <= {time_min_period})"
            )
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.embed_dim = embed_dim
        self.time_min_period = float(time_min_period)
        self.time_max_period = float(time_max_period)

        self.action_in = nn.Linear(action_dim, embed_dim)
        self.action_pos = nn.Parameter(torch.zeros(action_horizon, embed_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.action_out = nn.Linear(embed_dim, action_dim)

    def forward(
        self,
        x_t: torch.Tensor,            # (B, H, action_dim) interpolated sample
        t: torch.Tensor,              # (B,) float in [0, 1]
        cond_tokens: torch.Tensor,    # (B, N_cond, D)
    ) -> torch.Tensor:
        x = self.action_in(x_t) + self.action_pos
        t_emb = self._time_embedding(t)
        t_emb = self.time_mlp(t_emb).unsqueeze(1)
        x = x + t_emb

        out = self.decoder(tgt=x, memory=cond_tokens)
        return self.action_out(out)  # predicted velocity (B, H, action_dim)

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        if self.embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even for sin/cos time embedding, got {self.embed_dim}")
        t = t.float()
        half_dim = self.embed_dim // 2
        fraction = torch.linspace(0.0, 1.0, half_dim, device=t.device, dtype=t.dtype)
        periods = self.time_min_period * (self.time_max_period / self.time_min_period) ** fraction
        phase = t[:, None] * (2.0 * torch.pi / periods)[None, :]
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
