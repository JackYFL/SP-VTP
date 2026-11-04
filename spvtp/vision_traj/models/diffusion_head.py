"""Transformer diffusion head for action chunk prediction.

A small decoder-style transformer that takes noisy action tokens as queries
and condition tokens (task tokens ++ obs tokens) as the cross-attention
memory. Sinusoidal timestep embeddings are added to the action tokens.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_timestep(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) long or float → (B, dim) sinusoidal embedding."""
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
    )
    args = t.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class DiffusionHead(nn.Module):
    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        embed_dim: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.embed_dim = embed_dim

        self.action_in = nn.Linear(action_dim, embed_dim)
        self.action_pos = nn.Parameter(torch.zeros(action_horizon, embed_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
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
        noisy_actions: torch.Tensor,  # (B, H, action_dim)
        timesteps: torch.Tensor,      # (B,)
        cond_tokens: torch.Tensor,    # (B, N_cond, D)
    ) -> torch.Tensor:
        B, H, _ = noisy_actions.shape
        x = self.action_in(noisy_actions) + self.action_pos  # (B, H, D)
        t_emb = sinusoidal_timestep(timesteps, self.embed_dim)
        t_emb = self.time_mlp(t_emb).unsqueeze(1)  # (B, 1, D)
        x = x + t_emb

        out = self.decoder(tgt=x, memory=cond_tokens)  # (B, H, D)
        return self.action_out(out)  # (B, H, action_dim)
