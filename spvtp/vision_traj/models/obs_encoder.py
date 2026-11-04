"""Current-frame observation encoder.

Encodes the current RGB frame and recent action history into observation tokens.
By default it shares the TaskEncoder image backbone; controlled backbone
experiments may provide a separate observation encoder while leaving the task
path and trajectory policy unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spvtp.vision_traj.models.task_encoder import TaskEncoder, build_vision_backbone


class ObsEncoder(nn.Module):
    def __init__(
        self,
        task_encoder: TaskEncoder,
        action_dim: int,
        history_horizon: int,
        observation_encoder: Any = None,
    ) -> None:
        super().__init__()
        self.task_encoder = task_encoder
        self.history_horizon = history_horizon
        self.action_dim = action_dim

        D = task_encoder.embed_dim
        if observation_encoder is None:
            self.observation_backbone = None
            self.observation_img_proj = None
        else:
            self.observation_backbone = build_vision_backbone(observation_encoder)
            self.observation_img_proj = nn.Linear(
                self.observation_backbone.embed_dim, D
            )
        if history_horizon > 0:
            self.action_mlp = nn.Sequential(
                nn.Linear(action_dim, D),
                nn.GELU(),
                nn.Linear(D, D),
            )
            self.history_pos = nn.Parameter(torch.zeros(history_horizon, D))
        else:
            self.action_mlp = None
            self.history_pos = None

        # marker distinguishing current-frame token type from history token type
        self.type_embed = nn.Embedding(2, D)  # 0 = current frame, 1 = action history

    def forward(
        self,
        current_frame: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        if self.observation_backbone is None:
            img_tokens = self.task_encoder.encode_image(current_frame)
        else:
            img_tokens = self.observation_img_proj(
                self.observation_backbone(current_frame)
            )
        img_tokens = img_tokens + self.type_embed.weight[0]

        if self.history_horizon > 0 and action_history.shape[1] > 0:
            hist = self.action_mlp(action_history)  # (B, k, D)
            hist = hist + self.history_pos + self.type_embed.weight[1]
            return torch.cat([img_tokens, hist], dim=1)
        return img_tokens
