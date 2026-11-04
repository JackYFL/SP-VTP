"""Vision-conditioned trajectory policy.

Composes :class:`TaskEncoder`, :class:`ObsEncoder`, and either a
:class:`DiffusionHead` (DDPM/DDIM) or a :class:`FlowMatchingHead`
(rectified flow + Euler ODE) depending on ``cfg.head_type``. Action
normalization is handled by per-dimension mean/std buffers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler

from spvtp.vision_traj.models.diffusion_head import DiffusionHead
from spvtp.vision_traj.models.flow_matching_head import FlowMatchingHead
from spvtp.vision_traj.models.obs_encoder import ObsEncoder
from spvtp.vision_traj.models.task_encoder import TaskEncoder


@dataclass
class VisionTrajPolicyConfig:
    action_dim: int = 10
    action_horizon: int = 16
    history_horizon: int = 4
    use_task_input: bool = True
    embed_dim: int = 768
    vision_encoder: Any = field(default_factory=dict)
    observation_encoder: Any = None
    fusion_layers: int = 4
    fusion_heads: int = 12
    task_fusion_type: str = "cross_attention"  # "self_attention" | "cross_attention"
    task_return_image_summary: bool = True
    head_layers: int = 6
    head_heads: int = 12
    head_type: str = "flow_matching"      # "diffusion" | "flow_matching"
    num_train_timesteps: int = 100        # diffusion only
    num_inference_steps: int = 10         # diffusion: DDIM steps; flow: Euler steps
    prediction_type: str = "epsilon"      # diffusion only: "epsilon" | "sample"
    flow_time_sampling: str = "beta"      # flow matching: "uniform" | "beta"
    flow_beta_alpha: float = 1.5
    flow_beta_beta: float = 1.0
    flow_time_min_period: float = 4e-3    # OpenPI-style sin/cos time embedding
    flow_time_max_period: float = 4.0
    flow_time_scale: Optional[float] = None  # legacy, ignored

    def __post_init__(self) -> None:
        if self.embed_dim % self.fusion_heads != 0:
            raise ValueError(
                "policy.embed_dim must be divisible by policy.fusion_heads "
                f"(got embed_dim={self.embed_dim}, fusion_heads={self.fusion_heads})"
            )
        if self.embed_dim % self.head_heads != 0:
            raise ValueError(
                "policy.embed_dim must be divisible by policy.head_heads "
                f"(got embed_dim={self.embed_dim}, head_heads={self.head_heads})"
            )
        if self.flow_time_sampling not in {"uniform", "beta"}:
            raise ValueError(
                "policy.flow_time_sampling must be 'uniform' or 'beta' "
                f"(got {self.flow_time_sampling!r})"
            )
        if self.flow_beta_alpha <= 0 or self.flow_beta_beta <= 0:
            raise ValueError(
                "policy.flow_beta_alpha and policy.flow_beta_beta must be positive "
                f"(got {self.flow_beta_alpha}, {self.flow_beta_beta})"
            )
        if self.flow_time_min_period <= 0:
            raise ValueError(
                f"policy.flow_time_min_period must be positive, got {self.flow_time_min_period}"
            )
        if self.flow_time_max_period <= self.flow_time_min_period:
            raise ValueError(
                "policy.flow_time_max_period must be greater than policy.flow_time_min_period "
                f"(got {self.flow_time_max_period} <= {self.flow_time_min_period})"
            )


class VisionTrajPolicy(nn.Module):
    def __init__(self, cfg: VisionTrajPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.task_encoder = TaskEncoder(
            embed_dim=cfg.embed_dim,
            fusion_layers=cfg.fusion_layers,
            fusion_heads=cfg.fusion_heads,
            vision_encoder=cfg.vision_encoder,
            fusion_type=cfg.task_fusion_type,
            return_image_summary=cfg.task_return_image_summary,
        )
        self.obs_encoder = ObsEncoder(
            task_encoder=self.task_encoder,
            action_dim=cfg.action_dim,
            history_horizon=cfg.history_horizon,
            observation_encoder=cfg.observation_encoder,
        )
        if not cfg.use_task_input:
            # The shared image projection/backbone remain active for the
            # current-frame observation path. Prompt encoding and task fusion
            # are task-only modules and should not appear as trainable unused
            # parameters in a Current-Frame-Only baseline.
            task_only_modules = (
                self.task_encoder.prompt_encoder,
                self.task_encoder.type_embed,
                self.task_encoder.fusion,
            )
            for module in task_only_modules:
                for parameter in module.parameters():
                    parameter.requires_grad = False
            if cfg.observation_encoder is not None:
                for module in (
                    self.task_encoder.backbone,
                    self.task_encoder.img_proj,
                ):
                    for parameter in module.parameters():
                        parameter.requires_grad = False
        if cfg.head_type == "diffusion":
            self.diffusion_head = DiffusionHead(
                action_dim=cfg.action_dim,
                action_horizon=cfg.action_horizon,
                embed_dim=cfg.embed_dim,
                n_layers=cfg.head_layers,
                n_heads=cfg.head_heads,
            )
            self.train_scheduler = DDPMScheduler(
                num_train_timesteps=cfg.num_train_timesteps,
                beta_schedule="squaredcos_cap_v2",
                prediction_type=cfg.prediction_type,
            )
            self.infer_scheduler = DDIMScheduler(
                num_train_timesteps=cfg.num_train_timesteps,
                beta_schedule="squaredcos_cap_v2",
                prediction_type=cfg.prediction_type,
            )
        elif cfg.head_type == "flow_matching":
            self.flow_head = FlowMatchingHead(
                action_dim=cfg.action_dim,
                action_horizon=cfg.action_horizon,
                embed_dim=cfg.embed_dim,
                n_layers=cfg.head_layers,
                n_heads=cfg.head_heads,
                time_min_period=cfg.flow_time_min_period,
                time_max_period=cfg.flow_time_max_period,
            )
        else:
            raise ValueError(
                f"unknown head_type: {cfg.head_type!r} (expected 'diffusion' or 'flow_matching')"
            )

        # action normalization (learned from dataset, set via :meth:`set_normalizer`)
        self.register_buffer("action_mean", torch.zeros(cfg.action_dim))
        self.register_buffer("action_std", torch.ones(cfg.action_dim))

    # ------------------------------------------------------------------
    def set_normalizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        assert mean.shape == self.action_mean.shape
        assert std.shape == self.action_std.shape
        self.action_mean.copy_(mean)
        self.action_std.copy_(std.clamp(min=1e-4))

    def _normalize(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.action_mean) / self.action_std

    def _denormalize(self, a: torch.Tensor) -> torch.Tensor:
        return a * self.action_std + self.action_mean

    # ------------------------------------------------------------------
    def _encode_cond(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs_tokens = self.obs_encoder(batch["current_frame"], batch["action_history"])
        if not self.cfg.use_task_input:
            return obs_tokens
        task_tokens = self.task_encoder(
            batch["first_frame"], batch["prompt_object"], batch["prompt_target"]
        )
        return torch.cat([task_tokens, obs_tokens], dim=1)

    def _sample_flow_time(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.cfg.flow_time_sampling == "uniform":
            return torch.rand(batch_size, device=device, dtype=dtype)

        alpha = torch.as_tensor(self.cfg.flow_beta_alpha, device=device, dtype=torch.float32)
        beta = torch.as_tensor(self.cfg.flow_beta_beta, device=device, dtype=torch.float32)
        t = torch.distributions.Beta(alpha, beta).sample((batch_size,))
        return (t * 0.999 + 0.001).to(device=device, dtype=dtype)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Training loss. Kept as ``forward`` so DDP can hook it for gradient sync."""
        actions = self._normalize(batch["future_actions"])  # (B, H, A)
        B = actions.shape[0]
        cond = self._encode_cond(batch)

        if self.cfg.head_type == "diffusion":
            noise = torch.randn_like(actions)
            t = torch.randint(
                0, self.train_scheduler.config.num_train_timesteps,
                (B,), device=actions.device, dtype=torch.long,
            )
            noisy = self.train_scheduler.add_noise(actions, noise, t)
            pred = self.diffusion_head(noisy, t, cond)
            target = noise if self.cfg.prediction_type == "epsilon" else actions
            return F.mse_loss(pred, target)

        # flow matching: x_t = (1-t)*x_0 + t*ε,  target velocity = ε - x_0
        noise = torch.randn_like(actions)
        t = self._sample_flow_time(B, device=actions.device, dtype=actions.dtype)
        t_b = t.view(B, 1, 1)
        x_t = (1.0 - t_b) * actions + t_b * noise
        target = noise - actions
        pred = self.flow_head(x_t, t, cond)
        return F.mse_loss(pred, target)

    compute_loss = forward

    @torch.no_grad()
    def predict_action(
        self,
        batch: Dict[str, torch.Tensor],
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        B = batch["current_frame"].shape[0]
        H, A = self.cfg.action_horizon, self.cfg.action_dim

        if initial_noise is not None and generator is not None:
            raise ValueError("pass either initial_noise or generator, not both")

        cond = self._encode_cond(batch)
        if initial_noise is None:
            x = torch.randn((B, H, A), device=device, generator=generator)
        else:
            if tuple(initial_noise.shape) != (B, H, A):
                raise ValueError(
                    f"expected initial_noise shape {(B, H, A)}, got {tuple(initial_noise.shape)}"
                )
            x = initial_noise.to(device=device, dtype=cond.dtype)

        n_steps = num_inference_steps or self.cfg.num_inference_steps
        if n_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {n_steps}")

        if self.cfg.head_type == "diffusion":
            scheduler = self.infer_scheduler
            scheduler.set_timesteps(n_steps)
            for t in scheduler.timesteps:
                t_batch = t.expand(B).to(device)
                model_output = self.diffusion_head(x, t_batch, cond)
                x = scheduler.step(model_output, t, x, generator=generator).prev_sample
            return self._denormalize(x)

        # flow matching: forward Euler from t=1 (noise) to t=0 (data).
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t_val = 1.0 - k * dt
            t_batch = torch.full((B,), t_val, device=device, dtype=x.dtype)
            v = self.flow_head(x, t_batch, cond)
            x = x - v * dt
        return self._denormalize(x)
