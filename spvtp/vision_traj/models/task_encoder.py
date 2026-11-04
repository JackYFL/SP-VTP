"""Task encoder: first frame + object/target prompts → Task Tokens.

Design (see plan.md §5):
    * Frozen DINOv2 produces patch tokens of the first frame.
    * A lightweight SAM-style prompt encoder turns each bbox / point into
      learned tokens tagged with an ``object`` vs ``target`` role embedding.
    * A small transformer with cross-attention fuses the prompt tokens into
      the image tokens, producing the Task Tokens used as conditioning.

The same DINOv2 backbone is exposed via :meth:`forward_image` so the current
frame encoder can reuse it (shared weights, single forward per call).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import timm
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fourier / positional embedding for prompts


def fourier_pe(x: torch.Tensor, num_bands: int = 16, max_freq: float = 64.0) -> torch.Tensor:
    """x: (..., D_in) in [0,1] -> (..., D_in * 2 * num_bands)."""
    freqs = torch.linspace(1.0, max_freq, num_bands, device=x.device, dtype=x.dtype)
    xf = x.unsqueeze(-1) * freqs  # (..., D_in, num_bands)
    enc = torch.cat([torch.sin(xf), torch.cos(xf)], dim=-1)
    return enc.flatten(-2)


# ---------------------------------------------------------------------------
# Image backbone wrapper (timm)


def _get_vision_encoder_value(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def resolve_vision_encoder_config(vision_encoder: Any = None) -> dict[str, Any]:
    """Normalize the vision encoder config into one dict.

    ``vision_encoder`` may be:
    - ``None``: use the default timm DINOv2 config.
    - ``str``: a timm model name.
    - mapping/object with keys: type, model_name, pretrained, freeze, img_size.

    Supported ``type`` values:
    - ``timm`` / ``dinov2``: use ``model_name`` directly.
    - ``sam``: default ``samvit_base_patch16.sa1b`` through timm.
    - ``pe``: default ``vit_pe_core_base_patch16_224.fb`` through timm.
    - ``siglip``: default ``vit_base_patch16_siglip_224.v2_webli`` through timm.
    - ``eva2`` / ``eva02``: default ``eva02_base_patch14_224.mim_in22k`` through timm.
    - ``pi05_siglip``: the frozen SigLIP-So400M/14 vision tower extracted from
      the official pi0.5 VLA checkpoint.
    """
    cfg = {
        "type": "timm",
        "model_name": "vit_base_patch14_dinov2",
        "pretrained": True,
        "freeze": True,
        "img_size": 224,
    }
    if vision_encoder is None:
        return cfg
    if isinstance(vision_encoder, str):
        cfg["model_name"] = vision_encoder
        return cfg

    cfg["type"] = _get_vision_encoder_value(vision_encoder, "type", cfg["type"])
    cfg["model_name"] = _get_vision_encoder_value(
        vision_encoder, "model_name", cfg["model_name"]
    )
    cfg["pretrained"] = _get_vision_encoder_value(
        vision_encoder, "pretrained", cfg["pretrained"]
    )
    cfg["freeze"] = _get_vision_encoder_value(vision_encoder, "freeze", cfg["freeze"])
    cfg["img_size"] = _get_vision_encoder_value(
        vision_encoder, "img_size", cfg["img_size"]
    )
    cfg["checkpoint_path"] = _get_vision_encoder_value(
        vision_encoder, "checkpoint_path", None
    )
    cfg = _apply_vision_encoder_type_defaults(cfg)
    return cfg


def _apply_vision_encoder_type_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    encoder_type = str(cfg["type"]).lower()
    if encoder_type in ("timm", "dinov2"):
        return cfg

    defaults = {
        "sam": "samvit_base_patch16.sa1b",
        "pe": "vit_pe_core_base_patch16_224.fb",
        "siglip": "vit_base_patch16_siglip_224.v2_webli",
        "eva2": "eva02_base_patch14_224.mim_in22k",
        "eva02": "eva02_base_patch14_224.mim_in22k",
    }
    if encoder_type not in defaults:
        return cfg

    model_name = cfg.get("model_name")
    if not model_name or "dinov2" in str(model_name):
        cfg["model_name"] = defaults[encoder_type]
    return cfg


class TimmVisionBackbone(nn.Module):
    """timm vision model wrapper producing ``(B, N, C)`` image tokens."""

    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2",
        img_size: int = 224,
        pretrained: bool = True,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        try:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, img_size=img_size,
            )
        except TypeError:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0,
            )
        self.embed_dim: int = self.model.num_features
        self.num_patches: Optional[int] = getattr(
            getattr(self.model, "patch_embed", None), "num_patches", None
        )
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self._frozen = freeze

    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen:
            self.model.eval()
        return self

    @torch.no_grad()
    def _forward_frozen(self, img: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(img)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) → (B, N_tokens, C).  Includes CLS token if present."""
        if self._frozen:
            feats = self._forward_frozen(img)
        else:
            feats = self.model.forward_features(img)
        return self._as_tokens(feats)

    def _as_tokens(self, feats: Any) -> torch.Tensor:
        if isinstance(feats, Mapping):
            if "x_norm_patchtokens" in feats:
                tokens = feats["x_norm_patchtokens"]
                cls = feats.get("x_norm_clstoken")
                if cls is not None:
                    tokens = torch.cat([cls.unsqueeze(1), tokens], dim=1)
                return tokens
            for key in ("x", "features", "last_hidden_state"):
                if key in feats:
                    return self._as_tokens(feats[key])
            raise ValueError(f"unsupported timm feature dict keys: {sorted(feats)}")
        if isinstance(feats, (tuple, list)):
            if not feats:
                raise ValueError("empty timm feature output")
            return self._as_tokens(feats[-1])
        if feats.ndim == 4:
            return feats.flatten(2).transpose(1, 2)
        if feats.ndim == 3:
            return feats
        if feats.ndim == 2:
            return feats.unsqueeze(1)
        raise ValueError(f"unsupported timm feature shape: {tuple(feats.shape)}")


DinoV2Backbone = TimmVisionBackbone


class Pi05SiglipBackbone(nn.Module):
    """Frozen pi0.5 SigLIP-So400M vision tower producing patch tokens.

    The checkpoint is a strict timm-format extraction made by
    ``extract_pi05_siglip.py``.  Inputs arrive in ``[0, 1]`` from the shared
    dataset and are converted to pi0.5's native ``[-1, 1]`` convention.
    """

    MODEL_NAME = "vit_so400m_patch14_siglip_224"

    def __init__(
        self,
        checkpoint_path: str | Path,
        img_size: int = 224,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if int(img_size) != 224:
            raise ValueError(f"pi0.5 SigLIP expects 224x224 images, got {img_size}")
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"pi0.5 SigLIP checkpoint not found: {checkpoint_path}")
        self.model = timm.create_model(
            self.MODEL_NAME,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
            global_pool="",
        )
        from safetensors.torch import load_file

        state = load_file(str(checkpoint_path), device="cpu")
        self.model.load_state_dict(state, strict=True)
        self.embed_dim = int(self.model.num_features)
        self.num_patches = int(self.model.patch_embed.num_patches)
        self._frozen = bool(freeze)
        if self._frozen:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen:
            self.model.eval()
        return self

    @torch.no_grad()
    def _forward_frozen(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(image)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image * 2.0 - 1.0
        if self._frozen:
            return self._forward_frozen(image)
        return self.model.forward_features(image)


def build_vision_backbone(vision_encoder: Any) -> nn.Module:
    cfg = resolve_vision_encoder_config(vision_encoder)
    encoder_type = str(cfg["type"]).lower()
    if encoder_type in ("pi05_siglip", "pi0.5_siglip", "vla_pi05_siglip"):
        checkpoint_path = cfg.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("pi05_siglip requires vision_encoder.checkpoint_path")
        return Pi05SiglipBackbone(
            checkpoint_path=checkpoint_path,
            img_size=int(cfg["img_size"]),
            freeze=bool(cfg["freeze"]),
        )
    if encoder_type not in ("timm", "dinov2", "sam", "pe", "siglip", "eva2", "eva02"):
        raise ValueError(
            f"unknown vision_encoder.type: {cfg['type']!r} "
            "(expected a timm family or 'pi05_siglip')"
        )
    return TimmVisionBackbone(
        model_name=str(cfg["model_name"]),
        img_size=int(cfg["img_size"]),
        pretrained=bool(cfg["pretrained"]),
        freeze=bool(cfg["freeze"]),
    )


# ---------------------------------------------------------------------------
# Prompt encoder (lightweight SAM-style)


class PromptEncoder(nn.Module):
    """Encode bbox, point, missing-role, or no-prompt inputs into tokens."""

    def __init__(self, embed_dim: int, num_bands: int = 16) -> None:
        super().__init__()
        self.num_bands = num_bands
        # bbox has 2 corners; point has 1 point. We always feed 2 coords (a,b)
        # and a "corner kind" embedding so the same MLP handles both.
        in_dim = 2 * 2 * num_bands  # 2d xy * (sin,cos) * num_bands
        self.coord_mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # role: 0 = object-corner-a, 1 = object-corner-b, 2 = target-corner-a, 3 = target-corner-b
        # for point prompts we only use 0 and 2.
        self.role_embed = nn.Embedding(4, embed_dim)
        self.null_prompt = nn.Parameter(torch.zeros(2, embed_dim))

    def _encode_coords(self, xy: torch.Tensor, role_ids: torch.Tensor) -> torch.Tensor:
        """xy: (B, N, 2) in [0,1]; role_ids: (B, N) long → (B, N, D)."""
        pe = fourier_pe(xy, num_bands=self.num_bands)
        feat = self.coord_mlp(pe) + self.role_embed(role_ids)
        return feat

    def forward(
        self, prompt_object: torch.Tensor, prompt_target: torch.Tensor
    ) -> torch.Tensor:
        """Return role-tagged prompt tokens, including nulls for missing roles."""
        B = prompt_object.shape[0]
        if prompt_target.shape[0] != B:
            raise ValueError(
                "object and target prompt batch sizes must match "
                f"({B} != {prompt_target.shape[0]})"
            )

        obj_dim = prompt_object.shape[-1]
        tgt_dim = prompt_target.shape[-1]
        valid_dims = (0, 2, 4)
        if obj_dim not in valid_dims or tgt_dim not in valid_dims:
            raise ValueError(
                f"unexpected prompt dims object={obj_dim}, target={tgt_dim}; "
                f"expected each in {valid_dims}"
            )
        if obj_dim == 0 and tgt_dim == 0:
            return self.null_prompt.unsqueeze(0).expand(B, -1, -1)

        def encode_role(
            prompt: torch.Tensor,
            dim: int,
            role_ids: tuple[int, int],
            null_index: int,
        ) -> torch.Tensor:
            if dim == 0:
                null = self.null_prompt[null_index : null_index + 1]
                return null.unsqueeze(0).expand(B, -1, -1)
            n_tokens = 2 if dim == 4 else 1
            xy = prompt.view(B, n_tokens, 2)
            ids = torch.tensor(
                role_ids[:n_tokens], device=prompt.device, dtype=torch.long
            ).expand(B, n_tokens)
            return self._encode_coords(xy, ids)

        object_tokens = encode_role(prompt_object, obj_dim, (0, 1), null_index=0)
        target_tokens = encode_role(prompt_target, tgt_dim, (2, 3), null_index=1)
        return torch.cat([object_tokens, target_tokens], dim=1)


# ---------------------------------------------------------------------------
# Task encoder


class TaskEncoder(nn.Module):
    """First-frame image + prompts → Task Tokens.

    Returns a tensor of shape ``(B, N_task, D)`` which is concatenated with
    current-frame features (in the policy) to form the diffusion condition.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        fusion_layers: int = 4,
        fusion_heads: int = 6,
        vision_encoder: Any = None,
        fusion_type: str = "cross_attention",
        return_image_summary: bool = True,
    ) -> None:
        super().__init__()
        fusion_type = fusion_type.lower()
        if fusion_type not in ("self_attention", "cross_attention"):
            raise ValueError(
                f"unknown task fusion_type: {fusion_type!r} "
                "(expected 'self_attention' or 'cross_attention')"
            )
        self.backbone = build_vision_backbone(vision_encoder)
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type
        self.return_image_summary = return_image_summary
        self.img_proj = nn.Linear(self.backbone.embed_dim, embed_dim)

        self.prompt_encoder = PromptEncoder(embed_dim=embed_dim)
        # role-type embedding to distinguish image-token vs prompt-token in fusion
        self.type_embed = nn.Embedding(2, embed_dim)  # 0=img, 1=prompt

        if fusion_type == "self_attention":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=fusion_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True,
                activation="gelu",
            )
            self.fusion = nn.TransformerEncoder(enc_layer, num_layers=fusion_layers)
        else:
            dec_layer = nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=fusion_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True,
                activation="gelu",
            )
            self.fusion = nn.TransformerDecoder(dec_layer, num_layers=fusion_layers)

    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        """Returns (B, N, D) image tokens projected to embed_dim."""
        tokens = self.backbone(img)
        return self.img_proj(tokens)

    def forward(
        self,
        first_frame: torch.Tensor,
        prompt_object: torch.Tensor,
        prompt_target: torch.Tensor,
    ) -> torch.Tensor:
        img_tokens = self.encode_image(first_frame)  # (B, N_img, D)
        prompt_tokens = self.prompt_encoder(prompt_object, prompt_target)  # (B, K, D)

        img_tokens = img_tokens + self.type_embed.weight[0]
        prompt_tokens = prompt_tokens + self.type_embed.weight[1]

        if self.fusion_type == "self_attention":
            tokens = torch.cat([img_tokens, prompt_tokens], dim=1)  # (B, N_img+K, D)
            return self.fusion(tokens)

        fused_prompt = self.fusion(tgt=prompt_tokens, memory=img_tokens)
        if not self.return_image_summary:
            return fused_prompt

        image_summary = img_tokens[:, :1]
        return torch.cat([image_summary, fused_prompt], dim=1)
