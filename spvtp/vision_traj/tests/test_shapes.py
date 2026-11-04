"""Smoke tests: all modules build and forward-pass with expected shapes.

Run::

    python -m spvtp.vision_traj.tests.test_shapes
"""

from __future__ import annotations

import torch

from spvtp.vision_traj.models.policy import VisionTrajPolicy, VisionTrajPolicyConfig


def fake_batch(
    B: int,
    cfg: VisionTrajPolicyConfig,
    prompt_dim: int = 4,
    target_prompt_dim: int | None = None,
):
    img_size = cfg.vision_encoder.get("img_size", 224)
    target_prompt_dim = prompt_dim if target_prompt_dim is None else target_prompt_dim
    return {
        "first_frame": torch.randn(B, 3, img_size, img_size),
        "current_frame": torch.randn(B, 3, img_size, img_size),
        "prompt_object": torch.rand(B, prompt_dim),
        "prompt_target": torch.rand(B, target_prompt_dim),
        "action_history": torch.randn(B, cfg.history_horizon, cfg.action_dim),
        "future_actions": torch.randn(B, cfg.action_horizon, cfg.action_dim),
    }


def main() -> None:
    for head_type in ("diffusion", "flow_matching"):
        cfg = VisionTrajPolicyConfig(
            vision_encoder={
                "type": "timm",
                "model_name": "vit_large_patch14_dinov2",
                "pretrained": True,
                "freeze": True,
                "img_size": 224,
            },
            embed_dim=192,        # keep small for cpu smoke test
            fusion_layers=1,
            fusion_heads=4,
            head_layers=1,
            head_heads=4,
            head_type=head_type,
            num_train_timesteps=20,
            num_inference_steps=4,
        )
        policy = VisionTrajPolicy(cfg)
        policy.eval()

        for prompt_dims in ((4, 4), (2, 2), (0, 0), (4, 0), (0, 4)):
            batch = fake_batch(
                2,
                cfg,
                prompt_dim=prompt_dims[0],
                target_prompt_dim=prompt_dims[1],
            )
            loss = policy.compute_loss(batch)
            assert loss.isfinite(), "loss should be finite"
            initial_noise = torch.randn(2, cfg.action_horizon, cfg.action_dim)
            pred = policy.predict_action(batch, initial_noise=initial_noise)
            pred_repeat = policy.predict_action(batch, initial_noise=initial_noise.clone())
            assert pred.shape == (2, cfg.action_horizon, cfg.action_dim), pred.shape
            assert torch.allclose(pred, pred_repeat), "predict_action should be deterministic for fixed noise"
            print(
                f"head_type={head_type}  prompt_dims={prompt_dims}  "
                f"loss={float(loss):.4f}  pred.shape={tuple(pred.shape)}"
            )


if __name__ == "__main__":
    main()
