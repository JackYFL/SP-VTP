"""One-update smoke test for DINO task encoder + pi0.5 observation encoder."""

from __future__ import annotations

import argparse
import json

import torch

from spvtp.vision_traj.models.policy import VisionTrajPolicy, VisionTrajPolicyConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = VisionTrajPolicyConfig(
        action_dim=10,
        action_horizon=16,
        history_horizon=4,
        embed_dim=768,
        vision_encoder={
            "type": "timm",
            "model_name": "vit_base_patch14_dinov2",
            "pretrained": True,
            "freeze": True,
            "img_size": 224,
        },
        observation_encoder={
            "type": "pi05_siglip",
            "checkpoint_path": args.vision_checkpoint,
            "pretrained": False,
            "freeze": True,
            "img_size": 224,
        },
        fusion_layers=4,
        fusion_heads=12,
        task_fusion_type="cross_attention",
        task_return_image_summary=True,
        head_layers=6,
        head_heads=12,
        head_type="flow_matching",
        num_inference_steps=10,
    )
    policy = VisionTrajPolicy(cfg).to(device)
    policy.train()
    batch = {
        "first_frame": torch.rand(1, 3, 224, 224, device=device),
        "current_frame": torch.rand(1, 3, 224, 224, device=device),
        "prompt_object": torch.rand(1, 4, device=device),
        "prompt_target": torch.rand(1, 4, device=device),
        "action_history": torch.rand(1, 4, 10, device=device),
        "future_actions": torch.rand(1, 16, 10, device=device),
    }
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        condition = policy._encode_cond(batch)
        loss = policy(batch)
    loss.backward()

    task_frozen = all(
        not parameter.requires_grad
        for parameter in policy.task_encoder.backbone.parameters()
    )
    observation_frozen = all(
        not parameter.requires_grad
        for parameter in policy.obs_encoder.observation_backbone.parameters()
    )
    trainable_with_grad = sum(
        parameter.numel()
        for parameter in policy.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    result = {
        "loss": float(loss.detach()),
        "condition_shape": list(condition.shape),
        "task_tokens": 5,
        "observation_image_tokens": 256,
        "history_tokens": 4,
        "task_backbone_frozen": task_frozen,
        "observation_backbone_frozen": observation_frozen,
        "trainable_parameters_with_gradient": trainable_with_grad,
        "peak_gpu_memory_mib": (
            int(torch.cuda.max_memory_allocated() / (1024**2))
            if device.type == "cuda"
            else None
        ),
    }
    if not torch.isfinite(loss):
        raise RuntimeError("hybrid smoke-test loss is not finite")
    if not task_frozen or not observation_frozen or trainable_with_grad == 0:
        raise RuntimeError(f"unexpected gradient/freeze state: {result}")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
