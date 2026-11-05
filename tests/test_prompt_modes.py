"""Focused tests for full, missing-role, and no-prompt tokenization."""

from __future__ import annotations

import unittest

import torch

from spvtp.vision_traj.models.task_encoder import PromptEncoder
from spvtp.vision_traj.models.policy import VisionTrajPolicy, VisionTrajPolicyConfig


class PromptModeTest(unittest.TestCase):
    def test_prompt_encoder_supports_missing_roles(self) -> None:
        cases = [
            (4, 4, 4),
            (2, 2, 2),
            (0, 0, 2),
            (4, 0, 3),
            (0, 4, 3),
            (2, 0, 2),
            (0, 2, 2),
        ]
        batch_size = 3
        encoder = PromptEncoder(embed_dim=32, num_bands=4)
        for object_dim, target_dim, expected_tokens in cases:
            with self.subTest(object_dim=object_dim, target_dim=target_dim):
                object_prompt = torch.rand(batch_size, object_dim)
                target_prompt = torch.rand(batch_size, target_dim)
                tokens = encoder(object_prompt, target_prompt)
                self.assertEqual(tokens.shape, (batch_size, expected_tokens, 32))
                self.assertTrue(bool(torch.isfinite(tokens).all()))

    def test_prompt_encoder_rejects_invalid_role_dimension(self) -> None:
        encoder = PromptEncoder(embed_dim=32, num_bands=4)
        with self.assertRaisesRegex(ValueError, "unexpected prompt dims"):
            encoder(torch.rand(2, 3), torch.rand(2, 4))

    def test_policy_loss_with_object_only_and_target_only(self) -> None:
        cfg = VisionTrajPolicyConfig(
            action_horizon=4,
            history_horizon=2,
            embed_dim=32,
            vision_encoder={
                "type": "timm",
                "model_name": "resnet18",
                "pretrained": False,
                "freeze": True,
                "img_size": 64,
            },
            fusion_layers=1,
            fusion_heads=4,
            head_layers=1,
            head_heads=4,
            head_type="flow_matching",
            num_inference_steps=2,
        )
        policy = VisionTrajPolicy(cfg)
        for object_dim, target_dim in ((0, 0), (4, 0), (0, 4)):
            with self.subTest(object_dim=object_dim, target_dim=target_dim):
                batch = {
                    "first_frame": torch.rand(1, 3, 64, 64),
                    "current_frame": torch.rand(1, 3, 64, 64),
                    "prompt_object": torch.rand(1, object_dim),
                    "prompt_target": torch.rand(1, target_dim),
                    "action_history": torch.rand(1, 2, 10),
                    "future_actions": torch.rand(1, 4, 10),
                }
                loss = policy.compute_loss(batch)
                self.assertTrue(bool(torch.isfinite(loss)))

    def test_policy_supports_separate_observation_backbone(self) -> None:
        cfg = VisionTrajPolicyConfig(
            action_horizon=4,
            history_horizon=2,
            embed_dim=32,
            vision_encoder={
                "type": "timm",
                "model_name": "resnet18",
                "pretrained": False,
                "freeze": True,
                "img_size": 64,
            },
            observation_encoder={
                "type": "timm",
                "model_name": "resnet34",
                "pretrained": False,
                "freeze": True,
                "img_size": 64,
            },
            fusion_layers=1,
            fusion_heads=4,
            head_layers=1,
            head_heads=4,
            head_type="flow_matching",
            num_inference_steps=2,
        )
        policy = VisionTrajPolicy(cfg)
        self.assertIsNotNone(policy.obs_encoder.observation_backbone)
        self.assertIsNot(
            policy.obs_encoder.observation_backbone,
            policy.task_encoder.backbone,
        )
        batch = {
            "first_frame": torch.rand(1, 3, 64, 64),
            "current_frame": torch.rand(1, 3, 64, 64),
            "prompt_object": torch.rand(1, 4),
            "prompt_target": torch.rand(1, 4),
            "action_history": torch.rand(1, 2, 10),
            "future_actions": torch.rand(1, 4, 10),
        }
        loss = policy.compute_loss(batch)
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_current_frame_only_ignores_task_and_history(self) -> None:
        cfg = VisionTrajPolicyConfig(
            action_horizon=4,
            history_horizon=0,
            use_task_input=False,
            embed_dim=32,
            vision_encoder={
                "type": "timm",
                "model_name": "resnet18",
                "pretrained": False,
                "freeze": True,
                "img_size": 64,
            },
            fusion_layers=1,
            fusion_heads=4,
            head_layers=1,
            head_heads=4,
            head_type="flow_matching",
            num_inference_steps=2,
        )
        policy = VisionTrajPolicy(cfg).eval()
        batch = {
            "first_frame": torch.rand(2, 3, 64, 64),
            "current_frame": torch.rand(2, 3, 64, 64),
            "prompt_object": torch.rand(2, 4),
            "prompt_target": torch.rand(2, 4),
            "action_history": torch.empty(2, 0, 10),
            "future_actions": torch.rand(2, 4, 10),
        }
        changed_task = dict(batch)
        changed_task["first_frame"] = torch.rand_like(batch["first_frame"])
        changed_task["prompt_object"] = torch.rand_like(batch["prompt_object"])
        changed_task["prompt_target"] = torch.rand_like(batch["prompt_target"])
        noise = torch.randn(2, 4, 10)
        prediction = policy.predict_action(batch, initial_noise=noise)
        changed_prediction = policy.predict_action(changed_task, initial_noise=noise)
        self.assertTrue(torch.equal(prediction, changed_prediction))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in policy.task_encoder.fusion.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
