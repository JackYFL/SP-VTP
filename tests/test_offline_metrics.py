"""Tests for task-oriented offline trajectory metrics."""

from __future__ import annotations

import unittest

import torch

from spvtp.vision_traj.scripts.infer import compute_batch_metrics, rotation_geodesic_degrees
from spvtp.vision_traj.utils.rotation_utils import rotmat_to_rot6d


class OfflineMetricsTest(unittest.TestCase):
    def test_rotation_geodesic_identity_and_quarter_turn(self) -> None:
        identity = torch.eye(3)
        quarter_turn = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        identity_6d = rotmat_to_rot6d(identity)
        quarter_turn_6d = rotmat_to_rot6d(quarter_turn)
        self.assertAlmostEqual(
            float(rotation_geodesic_degrees(identity_6d, identity_6d)),
            0.0,
            places=5,
        )
        self.assertAlmostEqual(
            float(rotation_geodesic_degrees(identity_6d, quarter_turn_6d)),
            90.0,
            places=4,
        )

    def test_gripper_state_metrics_use_configured_threshold(self) -> None:
        pred = torch.zeros(1, 2, 10)
        gt = torch.zeros(1, 2, 10)
        identity_6d = rotmat_to_rot6d(torch.eye(3))
        pred[..., 3:9] = identity_6d
        gt[..., 3:9] = identity_6d
        pred[0, :, 9] = torch.tensor([0.01, 0.05])
        gt[0, :, 9] = torch.tensor([0.02, 0.01])
        metrics = compute_batch_metrics(pred, gt, gripper_state_threshold=0.03)
        self.assertAlmostEqual(float(metrics["grip_state_accuracy"][0]), 0.5)
        self.assertEqual(float(metrics["final_grip_state_correct"][0]), 0.0)


if __name__ == "__main__":
    unittest.main()
