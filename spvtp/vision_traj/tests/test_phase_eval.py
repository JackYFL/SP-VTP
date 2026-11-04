"""Tests for execution-phase segmentation."""

from __future__ import annotations

import unittest

import numpy as np

from spvtp.vision_traj.analysis.phase_eval import PHASES, segment_episode_phases


class PhaseSegmentationTest(unittest.TestCase):
    def test_detects_close_transport_and_release(self) -> None:
        n = 100
        gripper = np.r_[
            np.full(30, 0.10),
            np.linspace(0.10, 0.0, 8),
            np.zeros(45),
            np.linspace(0.0, 0.08, 7),
            np.full(10, 0.08),
        ]
        position = np.zeros((n, 3), dtype=np.float64)
        position[:, 0] = np.r_[
            np.linspace(0, 0.4, 30),
            np.linspace(0.4, 0.42, 15),
            np.linspace(0.42, 1.0, 35),
            np.linspace(1.0, 1.03, 20),
        ]
        boundary = segment_episode_phases(gripper, position)
        self.assertEqual(boundary.detection, "gripper_velocity_progress")
        self.assertLess(boundary.pre_grasp_start, boundary.grasp_start)
        self.assertLess(boundary.grasp_start, boundary.transport_start)
        self.assertLess(boundary.transport_start, boundary.placement_start)
        self.assertLess(boundary.placement_start, n)
        observed = {boundary.phase_at(index) for index in range(n)}
        self.assertEqual(observed, set(PHASES))

    def test_flat_gripper_uses_progress_fallback(self) -> None:
        n = 50
        boundary = segment_episode_phases(
            np.full(n, 0.05),
            np.zeros((n, 3)),
        )
        self.assertEqual(boundary.detection, "progress_fallback")
        self.assertEqual(
            {boundary.phase_at(index) for index in range(n)},
            set(PHASES),
        )


if __name__ == "__main__":
    unittest.main()
