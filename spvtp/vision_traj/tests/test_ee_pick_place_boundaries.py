import unittest

import numpy as np

from spvtp.vision_traj.analysis.ee_pick_place_boundaries import (
    detect_pick_place_landmarks,
    detect_position_landmarks,
)


def _synthetic_positions(length: int = 180) -> np.ndarray:
    anchors_t = np.asarray([0, 25, 65, 92, 125, 155, length - 1])
    anchors_y = np.asarray([0.10, 0.08, -0.12, 0.05, 0.09, -0.10, 0.02])
    timeline = np.arange(length)
    height = np.interp(timeline, anchors_t, anchors_y)
    positions = np.zeros((length, 3), dtype=np.float64)
    positions[:, 0] = np.linspace(0.0, 0.6, length)
    positions[:, 1] = height
    positions[:, 2] = 0.03 * np.sin(timeline / 15.0)
    return positions


class PositionBoundaryTest(unittest.TestCase):
    def test_two_contacts_and_apex(self) -> None:
        result = detect_position_landmarks(_synthetic_positions())
        self.assertLess(abs(result.pick_index - 65), 8)
        self.assertLess(abs(result.split_index - 125), 10)
        self.assertLess(abs(result.place_index - 155), 8)
        self.assertLess(result.pick_index, result.split_index)
        self.assertLess(result.split_index, result.place_index)

    def test_nan_samples_are_interpolated(self) -> None:
        positions = _synthetic_positions()
        positions[50:54, 1] = np.nan
        result = detect_position_landmarks(positions)
        self.assertLess(result.pick_index, result.split_index)
        self.assertLess(result.split_index, result.place_index)

    def test_short_episode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 24"):
            detect_position_landmarks(np.zeros((20, 3), dtype=np.float64))

    def test_gripper_transitions_refine_contacts(self) -> None:
        positions = _synthetic_positions()
        gripper = np.interp(
            np.arange(len(positions)),
            [0, 52, 68, 142, 158, len(positions) - 1],
            [0.10, 0.10, 0.0, 0.0, 0.10, 0.10],
        )
        result = detect_pick_place_landmarks(positions, gripper)
        self.assertTrue(result.gripper_used)
        self.assertIsNotNone(result.gripper_close_start)
        self.assertIsNotNone(result.gripper_release_start)
        self.assertLess(abs(result.pick_index - 65), 8)
        self.assertLess(abs(result.place_index - 155), 8)

    def test_invalid_gripper_uses_position_fallback(self) -> None:
        positions = _synthetic_positions()
        result = detect_pick_place_landmarks(positions, np.full(len(positions), 0.1))
        self.assertFalse(result.gripper_used)
        self.assertEqual(result.quality, "low")


if __name__ == "__main__":
    unittest.main()
