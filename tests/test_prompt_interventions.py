"""Tests for deterministic causal prompt interventions."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from spvtp.vision_traj.data.dataset import EpisodeRecord
from spvtp.vision_traj.data.prompt_interventions import apply_prompt_intervention


def _episode(key: str, task: str, object_box, target_box) -> EpisodeRecord:
    return EpisodeRecord(
        key=key,
        video_path=Path(f"/{key}.mp4"),
        zarr_root=Path(f"/{key}"),
        object_box=np.asarray(object_box, dtype=np.float32),
        target_box=np.asarray(target_box, dtype=np.float32),
        n_frames=100,
        scene="scene1",
        task=task,
    )


class PromptInterventionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.episodes = [
            _episode("a", "put_fork1_to_bowl1", [10, 10, 20, 20], [50, 50, 70, 70]),
            _episode("b", "put_fork2_to_bowl2", [20, 10, 30, 20], [60, 50, 80, 70]),
            _episode("c", "put_fork3_to_plate1", [30, 10, 40, 20], [70, 50, 90, 70]),
        ]
        self.frame_size = lambda path: (100, 100)

    def test_shuffled_roles_are_wrong_and_deterministic(self) -> None:
        first, first_manifest = apply_prompt_intervention(
            self.episodes,
            "shuffled_object",
            seed=7,
            frame_size_fn=self.frame_size,
        )
        second, second_manifest = apply_prompt_intervention(
            self.episodes,
            "shuffled_object",
            seed=7,
            frame_size_fn=self.frame_size,
        )
        self.assertEqual(first_manifest, second_manifest)
        for original, changed, row in zip(self.episodes, first, first_manifest):
            self.assertFalse(np.array_equal(original.object_box, changed.object_box))
            self.assertTrue(np.array_equal(original.target_box, changed.target_box))
            self.assertNotEqual(row["task"].split("_to_")[0], row["donor_task"].split("_to_")[0])

    def test_shuffled_target_preserves_object(self) -> None:
        changed, manifest = apply_prompt_intervention(
            self.episodes,
            "shuffled_target",
            frame_size_fn=self.frame_size,
        )
        for original, intervened, row in zip(self.episodes, changed, manifest):
            self.assertTrue(np.array_equal(original.object_box, intervened.object_box))
            self.assertFalse(np.array_equal(original.target_box, intervened.target_box))
            self.assertNotEqual(row["task"].split("_to_")[1], row["donor_task"].split("_to_")[1])

    def test_role_swap_exchanges_boxes(self) -> None:
        changed, manifest = apply_prompt_intervention(
            self.episodes,
            "role_swap",
            frame_size_fn=self.frame_size,
        )
        for original, intervened, row in zip(self.episodes, changed, manifest):
            self.assertTrue(np.array_equal(original.object_box, intervened.target_box))
            self.assertTrue(np.array_equal(original.target_box, intervened.object_box))
            self.assertIsNone(row["donor_episode_key"])

    def test_opaque_subscenes_use_different_task_donors(self) -> None:
        episodes = [
            _episode("sub-a", "subscene1", [10, 10, 20, 20], [50, 50, 70, 70]),
            _episode("sub-b", "subscene2", [20, 10, 30, 20], [60, 50, 80, 70]),
        ]
        _, manifest = apply_prompt_intervention(
            episodes,
            "shuffled_target",
            frame_size_fn=self.frame_size,
        )
        for row in manifest:
            self.assertEqual(row["role_source"], "opaque_task_fallback")
            self.assertNotEqual(row["task"], row["donor_task"])

    def test_jitter_is_deterministic_bounded_and_size_preserving(self) -> None:
        first, first_manifest = apply_prompt_intervention(
            self.episodes,
            "jitter_10",
            seed=11,
            frame_size_fn=self.frame_size,
        )
        second, second_manifest = apply_prompt_intervention(
            self.episodes,
            "jitter_10",
            seed=11,
            frame_size_fn=self.frame_size,
        )
        self.assertEqual(first_manifest, second_manifest)
        for original, changed in zip(self.episodes, first):
            np.testing.assert_allclose(
                original.object_box[2:] - original.object_box[:2],
                changed.object_box[2:] - changed.object_box[:2],
                atol=1.0e-5,
            )
            np.testing.assert_allclose(
                original.target_box[2:] - original.target_box[:2],
                changed.target_box[2:] - changed.target_box[:2],
                atol=1.0e-5,
            )
            self.assertTrue(np.all(changed.object_box >= 0.0))
            self.assertTrue(np.all(changed.object_box <= 100.0))
            self.assertTrue(np.all(changed.target_box >= 0.0))
            self.assertTrue(np.all(changed.target_box <= 100.0))

    def test_scale_changes_size_about_fixed_center(self) -> None:
        changed, manifest = apply_prompt_intervention(
            self.episodes,
            "scale_120",
            frame_size_fn=self.frame_size,
        )
        for original, intervened, row in zip(self.episodes, changed, manifest):
            np.testing.assert_allclose(
                original.object_box[:2] + original.object_box[2:],
                intervened.object_box[:2] + intervened.object_box[2:],
                atol=1.0e-5,
            )
            np.testing.assert_allclose(
                intervened.object_box[2:] - intervened.object_box[:2],
                1.2 * (original.object_box[2:] - original.object_box[:2]),
                atol=1.0e-5,
            )
            self.assertEqual(row["box_scale_factor"], 1.2)

    def test_missing_role_preserves_boxes_for_dataset_override(self) -> None:
        changed, manifest = apply_prompt_intervention(
            self.episodes,
            "missing_target",
            frame_size_fn=self.frame_size,
        )
        for original, intervened, row in zip(self.episodes, changed, manifest):
            np.testing.assert_array_equal(original.object_box, intervened.object_box)
            np.testing.assert_array_equal(original.target_box, intervened.target_box)
            self.assertEqual(row["missing_role"], "target")


if __name__ == "__main__":
    unittest.main()
