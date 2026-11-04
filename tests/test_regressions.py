"""Regressions for compact-frame caches and the first resumed optimizer update."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import zarr

from tests.helpers import tiny_config, write_dataset
from spvtp.vision_traj.data.build_frame_cache import _build_episode, _cache_paths
from spvtp.vision_traj.runtime import make_dataset, select_episodes


class CacheAlignmentTests(unittest.TestCase):
    def fixture(self, folder, compact=True):
        root, annotations = write_dataset(folder)
        cfg = tiny_config(root, annotations, Path(folder) / 'run')
        cfg.policy.action_horizon, cfg.policy.history_horizon = 2, 1
        episodes = select_episodes(cfg, 'val')[:1]
        group = zarr.open(str(episodes[0].zarr_root), mode='a')
        valid = np.ones(16, dtype=bool)
        valid[[0, 3, 7]] = False
        group['valid_indices'] = valid
        if compact:
            group['pose_interp'] = group['pose_interp'][:][valid]
            group['gripper_widths'] = group['gripper_widths'][:][valid]
        return cfg, episodes, group

    def assert_cache_matches(self, cfg, episodes, cache):
        direct = make_dataset(cfg, episodes)
        cached = make_dataset(cfg, episodes, frame_cache_root=cache)
        self.assertGreater(len(direct), 0)
        self.assertEqual(len(direct), len(cached))
        for index in (0, len(direct) - 1):
            for key, value in direct[index].items():
                self.assertTrue(torch.equal(value, cached[index][key]), (index, key))

    def test_compact_cache_matches_all_prompt_forms_and_strides(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, episodes, _ = self.fixture(folder)
            cache = Path(folder) / 'cache'
            for stride in (1, 2, 3):
                cfg.data.stride = stride
                _build_episode(episodes[0], str(cache), stride, 32, False)
                for prompt in ('none', 'bbox', 'point', 'vision_bbox',
                               'vision_bbox_and_bbox', 'object_only', 'target_only'):
                    with self.subTest(stride=stride, prompt=prompt):
                        cfg.data.prompt_form = prompt
                        self.assert_cache_matches(cfg, episodes, cache)

    def test_legacy_truncated_cache_is_rebuilt_without_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, episodes, _ = self.fixture(folder)
            cache = Path(folder) / 'cache'
            _build_episode(episodes[0], str(cache), 1, 32, False)
            array, metadata_path = _cache_paths(cache, episodes[0])
            # Reproduce a self-consistent old cache that silently omitted tail frames.
            np.save(array, np.load(array)[:-3])
            metadata = json.loads(metadata_path.read_text())
            metadata['n_strided_frames'] -= 3
            metadata.pop('raw_indices_sha256', None)
            metadata_path.write_text(json.dumps(metadata))
            result = _build_episode(episodes[0], str(cache), 1, 32, False)
            self.assertEqual(result['status'], 'built')
            self.assert_cache_matches(cfg, episodes, cache)
            self.assertEqual(_build_episode(episodes[0], str(cache), 1, 32, False)['status'], 'skipped')

    def test_mask_change_with_same_count_and_endpoints_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, episodes, group = self.fixture(folder, compact=False)
            cache = Path(folder) / 'cache'
            _build_episode(episodes[0], str(cache), 1, 32, False)
            valid = group['valid_indices'][:]
            valid[3], valid[4] = True, False
            group['valid_indices'] = valid
            self.assertEqual(_build_episode(episodes[0], str(cache), 1, 32, False)['status'], 'built')
            self.assert_cache_matches(cfg, episodes, cache)

    def test_bad_array_lengths_raise_even_when_cache_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            _, episodes, group = self.fixture(folder)
            cache = Path(folder) / 'cache'
            _build_episode(episodes[0], str(cache), 1, 32, False)
            group['pose_interp'] = group['pose_interp'][:-1]
            with self.assertRaisesRegex(ValueError, 'matches neither raw nor valid'):
                _build_episode(episodes[0], str(cache), 1, 32, False)


class ResumeTests(unittest.TestCase):
    def test_first_resumed_step_updates_weights_and_preserves_schedule(self):
        from spvtp.vision_traj.engine.trainer import train
        with tempfile.TemporaryDirectory() as folder:
            root, annotations = write_dataset(folder)
            cfg = tiny_config(root, annotations, Path(folder) / 'train')
            cfg.train.num_epochs = 3
            cfg.train.scheduler.warmup_steps = 4
            cfg.train.max_steps = 1
            cfg.train.run_final_eval = False
            cfg.train.val_every_epochs = 100
            path = train(cfg)
            before = torch.load(path, map_location='cpu', weights_only=False)
            saved_lr = before['optimizer']['param_groups'][0]['lr']
            self.assertGreater(saved_lr, 0)
            self.assertEqual(saved_lr, before['scheduler']['_last_lr'][0])
            cfg.train.resume = str(path)
            cfg.train.output_dir = str(Path(folder) / 'resume')
            cfg.train.max_steps = 2
            path = train(cfg)
            after = torch.load(path, map_location='cpu', weights_only=False)
            self.assertEqual(after['step'], 2)
            self.assertEqual(after['scheduler']['last_epoch'], 2)
            self.assertEqual(after['optimizer']['param_groups'][0]['lr'], after['scheduler']['_last_lr'][0])
            self.assertAlmostEqual(after['optimizer']['param_groups'][0]['lr'], 2 * saved_lr)
            name = 'flow_head.action_out.weight'
            self.assertGreater(float((after['policy'][name] - before['policy'][name]).abs().max()), 0)
            self.assertTrue(torch.equal(before['policy']['action_mean'], after['policy']['action_mean']))


if __name__ == '__main__':
    unittest.main()
