import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.helpers import tiny_config, write_dataset
from spvtp.demonstration_collection.session import RecordingSession
from spvtp.vision_traj.runtime import (compute_action_stats, load_checkpoint, make_dataset,
    make_initial_noise, select_episodes)
from spvtp.vision_traj.evaluation.metrics import compute_batch_metrics
from spvtp.vision_traj.evaluation.runner import evaluate_dataset


class WorkflowTests(unittest.TestCase):
    def test_metrics_known_translation_rotation_and_gripper(self):
        gt = torch.zeros(1, 2, 10)
        gt[..., 3] = gt[..., 7] = 1
        gt[..., 9] = 0.04
        pred = gt.clone()
        pred[0, :, 0] = torch.tensor([0.01, 0.03])
        pred[0, -1, 9] = 0.01
        metrics = compute_batch_metrics(pred, gt)
        self.assertAlmostEqual(float(metrics['pos_l2']), .02, places=6)
        self.assertAlmostEqual(float(metrics['final_displacement_error']), .03, places=6)
        self.assertEqual(float(metrics['success_at_2cm']), 0)
        self.assertEqual(float(metrics['success_at_5cm']), 1)
        self.assertEqual(float(metrics['final_grip_state_correct']), 0)
        self.assertEqual(float(metrics['combined_success_10cm_20deg_grip']), 0)

    def test_noise_independent_of_batching(self):
        together = make_initial_noise([8, 2], 4, 10, 7)
        self.assertTrue(torch.equal(together[1], make_initial_noise([2], 4, 10, 7)[0]))

    def test_console_command_has_success_exit_status(self):
        from spvtp.cli import main
        from unittest.mock import Mock
        implementation = Mock()
        implementation.main.return_value = {'result': 'ok'}
        with patch('spvtp.cli.import_module', return_value=implementation):
            self.assertEqual(main(['eval', '--ckpt', 'example.pt']), 0)

    def test_cached_and_dynamic_prompt_pixels_match_direct_decode(self):
        from dataclasses import replace
        from spvtp.vision_traj.data.build_frame_cache import _build_episode
        with tempfile.TemporaryDirectory() as folder:
            root, annotations = write_dataset(folder)
            cfg = tiny_config(root, annotations, Path(folder) / 'run')
            episodes = select_episodes(cfg, 'val')[:1]
            cache = Path(folder) / 'cache'
            _build_episode(episodes[0], str(cache), 1, 32, False)
            direct = make_dataset(cfg, episodes)[0]
            cached = make_dataset(cfg, episodes, frame_cache_root=cache)[0]
            for key in ('first_frame', 'current_frame', 'future_actions'):
                self.assertTrue(torch.equal(direct[key], cached[key]))
            changed = [replace(episodes[0], target_box=np.array([11, 12, 20, 21], dtype='float32'))]
            direct = make_dataset(cfg, changed)[0]
            dynamic = make_dataset(cfg, changed, frame_cache_root=cache, dynamic_prompt_render=True)[0]
            self.assertTrue(torch.equal(direct['first_frame'], dynamic['first_frame']))

    def test_public_policy_export_uses_canonical_implementation(self):
        from spvtp.vision_traj import VisionTrajPolicy
        implementation = importlib.import_module('spvtp.vision_traj.models.policy')
        self.assertIs(VisionTrajPolicy, implementation.VisionTrajPolicy)

    def test_vla_evaluation_reuses_metrics_and_restores_rng(self):
        from types import SimpleNamespace
        from spvtp.vision_traj.baselines.evaluation import evaluate_vla
        model = torch.nn.Linear(1, 1)
        accelerator = SimpleNamespace(device=torch.device('cpu'), is_main_process=True,
                                      gather_for_metrics=lambda value: value)
        target = torch.zeros(2, 4, 10)
        target[..., 3] = target[..., 7] = 1
        loader = [dict(index=torch.tensor([0, 1]), action=target)]
        state = torch.random.get_rng_state()
        with tempfile.TemporaryDirectory() as output:
            summary = evaluate_vla(accelerator, model, loader, output, 0, None,
                                   lambda batch: batch['action'])
        self.assertEqual(summary['mean_final_displacement_error'], 0)
        self.assertEqual(summary['combined_success_10cm_20deg_grip'], 1)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(model.training)

    def test_cli_help_does_not_load_camera_sdks(self):
        for command in ([], ['collect'], ['process'], ['train'], ['eval'], ['rollout'], ['rollout-suite'], ['vla']):
            result = subprocess.run([sys.executable, '-m', 'spvtp', *command, '--help'],
                cwd='/tmp', capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_actions_and_normalization_do_not_decode_rgb(self):
        with tempfile.TemporaryDirectory() as folder:
            root, annotations = write_dataset(folder)
            cfg = tiny_config(root, annotations, Path(folder) / 'run')
            dataset = make_dataset(cfg, select_episodes(cfg, 'train'))
            sample = dataset[0]
            with patch.object(dataset._cache, 'read_frame', side_effect=AssertionError('RGB decoded')):
                history, future = dataset.action_arrays(0)
                mean, std = compute_action_stats(dataset, 8)
            np.testing.assert_allclose(history, sample['action_history'])
            np.testing.assert_allclose(future, sample['future_actions'])
            self.assertEqual(mean.shape, (10,))
            self.assertTrue(torch.isfinite(std).all())

    def test_invalid_sample_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            root, annotations = write_dataset(folder)
            cfg = tiny_config(root, annotations, Path(folder) / 'run')
            dataset = make_dataset(cfg, select_episodes(cfg, 'train'))
            with patch.object(dataset, '_load_sample', side_effect=OSError('bad video')) as load:
                with self.assertRaises(RuntimeError):
                    dataset[0]
                self.assertEqual(load.call_count, 1)

    def test_train_checkpoint_and_standalone_eval_match(self):
        from spvtp.vision_traj.engine.trainer import train
        from spvtp.vision_traj.scripts.infer import main as evaluate_cli
        with tempfile.TemporaryDirectory() as folder:
            root, annotations = write_dataset(folder)
            output = Path(folder) / 'run'
            cfg = tiny_config(root, annotations, output)
            checkpoint = train(cfg)
            trained = json.loads((output / 'eval_metrics/all/summary.json').read_text())
            model, restored, _ = load_checkpoint(checkpoint)
            model.eval()
            dataset = make_dataset(restored, select_episodes(restored, 'val', output / 'split_manifest.json'))
            summaries, _ = evaluate_dataset(model, dataset, Path(folder) / 'reeval', batch_size=5, save_steps=True)
            for key in ('mean_pos_l2', 'mean_final_displacement_error', 'mean_rotation_geodesic_deg', 'mean_grip_l1'):
                self.assertAlmostEqual(summaries['all'][key], trained[key], places=5)
            self.assertEqual(sum(summaries[s]['samples'] for s in ('scene1','scene2','scene3')), len(dataset))
            self.assertTrue((Path(folder) / 'reeval/all/step_metrics.csv').exists())
            standalone = evaluate_cli(['--ckpt', str(checkpoint), '--device', 'cpu', '--batch-size', '4',
                                       '--output-dir', str(Path(folder) / 'cli')])
            self.assertEqual(standalone['all']['samples'], trained['samples'])
            self.assertAlmostEqual(standalone['all']['mean_pos_l2'], trained['mean_pos_l2'], places=6)


class RecordingTests(unittest.TestCase):
    def test_episode_numbering_does_not_overwrite_gaps(self):
        import zarr
        from spvtp.demonstration_collection.recorder.video import VideoRecorder
        from spvtp.demonstration_collection.camera.camera import FrameData
        with tempfile.TemporaryDirectory() as folder:
            root = zarr.open(folder, mode='a')
            root.create_group('episode_0')
            root.create_group('episode_4')
            recorder = VideoRecorder(record_fps=10, stream_fps=10, video_record_path=folder,
                                     camera_sources=[], frame_data_class=FrameData)
            self.assertEqual(recorder.episode_id, 5)

    def test_state_actions_and_cleanup(self):
        from unittest.mock import Mock
        manager = Mock()
        with RecordingSession(manager) as session:
            session.handle_key('s')
            session.handle_key('w')
            manager.save_recordings.assert_called_once()
            session.handle_key('a')
            self.assertFalse(session.handle_key('q'))
        manager.stop_streaming.assert_called_once()
        self.assertEqual(manager.start_recording.call_count, 2)

    def test_cleanup_on_preview_failure(self):
        from unittest.mock import Mock
        manager = Mock()
        with self.assertRaises(RuntimeError):
            with RecordingSession(manager):
                raise RuntimeError('preview failed')
        manager.stop_streaming.assert_called_once()


if __name__ == '__main__':
    unittest.main()
