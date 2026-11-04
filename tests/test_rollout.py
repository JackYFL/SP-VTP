"""Analytic, no-download checks for history feedback, SE(3) and complete tails."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import zarr

from tests.helpers import tiny_config, write_dataset
from spvtp.vision_traj.evaluation.rollout import (ROLLOUT_MODES, evaluate_rollout,
                                               noise_index, predicted_history)
from spvtp.vision_traj.runtime import make_dataset, select_episodes


class ConstantMotion(torch.nn.Module):
    def __init__(self, horizon=4, speed=.01):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(0.))
        self.horizon, self.speed = horizon, speed
        self.histories, self.noises = [], []

    def predict_action(self, batch, initial_noise):
        assert 'future_actions' not in batch
        self.histories.extend(batch['action_history'].clone().cpu())
        self.noises.extend(initial_noise.clone())
        pred = torch.zeros(len(batch['current_frame']), self.horizon, 10,
                           device=self.parameter.device)
        pred[..., 0] = self.speed * torch.arange(1, self.horizon+1)
        pred[..., 3] = pred[..., 7] = 1
        pred[..., 9] = .04
        return pred


class RolloutTests(unittest.TestCase):
    def fixture(self, folder, episodes=1, rotated=False):
        root, annotations = write_dataset(folder)
        cfg = tiny_config(root, annotations, Path(folder) / 'unused')
        selected = select_episodes(cfg, 'val')[:episodes]
        for episode in selected:
            group = zarr.open(str(episode.zarr_root), mode='a')
            group['gripper_widths'] = np.full(16, .04, dtype='float32')
            if rotated:
                transform = np.array([[0,-1,0,.3],[1,0,0,-.2],[0,0,1,.1],[0,0,0,1.]])
                group['pose_interp'] = transform @ group['pose_interp'][:]
        return make_dataset(cfg, selected)

    def test_tail_and_current_relative_waypoints_cover_every_future_step(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = self.fixture(folder)
            for mode in ROLLOUT_MODES:
                with self.subTest(mode=mode):
                    model = ConstantMotion()
                    summaries, records = evaluate_rollout(model, dataset, mode=mode)
                    self.assertTrue(model.training)
                    self.assertEqual([r['valid_horizon'] for r in records['chunks']], [4,4,4,1])
                    self.assertEqual([r['future_step'] for r in records['steps']], list(range(3,16)))
                    self.assertEqual(records['steps'][-1]['is_episode_endpoint'], 1)
                    self.assertLess(summaries['all']['mean_final_displacement_error'], 1e-6)
                    self.assertEqual(summaries['all']['prediction_steps'], 13)
                    self.assertEqual(summaries['all']['partial_horizon_chunks'], 1)

    def test_bias_accumulates_without_gt_origin_resets(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = self.fixture(folder)
            ends = {}
            for mode in ROLLOUT_MODES:
                summary, _ = evaluate_rollout(ConstantMotion(speed=.02), dataset, mode=mode)
                ends[mode] = summary['all']['mean_final_displacement_error']
            self.assertAlmostEqual(ends['gt_history'], .01, places=6)
            self.assertAlmostEqual(ends['history_feedback'], .01, places=6)
            self.assertAlmostEqual(ends['cumulative'], .13, places=6)

    def test_feedback_history_reframed_and_excludes_current_state(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = self.fixture(folder)
            expected = {'gt_history': [-.02,-.01], 'history_feedback': [0,.02],
                        'cumulative': [-.04,-.02]}
            first = None
            for mode in ROLLOUT_MODES:
                model = ConstantMotion(speed=.02)
                evaluate_rollout(model, dataset, mode=mode)
                torch.testing.assert_close(model.histories[1][:,0], torch.tensor(expected[mode]), atol=1e-6, rtol=0)
                if first is None:
                    first = model.histories[0]
                else:
                    self.assertTrue(torch.equal(first, model.histories[0]))

    def test_rotated_world_frame_uses_right_composition(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = self.fixture(folder, rotated=True)
            for mode in ROLLOUT_MODES:
                summary, _ = evaluate_rollout(ConstantMotion(), dataset, mode=mode)
                self.assertLess(summary['all']['mean_pos_l2'], 1e-6)

    def test_history_rotation_and_absolute_gripper_width(self):
        poses = torch.eye(4,dtype=torch.float64).repeat(3,1,1)
        poses[:,:3,:3] = torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]],dtype=torch.float64)
        poses[:,0,3] = 1
        poses[:,1,3] = torch.tensor([1.,1.5,2.])
        widths = torch.tensor([.02,.05,.08],dtype=torch.float64)
        history = predicted_history(poses,widths,2,2,poses[2])
        torch.testing.assert_close(history[:,:3],torch.tensor([[-1.,0,0],[-.5,0,0]]))
        torch.testing.assert_close(history[:,9],torch.tensor([.02,.05]))
        torch.testing.assert_close(history[:,3:9],torch.tensor([[1.,0,0,0,1,0]]).repeat(2,1))

    def test_exact_full_chunk_ending_has_no_dropped_endpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = self.fixture(folder)
            dataset.sample_start_step = 3
            summary, records = evaluate_rollout(ConstantMotion(),dataset,mode='cumulative')
            self.assertEqual([r['valid_horizon'] for r in records['chunks']],[4,4,4])
            self.assertEqual(summary['all']['prediction_steps'],12)
            self.assertEqual(summary['all']['partial_horizon_chunks'],0)
            self.assertEqual(records['steps'][-1]['future_step'],15)

    def test_noise_paired_across_modes_and_episode_batching(self):
        self.assertEqual(noise_index('scene\\task\\ep', 4), noise_index('scene/task/ep', 4))
        with tempfile.TemporaryDirectory() as folder:
            dataset = self.fixture(folder, episodes=3)
            first, second = ConstantMotion(), ConstantMotion()
            evaluate_rollout(first, dataset, mode='gt_history', batch_size=1)
            evaluate_rollout(second, dataset, mode='cumulative', batch_size=3)
            self.assertTrue(torch.equal(torch.stack(first.noises), torch.stack(second.noises)))

    def test_existing_sample_api_unchanged_and_tail_is_opt_in(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = self.fixture(folder)
            ei, step = dataset._samples[0]
            for key, value in dataset[0].items():
                self.assertTrue(torch.equal(value, dataset.sample_at(ei,step)[key]))
            with self.assertRaises(IndexError):
                dataset.sample_at(0,14)
            self.assertEqual(dataset.sample_at(0,14,allow_partial=True)['future_actions'].shape, (1,10))

    def test_output_aggregation_and_empty_history(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = self.fixture(folder, episodes=3)
            dataset.history_horizon = 0
            output = Path(folder) / 'rollout'
            summary, records = evaluate_rollout(ConstantMotion(), dataset, output, mode='cumulative')
            self.assertEqual(summary['all']['episodes'], 3)
            self.assertEqual(sum(summary[s]['prediction_steps'] for s in ('scene1','scene2','scene3')), 39)
            self.assertTrue((output / 'all/drift_curve.csv').is_file())
            self.assertTrue((output / 'all/phase_metrics.csv').is_file())
            self.assertTrue((output / 'all/full_chunks_summary.json').is_file())
            self.assertTrue((output / 'phase_boundaries.json').is_file())


if __name__ == '__main__':
    unittest.main()
