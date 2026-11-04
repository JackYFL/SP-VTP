from pathlib import Path
import subprocess
import sys
import tempfile
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from spvtp.demonstration_processing.pipeline import detect_tags, process


class ProcessingTests(unittest.TestCase):
    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('hydra', 'scipy', 'ffmpeg', 'av', 'colorama')),
                         'optional processing dependencies are not installed')
    def test_sync_output_keeps_raw_frame_alignment(self):
        import cv2
        import numpy as np
        import zarr
        from spvtp.vision_traj.data.dataset import build_episode_index
        from spvtp.vision_traj.runtime import make_dataset
        from tests.helpers import tiny_config
        import json
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            raw = folder / 'raw' / 'episode_1'
            root = zarr.open(str(raw), mode='a')
            times = np.arange(16, dtype=float) / 10
            poses = np.repeat(np.eye(4)[None], 16, axis=0)
            poses[:, 0, 3] = np.arange(16) * .01
            root['camera_0/receive_time'] = times
            root['camera_0/pose'] = poses
            root['camera_1/receive_time'] = times
            writer = cv2.VideoWriter(str(raw / 'camera_1.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (32, 32))
            self.assertTrue(writer.isOpened())
            for _ in times:
                writer.write(np.zeros((32, 32, 3), dtype='uint8'))
            writer.release()
            output = folder / 'data/scene1/task/recording_output_processed'
            process(overrides=[f'input_dir={raw.parent}', f'output_dir={output}', 'stages=[sync_cameras]',
                'sync_cameras.camera_latency=0.1', 'sync_cameras.tracking_latency=0',
                'sync_cameras.T_ee_cam=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]'])
            prepared = zarr.open(str(output / 'episode_1'), mode='a')
            self.assertFalse(prepared['valid_indices'][0])
            self.assertEqual(prepared['pose_interp'].shape, (16, 4, 4))
            self.assertAlmostEqual(float(prepared['pose_interp'][1, 0, 3]), 0)
            prepared['gripper_widths'] = np.full(16, .02, dtype='float32')
            annotations = folder / 'annotations.json'
            annotations.write_text(json.dumps({'scene1/task/episode_1/camera_1.mp4':
                {'0': dict(object=[1,1,10,10], target=[12,12,20,20])}}))
            cfg = tiny_config(folder / 'data', annotations, folder / 'run')
            dataset = make_dataset(cfg, build_episode_index(folder / 'data', annotations))
            future = dataset[0]['future_actions']
            np.testing.assert_allclose(future[:,0], [.01,.02,.03,.04], atol=1e-6)

    def test_legacy_compacted_arrays_align_with_raw_video_indices(self):
        import numpy as np
        from spvtp.common.episode_arrays import align_valid_frames
        valid = np.array([False, True, True, False, True])
        aligned = align_valid_frames(np.array([.01, .02, .03]), valid)
        np.testing.assert_allclose(aligned[valid], [.01, .02, .03])
        self.assertTrue(np.isnan(aligned[~valid]).all())
        with self.assertRaises(ValueError):
            align_valid_frames(np.zeros(2), valid)

    def test_packaged_calibration_and_config_overrides(self):
        module = SimpleNamespace(main=SimpleNamespace(__wrapped__=lambda cfg: cfg))
        with patch('spvtp.demonstration_processing.pipeline.import_module', return_value=module):
            cfg = process(overrides=['input_dir=/tmp/raw', 'stages=[sync_cameras]'])
        self.assertEqual(cfg.input_dir, '/tmp/raw')
        self.assertEqual(list(cfg.stages), ['sync_cameras'])
        self.assertTrue(Path(cfg.detect.aruco_config).is_file())
        self.assertTrue(Path(cfg.detect.oak_intrinsics).is_file())

    def test_tag_detection_uses_active_python_and_surfaces_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            episode = Path(folder) / 'episode_1'
            episode.mkdir()
            (episode / 'camera_1.mp4').touch()
            cfg = OmegaConf.create(dict(num_workers=1, oak_intrinsics='gopro_intrinsics_2_7k.json',
                aruco_config='aruco_config.yaml', ar_detection_vis_dir=folder, visualize=False))
            with patch('spvtp.demonstration_processing.pipeline.subprocess.run') as run:
                detect_tags(lambda _: [episode], cfg, Path(folder) / 'processed')
                self.assertEqual(run.call_args.args[0][:3],
                                 [sys.executable, '-m', 'spvtp.scripts.detect_aruco_oak'])
                self.assertTrue(run.call_args.kwargs['check'])
                run.side_effect = subprocess.CalledProcessError(1, 'detector')
                with self.assertRaises(subprocess.CalledProcessError):
                    detect_tags(lambda _: [episode], cfg, Path(folder) / 'processed')
