"""Protect the renamed namespace, lazy entry points and subprocess imports."""
from importlib import resources, util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import spvtp
from spvtp.cli import COMMANDS
from spvtp.common.import_umi_source import get_umi_dir, get_umi_subprocess_env


class NamespaceTests(unittest.TestCase):
    def test_runtime_package_contains_no_automated_tests(self):
        root = Path(spvtp.__file__).resolve().parent
        self.assertEqual(list(root.glob('**/tests/test_*.py')), [])

    def test_vision_traj_top_level_contains_only_public_api_and_runtime(self):
        root = Path(spvtp.__file__).resolve().parent / 'vision_traj'
        self.assertEqual({path.name for path in root.glob('*.py')},
                         {'__init__.py', 'runtime.py'})

    def test_removed_flat_modules_are_not_importable(self):
        removed = (
            'build_frame_cache', 'build_language_prompt_map', 'dataset', 'dataset_stats',
            'diffusion_head', 'ee_pick_place_boundaries', 'extract_pi05_siglip',
            'flow_matching_head', 'groot_baseline', 'infer', 'obs_encoder',
            'openvla_oft_baseline', 'phase_eval', 'pi05_baseline', 'pick_place_analysis',
            'policy', 'prompt_interventions', 'rotation_utils', 'smoke_task_encoder_pi05_obs',
            'smolvla_baseline', 'summarize_multiseed', 'summarize_prompt_interventions',
            'summarize_success_thresholds', 'summarize_vla_language_ablation',
            'task_encoder', 'train', 'vis_stitched_traj', 'viz',
        )
        self.assertEqual(len(removed), 28)
        for name in removed:
            with self.subTest(module=name):
                self.assertIsNone(util.find_spec(f'spvtp.vision_traj.{name}'))

    def test_vision_cli_targets_canonical_subpackages(self):
        for module, _, _ in COMMANDS.values():
            if module.startswith('spvtp.vision_traj.'):
                with self.subTest(module=module):
                    self.assertIn(module.split('.')[2], {'scripts', 'data', 'analysis'})

    def test_cli_modules_and_packaged_assets_use_new_namespace(self):
        self.assertEqual(Path(spvtp.__file__).parent.name, 'spvtp')
        for module, _, _ in COMMANDS.values():
            self.assertTrue(module.startswith('spvtp.'), module)
            self.assertIsNotNone(util.find_spec(module), module)
        for name in ('groot', 'pi05', 'smolvla', 'openvla_oft'):
            self.assertIsNotNone(util.find_spec(f'spvtp.vision_traj.baselines.{name}_baseline'))
        self.assertTrue(resources.files('spvtp.vision_traj.configs').joinpath('config.yaml').is_file())
        self.assertTrue(resources.files('spvtp.demonstration_processing').joinpath(
            'calibration/aruco_config.yaml').is_file())

    def test_core_imports_do_not_fall_back_to_legacy_namespace(self):
        code = '''
import sys
class RejectLegacy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'umi_day' or fullname.startswith('umi_day.'):
            raise AssertionError('Unexpected legacy-package dependency: ' + fullname)
sys.meta_path.insert(0, RejectLegacy())
from spvtp.cli import COMMANDS
from spvtp.vision_traj.runtime import load_config
from spvtp.demonstration_processing.pipeline import process
from spvtp.demonstration_collection.session import RecordingSession
assert load_config().data.prompt_form == 'vision_bbox_and_bbox'
'''
        result = subprocess.run([sys.executable, '-c', code], cwd='/tmp',
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_processing_subprocess_paths_preserve_existing_pythonpath(self):
        with patch.dict(os.environ, {'PYTHONPATH': '/tmp/spvtp-extra-dependencies'}):
            environment = get_umi_subprocess_env()
        paths = environment['PYTHONPATH'].split(os.pathsep)
        self.assertIn(str(Path(spvtp.__file__).resolve().parent.parent), paths)
        self.assertIn(get_umi_dir(), paths)
        self.assertIn('/tmp/spvtp-extra-dependencies', paths)
        result = subprocess.run([sys.executable, '-c', 'import spvtp; import umi'],
                                cwd='/tmp', env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
