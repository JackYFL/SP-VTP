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
