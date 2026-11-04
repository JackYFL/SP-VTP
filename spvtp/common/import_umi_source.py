import os
import sys
from pathlib import Path

def get_umi_dir():
    # SP-VTP vendors the UMI utilities needed by the data-processing pipeline.
    umi_dir = os.path.join(os.path.dirname(__file__), '..', 'demonstration_processing')
    umi_dir = Path(umi_dir).absolute().resolve().as_posix()
    return umi_dir

def get_umi_subprocess_env():
    """When you run a subprocess to run script in UMI dir you need to provide path to UMI codebase"""
    paths = [str(Path(__file__).resolve().parents[2]), get_umi_dir(), os.environ.get('PYTHONPATH', '')]
    return {**os.environ, 'PYTHONPATH': os.pathsep.join(path for path in paths if path)}
