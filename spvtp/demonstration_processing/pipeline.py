"""Programmatic adapter around the existing calibrated processing stages."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from importlib.resources import files
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf


def calibration_path(value):
    value = Path(value).expanduser()
    if value.is_absolute() or value.exists():
        return str(value.resolve())
    candidate = Path(__file__).parent / 'calibration' / value.name
    return str(candidate) if candidate.is_file() else str(value)


def detect_tags(demonstration_iterator, cfg, output_dir):
    """Run independent tag-detection jobs and propagate worker failures."""
    def detect(directory):
        directory = Path(directory)
        video = directory / 'camera_1.mp4'
        if not video.is_file():
            raise FileNotFoundError(video)
        output = Path(output_dir) / directory.name / 'tag_detection.pkl'
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', 'spvtp.scripts.detect_aruco_oak',
            '--input', str(video), '--output', str(output),
            '--intrinsics_json', calibration_path(cfg.oak_intrinsics),
            '--aruco_yaml', calibration_path(cfg.aruco_config),
            '--ar_detection_vis_dir', str(Path(cfg.ar_detection_vis_dir).absolute()),
            '--visualize', str(cfg.visualize), '--num_workers', '1']
        subprocess.run(command, check=True)
    with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
        list(executor.map(detect, demonstration_iterator(None)))


def process(config=None, *, backend='iphone-gopro', overrides=()):
    name = 'process_demos_' + backend.replace('-', '_')
    config = config or files('spvtp.demonstration_processing').joinpath('config').joinpath(name + '.yaml')
    cfg = OmegaConf.merge(OmegaConf.load(str(config)), OmegaConf.from_dotlist(list(overrides)))
    if 'detect' in cfg:
        for key in ('aruco_config', 'oak_intrinsics', 'iphone_calibration'):
            if cfg.detect.get(key):
                cfg.detect[key] = calibration_path(cfg.detect[key])
    implementation = 'iphone_pipeline' if backend in ('iphone-gopro', 'iphone-oak') else name
    module = import_module(f'spvtp.demonstration_processing.{implementation}')
    # Bypass Hydra's process-global working-directory and argv machinery.
    return getattr(module.main, '__wrapped__', module.main)(cfg)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Synchronize, calibrate and process recorded episodes')
    parser.add_argument('--backend', choices=['iphone-gopro', 'iphone-oak', 'iphone', 'gopro'], default='iphone-gopro')
    parser.add_argument('--config')
    parser.add_argument('overrides', nargs='*', help='Hydra-style dotted configuration overrides')
    args = parser.parse_args(argv)
    return process(args.config, backend=args.backend, overrides=args.overrides)
