"""Small real video/Zarr fixtures shared by end-to-end migration tests."""
import json
from pathlib import Path

import cv2
import numpy as np
import zarr


def write_dataset(folder):
    root = Path(folder) / 'data'
    annotations = {}
    for scene in ('scene1', 'scene2', 'scene3'):
        for task in ('put_fork1_to_plate1', 'put_fork2_to_plate2'):
            episode = root / scene / task / 'recording_output_processed' / 'episode_1'
            episode.mkdir(parents=True)
            writer = cv2.VideoWriter(str(episode / 'camera_1.mp4'),
                                     cv2.VideoWriter_fourcc(*'mp4v'), 10, (32, 32))
            if not writer.isOpened():
                raise RuntimeError('OpenCV mp4v encoder unavailable')
            for index in range(16):
                writer.write(np.full((32, 32, 3), 20 + index * 8, dtype=np.uint8))
            writer.release()
            group = zarr.open(str(episode), mode='a')
            pose = np.repeat(np.eye(4)[None], 16, axis=0)
            pose[:, 0, 3] = np.arange(16) * 0.01
            group['pose_interp'] = pose
            group['valid_indices'] = np.ones(16, dtype=bool)
            group['gripper_widths'] = np.linspace(0.01, 0.07, 16).astype('float32')
            key = f'{scene}/{task}/episode_1/camera_1.mp4'
            annotations[key] = {'0': dict(object=[2, 2, 10, 10], target=[18, 18, 29, 29])}
    path = Path(folder) / 'annotations.json'
    path.write_text(json.dumps(annotations))
    return root, path


def tiny_config(root, annotations, output):
    from spvtp.vision_traj.runtime import load_config
    return load_config(overrides=[f'data.root={root}', f'data.annotations_json={annotations}',
        'data.image_size=32', 'data.stride=1', 'data.num_workers=0',
        'policy.action_horizon=4', 'policy.history_horizon=2', 'policy.embed_dim=32',
        'policy.fusion_layers=1', 'policy.fusion_heads=4', 'policy.head_layers=1',
        'policy.head_heads=4', 'policy.num_inference_steps=2',
        'policy.vision_encoder.model_name=vit_tiny_patch16_224',
        'policy.vision_encoder.pretrained=false', 'policy.vision_encoder.img_size=32',
        f'train.output_dir={output}', 'train.device=cpu', 'train.mixed_precision=no',
        'train.batch_size=4', 'train.grad_accum_steps=1', 'train.num_epochs=1',
        'train.max_steps=2', 'train.val_every_epochs=1', 'train.val_max_batches=1',
        'train.save_every_epochs=10', 'train.stats_samples=8', 'train.log_every=1'])
