"""Shared configuration, data construction, device and checkpoint utilities."""
from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from .data.dataset import (VisionTrajDataset, build_episode_index,
                           filter_episodes_by_scenes, resolve_split_seed,
                           split_episodes, worker_init_fn)


def load_config(path=None, overrides=()):
    path = path or files('spvtp.vision_traj.configs').joinpath('config.yaml')
    return OmegaConf.merge(OmegaConf.load(str(path)), OmegaConf.from_dotlist(list(overrides)))


def select_episodes(cfg, split='val', manifest=None, *, episodes=None):
    if episodes is None:
        episodes = build_episode_index(cfg.data.root, cfg.data.annotations_json)
        if cfg.data.get('scenes'):
            episodes = filter_episodes_by_scenes(episodes, list(cfg.data.scenes))
    if split == 'all':
        return episodes
    if manifest:
        rows = json.loads(Path(manifest).read_text())[f'{split}_episodes']
        indexed = {ep.key.replace('\\', '/'): ep for ep in episodes}
        keys = [row['episode_key'].replace('\\', '/') for row in rows]
        missing = set(keys) - indexed.keys()
        if missing:
            raise ValueError(f'Split manifest contains unavailable episodes: {sorted(missing)[:3]}')
        return [indexed[key] for key in keys]
    train, val = split_episodes(episodes, float(cfg.data.val_ratio), resolve_split_seed(cfg))
    return {'train': train, 'val': val}[split]


def make_dataset(cfg, episodes, **overrides):
    options = dict(action_horizon=cfg.policy.action_horizon,
                   history_horizon=cfg.policy.history_horizon,
                   image_size=cfg.data.image_size, stride=cfg.data.stride,
                   prompt_form=cfg.data.prompt_form,
                   frame_cache_root=cfg.data.get('frame_cache_root'),
                   sample_start_step=cfg.data.get('sample_start_step'))
    options.update(overrides)
    return VisionTrajDataset(episodes, **options)


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {**self.dataset[index], 'sample_index': torch.tensor(index)}


def make_loader(dataset, batch_size, num_workers=0, *, shuffle=False, indexed=False):
    return DataLoader(IndexedDataset(dataset) if indexed else dataset,
                      batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0, worker_init_fn=worker_init_fn,
                      multiprocessing_context='forkserver' if num_workers else None)


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def make_initial_noise(sample_indices, horizon, action_dim, base_seed=0):
    return torch.stack([
        torch.randn(horizon, action_dim, generator=torch.Generator().manual_seed(base_seed + int(i)))
        for i in sample_indices
    ])


def compute_action_stats(dataset, n_samples=256, seed=42):
    if not len(dataset):
        raise ValueError('Cannot normalize actions from an empty training set')
    indices = np.random.RandomState(seed).choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    actions = torch.from_numpy(np.concatenate([dataset.action_arrays(int(i))[1] for i in indices]))
    return actions.mean(0), actions.std(0, correction=int(len(actions) > 1)).clamp_min(1e-4)


def build_policy(cfg, *, from_checkpoint=False):
    from .models.policy import VisionTrajPolicy, VisionTrajPolicyConfig
    options = OmegaConf.to_container(cfg.policy, resolve=True)
    if from_checkpoint:
        # A full policy checkpoint already contains the vision weights.
        for name in ('vision_encoder', 'observation_encoder'):
            if isinstance(options.get(name), dict):
                options[name]['pretrained'] = False
    return VisionTrajPolicy(VisionTrajPolicyConfig(**options))


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    cfg = OmegaConf.create(checkpoint['cfg'])
    policy = build_policy(cfg, from_checkpoint=True)
    policy.load_state_dict(checkpoint['policy'], strict=True)
    return policy.to(device), cfg, checkpoint


def save_checkpoint(path, policy, cfg, **state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.tmp')
    torch.save(dict(policy=policy.state_dict(), cfg=OmegaConf.to_container(cfg, resolve=True),
                    format_version=1, **state), pending)
    pending.replace(path)


def write_split_manifest(path, cfg, train, val):
    if {ep.key for ep in train} & {ep.key for ep in val}:
        raise ValueError('Training and validation episodes overlap')
    def rows(episodes):
        return [dict(episode_key=ep.key, scene=ep.scene, task=ep.task,
                     zarr_root=str(ep.zarr_root), video_path=str(ep.video_path)) for ep in episodes]
    Path(path).write_text(json.dumps(dict(split_unit='task_within_scene', seed=resolve_split_seed(cfg),
        training_seed=int(cfg.train.seed), val_ratio=float(cfg.data.val_ratio),
        train_episodes=rows(train), val_episodes=rows(val)), indent=2))
