"""Dataset for vision-conditioned trajectory prediction (pick-and-place with fork).

Each training sample is constructed from a single (episode, current_step) pair:

  inputs:
    - first_frame      : first-frame RGB (fixed per episode, serves as task spec)
    - prompt           : object / target bbox (or point) on the first frame
    - current_frame    : RGB at the current step
    - action_history   : last k relative actions (pose + gripper)
  target:
    - future_actions   : next H relative actions (pose + gripper)

Actions are expressed in the *current* camera frame:
    a_t[h] = T_{cam_t}^{-1} @ T_{cam_{t+h}}
flattened to [dx, dy, dz, rot6d(6), gripper(1)] = 10 dims per step.
"""

from __future__ import annotations

import json
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
from spvtp.common.episode_arrays import align_valid_frames

# Prevent OpenCV's internal thread pool from deadlocking with DataLoader
# workers after fork. Must be called before any cv2 call in each process.
cv2.setNumThreads(0)


def worker_init_fn(worker_id: int) -> None:  # noqa: ARG001
    """Pass to ``DataLoader(worker_init_fn=...)`` so every worker disables
    OpenCV threading. Also re-seeds numpy per worker."""
    cv2.setNumThreads(0)
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)

from spvtp.vision_traj.utils.rotation_utils import (
    decompose_se3,
    rotmat_to_rot6d,
    se3_relative,
)


# ---------------------------------------------------------------------------
# Episode index


@dataclass
class EpisodeRecord:
    """Metadata for a single episode — enough to open its data lazily."""

    key: str                  # original annotation key (windows-style path)
    video_path: Path          # absolute path to camera_1.mp4
    zarr_root: Path           # absolute path to episode directory (a zarr group)
    object_box: np.ndarray    # (4,) x1,y1,x2,y2 in first-frame image pixels
    target_box: np.ndarray    # (4,)
    n_frames: int             # number of valid frames (after masking)
    scene: str = ""           # scene name parsed from annotation key
    task: str = ""            # task name parsed from annotation key


def _canonicalize_annotation_key(key: str) -> Tuple[str, str, str]:
    """Turn 'scene1\\put_fork1_to_plate1\\episode_0\\camera_1.mp4' -> parts."""
    parts = key.replace("\\", "/").split("/")
    # [scene, task, episode, camera_1.mp4]
    assert len(parts) >= 4, f"unexpected key format: {key}"
    return parts[0], parts[1], parts[2]


def build_episode_index(
    data_root: str | os.PathLike,
    annotations_json: str | os.PathLike,
) -> List[EpisodeRecord]:
    """Scan annotations + disk, return a list of usable episodes.

    Expected layout (see repo README):
        <data_root>/<scene>/<task>/recording_output_processed/<episode>/
            camera_1.mp4
            pose_interp  (zarr)
            gripper_widths  (zarr)
            valid_indices  (zarr)
    """
    data_root = Path(data_root)
    with open(annotations_json, "r") as f:
        anns: Dict[str, dict] = json.load(f)

    records: List[EpisodeRecord] = []
    missing = 0
    for key, ann in anns.items():
        scene, task, episode = _canonicalize_annotation_key(key)
        ep_dir = data_root / scene / task / "recording_output_processed" / episode
        video_path = ep_dir / "camera_1.mp4"
        if not video_path.exists():
            missing += 1
            continue
        first = ann.get("0")
        if first is None or "object" not in first or "target" not in first:
            continue

        try:
            root = zarr.open(str(ep_dir), mode="r")
            keys = set(root.keys()) if hasattr(root, "keys") else set()
            required = {"valid_indices", "pose_interp", "gripper_widths"}
            if not required.issubset(keys):
                missing += 1
                continue
            valid = np.asarray(root["valid_indices"][:], dtype=bool)
        except Exception:
            continue
        n_valid = int(valid.sum())
        if n_valid < 8:
            continue

        records.append(
            EpisodeRecord(
                key=key,
                video_path=video_path,
                zarr_root=ep_dir,
                object_box=np.asarray(first["object"], dtype=np.float32),
                target_box=np.asarray(first["target"], dtype=np.float32),
                n_frames=n_valid,
                scene=scene,
                task=task,
            )
        )

    if missing:
        print(f"[build_episode_index] skipped {missing} episodes with missing videos")
    return records


def filter_episodes_by_scenes(
    episodes: Sequence[EpisodeRecord],
    scenes: Optional[Iterable[str]],
) -> List[EpisodeRecord]:
    """Return only episodes whose ``scene`` is in ``scenes``.

    ``None`` (or empty) means *no filter* — return everything. Unknown scene
    names raise so typos don't silently produce an empty dataset.
    """
    if not scenes:
        return list(episodes)
    wanted = {str(s) for s in scenes}
    available = {ep.scene for ep in episodes}
    unknown = wanted - available
    if unknown:
        raise ValueError(
            f"scenes filter contains unknown scenes {sorted(unknown)}; "
            f"available: {sorted(available)}"
        )
    return [ep for ep in episodes if ep.scene in wanted]


def resolve_split_seed(cfg) -> int:
    """Return the data-split seed, falling back for legacy checkpoints.

    Older configs used ``train.seed`` for both dataset splitting and model
    randomness. New multi-seed experiments set ``data.split_seed`` so the
    validation episodes stay fixed while initialization and sampling vary.
    """
    data_cfg = getattr(cfg, "data", None)
    if data_cfg is not None:
        split_seed = (
            data_cfg.get("split_seed", None)
            if hasattr(data_cfg, "get")
            else getattr(data_cfg, "split_seed", None)
        )
        if split_seed is not None:
            return int(split_seed)
    return int(cfg.train.seed)


def split_episodes(
    episodes: Sequence[EpisodeRecord],
    val_ratio: float,
    seed: int,
) -> Tuple[List[EpisodeRecord], List[EpisodeRecord]]:
    """Split episodes by *task within each scene*.

    For every scene, group its episodes by task, deterministically shuffle the
    task names, and route ``round(n_tasks * val_ratio)`` (clamped to leave at
    least one train task) tasks to val. All episodes of a val task go to val;
    all episodes of a train task go to train. Scenes with only one task keep
    that task in train (no val for that scene).
    """
    by_scene: Dict[str, Dict[str, list]] = {}
    for ep in episodes:
        by_scene.setdefault(ep.scene, {}).setdefault(ep.task, []).append(ep)

    train: List[EpisodeRecord] = []
    val: List[EpisodeRecord] = []
    val_task_log: List[Tuple[str, str]] = []
    for scene in sorted(by_scene):
        tasks = sorted(by_scene[scene])
        rng = random.Random(f"{seed}:{scene}")
        rng.shuffle(tasks)
        n_tasks = len(tasks)
        if n_tasks <= 1:
            n_val = 0
        else:
            n_val = max(1, int(round(n_tasks * val_ratio)))
            n_val = min(n_val, n_tasks - 1)
        val_tasks = set(tasks[:n_val])
        for task in tasks:
            bucket = val if task in val_tasks else train
            bucket.extend(by_scene[scene][task])
            if task in val_tasks:
                val_task_log.append((scene, task))

    if val_task_log:
        preview = ", ".join(f"{s}/{t}" for s, t in val_task_log[:6])
        more = f" (+{len(val_task_log) - 6} more)" if len(val_task_log) > 6 else ""
        print(f"[split_episodes] val tasks ({len(val_task_log)}): {preview}{more}")
    return train, val


# ---------------------------------------------------------------------------
# Video cache (per-worker lazy VideoCapture)


class _VideoCache:
    """Per-process cache of cv2.VideoCapture + per-episode first frame.

    cv2.VideoCapture is not fork-safe; each dataloader worker gets its own cache
    via ``worker_init_fn`` or on-demand creation.
    """

    def __init__(self, max_open: int = 32):
        self.max_open = max_open
        self._cap: Dict[str, cv2.VideoCapture] = {}
        self._first_frame: Dict[str, np.ndarray] = {}

    def _cap_for(self, path: str) -> cv2.VideoCapture:
        cap = self._cap.get(path)
        if cap is None:
            if len(self._cap) >= self.max_open:
                k, old = next(iter(self._cap.items()))
                old.release()
                self._cap.pop(k)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise IOError(f"cannot open video {path}")
            self._cap[path] = cap
        return cap

    def read_frame(self, path: str, index: int) -> np.ndarray:
        cap = self._cap_for(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            raise IOError(f"failed to read frame {index} from {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def first_frame(self, path: str) -> np.ndarray:
        if path not in self._first_frame:
            self._first_frame[path] = self.read_frame(path, 0)
        return self._first_frame[path]


class _FrameArrayCache:
    """Per-worker LRU of memory-mapped, predecoded episode frame arrays."""

    def __init__(self, max_open: int = 64):
        self.max_open = max_open
        self._arrays: OrderedDict[str, np.ndarray] = OrderedDict()
        self._metadata: Dict[str, dict] = {}

    def _array_for(self, path: str) -> np.ndarray:
        array = self._arrays.pop(path, None)
        if array is None:
            if len(self._arrays) >= self.max_open:
                self._arrays.popitem(last=False)
            array = np.load(path, mmap_mode="r")
        self._arrays[path] = array
        return array

    def metadata(self, path: str) -> dict:
        metadata = self._metadata.get(path)
        if metadata is None:
            with open(str(Path(path).with_suffix(".json")), "r") as stream:
                metadata = json.load(stream)
            self._metadata[path] = metadata
        return metadata

    def read(self, path: str, index: int) -> np.ndarray:
        # Copy the mmap slice so torch receives writable, contiguous memory.
        return np.array(self._array_for(path)[index], copy=True)


# ---------------------------------------------------------------------------
# Dataset


class VisionTrajDataset(Dataset):
    """Per-frame samples for diffusion-policy-style training.

    Args:
        episodes: list of :class:`EpisodeRecord` produced by :func:`build_episode_index`.
        action_horizon: number of future action steps H (e.g. 16).
        history_horizon: number of past action steps k (e.g. 4); set 0 to disable.
        image_size: output square crop size for both first and current frames.
        stride: subsampling stride on raw video/pose (original 30fps → 10fps uses 3).
        prompt_form: 'bbox', 'point', 'none', 'vision_bbox',
            'vision_bbox_and_bbox', 'object_only', or 'target_only'.  The
            object/target-only modes use both the rendered bbox and coordinate
            bbox for the retained role, matching the default SPOT prompt path.
    """

    def __init__(
        self,
        episodes: List[EpisodeRecord],
        action_horizon: int = 16,
        history_horizon: int = 4,
        image_size: int = 224,
        stride: int = 3,
        prompt_form: str = "bbox",
        frame_cache_root: Optional[str | os.PathLike] = None,
        dynamic_prompt_render: bool = False,
        sample_start_step: Optional[int] = None,
    ) -> None:
        super().__init__()
        valid_prompt_forms = (
            "bbox",
            "point",
            "none",
            "vision_bbox",
            "vision_bbox_and_bbox",
            "object_only",
            "target_only",
        )
        if prompt_form not in valid_prompt_forms:
            raise ValueError(
                f"unknown prompt_form: {prompt_form!r} "
                f"(expected one of {valid_prompt_forms})"
            )
        self.episodes = episodes
        self.action_horizon = action_horizon
        self.history_horizon = history_horizon
        self.image_size = image_size
        self.stride = stride
        self.prompt_form = prompt_form
        self.frame_cache_root = (
            Path(frame_cache_root) if frame_cache_root not in (None, "") else None
        )
        self.dynamic_prompt_render = bool(dynamic_prompt_render)
        self._dynamic_first_frames: Dict[str, torch.Tensor] = {}
        self.sample_start_step = (
            self.history_horizon
            if sample_start_step is None
            else int(sample_start_step)
        )
        if self.sample_start_step < self.history_horizon:
            raise ValueError(
                "sample_start_step must be at least history_horizon "
                f"({self.sample_start_step} < {self.history_horizon})"
            )

        # build a flat (episode_idx, current_step) index. Each step is measured
        # in *strided* units; raw video index = step * stride among valid frames.
        self._samples: List[Tuple[int, int]] = []
        self._poses: List[np.ndarray] = []
        self._grippers: List[np.ndarray] = []
        self._valid_raw_idx: List[np.ndarray] = []
        self._frame_hw: List[Tuple[int, int]] = []
        self._frame_cache_paths: List[Optional[Path]] = []
        self.episodes = []  # filtered list of episodes that loaded successfully
        dropped = 0
        for ep in episodes:
            try:
                root = zarr.open(str(ep.zarr_root), mode="r")
                valid = np.asarray(root["valid_indices"][:], dtype=bool)
                pose = np.asarray(root["pose_interp"][:], dtype=np.float64)
                gripper = np.asarray(root["gripper_widths"][:], dtype=np.float32)
            except (KeyError, FileNotFoundError, OSError):
                dropped += 1
                continue

            pose = align_valid_frames(pose, valid)
            gripper = align_valid_frames(gripper, valid)

            # clip all arrays to the common length so that fancy indexing
            # by valid positions is always safe.
            T = min(len(valid), len(pose), len(gripper))
            if T < self.action_horizon + self.history_horizon + 2:
                dropped += 1
                continue
            valid = valid[:T]
            pose = pose[:T]
            gripper = gripper[:T]

            valid_idx = np.where(valid)[0]
            valid_idx = valid_idx[:: self.stride]
            if len(valid_idx) < self.action_horizon + self.history_horizon + 2:
                dropped += 1
                continue

            ei = len(self.episodes)
            self.episodes.append(ep)
            self._poses.append(pose)
            self._grippers.append(gripper)
            self._valid_raw_idx.append(valid_idx)
            self._frame_hw.append((0, 0))  # filled on first access
            if self.frame_cache_root is None:
                self._frame_cache_paths.append(None)
            else:
                cache_path = (
                    self.frame_cache_root
                    / ep.scene
                    / ep.task
                    / f"{ep.zarr_root.name}.npy"
                )
                metadata_path = cache_path.with_suffix(".json")
                if not cache_path.is_file() or not metadata_path.is_file():
                    raise FileNotFoundError(
                        f"missing predecoded frame cache for {ep.key}: {cache_path}"
                    )
                self._frame_cache_paths.append(cache_path)

            # current-step range: any step that has both history+future covered
            hi_lo = self.sample_start_step
            hi_hi = len(valid_idx) - self.action_horizon - 1
            for step in range(hi_lo, hi_hi + 1):
                self._samples.append((ei, step))

        if dropped:
            print(f"[VisionTrajDataset] dropped {dropped} episodes (missing keys / too short)")
        self._cache = _VideoCache()
        self._frame_array_cache = _FrameArrayCache()

    # ------------------------------------------------------------------
    # cv2.VideoCapture is not picklable. With forkserver/spawn, the dataset
    # is pickled and shipped to worker processes; strip the cache on dump and
    # recreate it fresh in each worker on load.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        state["_frame_array_cache"] = None
        # Each worker lazily renders its own compact 224x224 task-frame cache.
        # Do not duplicate any entries materialized in the parent process.
        state["_dynamic_first_frames"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache = _VideoCache()
        self._frame_array_cache = _FrameArrayCache()
        self._dynamic_first_frames = {}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    def _preprocess_frame(
        self, frame: np.ndarray, box_scale: Tuple[float, float]
    ) -> torch.Tensor:
        """Resize to (image_size, image_size). Returns (3, H, W) float tensor in [0,1]."""
        H, W = frame.shape[:2]
        frame = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        return frame

    def _scale_box(
        self, box: np.ndarray, src_hw: Tuple[int, int]
    ) -> np.ndarray:
        H, W = src_hw
        sx = self.image_size / W
        sy = self.image_size / H
        x1, y1, x2, y2 = box
        return np.asarray(
            [x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32
        )

    def _draw_prompt_boxes(self, frame, object_box=None, target_box=None):
        from .images import draw_prompt_boxes
        return draw_prompt_boxes(frame, object_box, target_box)

    def _dynamic_prompt_frame(self, ep: EpisodeRecord) -> torch.Tensor:
        """Render intervened boxes from the source-resolution first frame.

        The frame-array cache contains task images pre-rendered with the
        annotation's original boxes.  Reusing those images for shuffled,
        jittered, scaled, or role-swapped boxes would leave the visual cue
        unchanged while altering only the coordinate token.  Decode frame 0
        once per episode/worker for those interventions, render at the source
        resolution, and retain only the compact preprocessed tensor.
        """
        cached = self._dynamic_first_frames.get(ep.key)
        if cached is not None:
            return cached
        first = self._cache.read_frame(str(ep.video_path), 0)
        if self.prompt_form in ("vision_bbox", "vision_bbox_and_bbox"):
            first = self._draw_prompt_boxes(
                first, object_box=ep.object_box, target_box=ep.target_box
            )
        elif self.prompt_form == "object_only":
            first = self._draw_prompt_boxes(first, object_box=ep.object_box)
        elif self.prompt_form == "target_only":
            first = self._draw_prompt_boxes(first, target_box=ep.target_box)
        rendered = self._preprocess_frame(first, (1.0, 1.0))
        self._dynamic_first_frames[ep.key] = rendered
        return rendered

    def _relative_action(
        self, T_cur: np.ndarray, T_future: np.ndarray
    ) -> np.ndarray:
        """(4,4) current, (H,4,4) future -> (H, 9) [dpos(3), rot6d(6)]."""
        # Compose in numpy (SE(3) inverse is just transpose of R + translation)
        R_cur = T_cur[:3, :3]
        t_cur = T_cur[:3, 3]
        R_cur_inv = R_cur.T
        t_cur_inv = -R_cur_inv @ t_cur
        T_cur_inv = np.eye(4, dtype=np.float64)
        T_cur_inv[:3, :3] = R_cur_inv
        T_cur_inv[:3, 3] = t_cur_inv
        rel = T_cur_inv @ T_future  # (H, 4, 4) via broadcasting
        pos = rel[:, :3, 3]
        R = rel[:, :3, :3]
        rot6d = np.concatenate([R[:, :, 0], R[:, :, 1]], axis=-1)
        return np.concatenate([pos, rot6d], axis=-1).astype(np.float32)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        try:
            return self._load_sample(index)
        except Exception as error:
            raise RuntimeError(f'Failed to load trajectory sample {index}') from error

    def action_arrays(self, index: int):
        """Return history and targets without decoding video."""
        ei, step = self._samples[index]
        return self.actions_at(ei, step)

    def actions_at(self, ei: int, step: int):
        """Recorded relative actions at any valid current step, including a short tail."""
        raw = self._valid_raw_idx[ei]
        if not self.history_horizon <= step < len(raw) - 1:
            raise IndexError(f'Invalid current step {step} for episode {ei}')
        current = self._poses[ei][raw[step]]
        def actions(indices):
            relative = self._relative_action(current, self._poses[ei][indices])
            grip = self._grippers[ei][indices].astype(np.float32).reshape(-1, 1)
            return np.concatenate([relative, grip], axis=-1)
        history = (actions(raw[step-self.history_horizon:step]) if self.history_horizon
                   else np.zeros((0, 10), dtype=np.float32))
        return history, actions(raw[step+1:step+1+self.action_horizon])

    def _load_sample(self, index: int) -> Dict[str, torch.Tensor]:
        ei, step = self._samples[index]
        return self.sample_at(ei, step)

    def sample_at(self, ei: int, step: int, *, allow_partial: bool = False) -> Dict[str, torch.Tensor]:
        """Load an explicit episode timestep; partial targets are opt-in for rollouts."""
        ep = self.episodes[ei]
        valid_idx = self._valid_raw_idx[ei]
        pose = self._poses[ei]
        gripper = self._grippers[ei]
        is_final_chunk = step == len(valid_idx) - self.action_horizon - 1

        # raw frame indices
        cur_raw = int(valid_idx[step])
        fut_raw = valid_idx[step + 1 : step + 1 + self.action_horizon]
        if not len(fut_raw) or (not allow_partial and len(fut_raw) != self.action_horizon):
            raise IndexError(f'Incomplete future window at episode {ei}, step {step}')

        cache_path = self._frame_cache_paths[ei]
        if cache_path is None:
            first = self._cache.first_frame(str(ep.video_path))
            current = self._cache.read_frame(str(ep.video_path), cur_raw)
            H, W = first.shape[:2]

            if self.prompt_form in ("vision_bbox", "vision_bbox_and_bbox"):
                first_for_model = self._draw_prompt_boxes(
                    first, object_box=ep.object_box, target_box=ep.target_box
                )
            elif self.prompt_form == "object_only":
                first_for_model = self._draw_prompt_boxes(first, object_box=ep.object_box)
            elif self.prompt_form == "target_only":
                first_for_model = self._draw_prompt_boxes(first, target_box=ep.target_box)
            else:
                first_for_model = first
            first_t = self._preprocess_frame(first_for_model, (1.0, 1.0))
            current_t = self._preprocess_frame(current, (1.0, 1.0))
        else:
            cache_key = str(cache_path)
            metadata = self._frame_array_cache.metadata(cache_key)
            if int(metadata["stride"]) != self.stride:
                raise ValueError(f"cache stride mismatch for {ep.key}")
            if int(metadata["image_size"]) != self.image_size:
                raise ValueError(f"cache image-size mismatch for {ep.key}")
            if int(metadata["n_strided_frames"]) != len(valid_idx):
                raise ValueError(f"cache frame-count mismatch for {ep.key}")
            H, W = (int(value) for value in metadata["source_hw"])
            if self.dynamic_prompt_render:
                first_t = self._dynamic_prompt_frame(ep)
            else:
                if self.prompt_form in ("vision_bbox", "vision_bbox_and_bbox"):
                    task_index = 1
                elif self.prompt_form == "object_only":
                    task_index = 2
                elif self.prompt_form == "target_only":
                    task_index = 3
                else:
                    task_index = 0
                first_for_model = self._frame_array_cache.read(cache_key, task_index)
                first_t = self._preprocess_frame(first_for_model, (1.0, 1.0))
            current = self._frame_array_cache.read(cache_key, 4 + step)
            current_t = self._preprocess_frame(current, (1.0, 1.0))

        # box coords -> resized image coords, normalized to [0,1]
        obj_box = self._scale_box(ep.object_box, (H, W)) / self.image_size
        tgt_box = self._scale_box(ep.target_box, (H, W)) / self.image_size

        if self.prompt_form in ("none", "vision_bbox"):
            prompt_object = torch.empty(0, dtype=torch.float32)
            prompt_target = torch.empty(0, dtype=torch.float32)
        elif self.prompt_form == "point":
            # use bbox center
            obj_pt = np.asarray(
                [(obj_box[0] + obj_box[2]) / 2, (obj_box[1] + obj_box[3]) / 2],
                dtype=np.float32,
            )
            tgt_pt = np.asarray(
                [(tgt_box[0] + tgt_box[2]) / 2, (tgt_box[1] + tgt_box[3]) / 2],
                dtype=np.float32,
            )
            prompt_object = torch.from_numpy(obj_pt)
            prompt_target = torch.from_numpy(tgt_pt)
        elif self.prompt_form in ("bbox", "vision_bbox_and_bbox"):
            prompt_object = torch.from_numpy(obj_box)
            prompt_target = torch.from_numpy(tgt_box)
        elif self.prompt_form == "object_only":
            prompt_object = torch.from_numpy(obj_box)
            prompt_target = torch.empty(0, dtype=torch.float32)
        elif self.prompt_form == "target_only":
            prompt_object = torch.empty(0, dtype=torch.float32)
            prompt_target = torch.from_numpy(tgt_box)
        else:
            raise ValueError(f"unknown prompt_form: {self.prompt_form!r}")

        history_actions, future_actions = self.actions_at(ei, step)

        return {
            "first_frame": first_t,               # (3,H,W)
            "current_frame": current_t,           # (3,H,W)
            "prompt_object": prompt_object,       # (4,) bbox, (2,) point, or empty
            "prompt_target": prompt_target,       # may differ for object/target-only
            "action_history": torch.from_numpy(history_actions).float(),  # (k,10)
            "future_actions": torch.from_numpy(future_actions).float(),   # (H,10)
            "is_final_chunk": torch.tensor(is_final_chunk, dtype=torch.bool),
        }
