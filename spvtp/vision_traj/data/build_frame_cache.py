"""Predecode the exact vision-trajectory training frames into local mmap arrays.

Each episode produces one ``.npy`` array with four task-frame variants followed
by all valid strided current frames.  This replaces random MP4 seeking in the
training DataLoader while preserving the model inputs pixel-for-pixel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import zarr

from spvtp.common.episode_arrays import align_valid_frames
from spvtp.vision_traj.data.dataset import (
    EpisodeRecord,
    VisionTrajDataset,
    build_episode_index,
)


TASK_FRAME_COUNT = 4


from .images import draw_prompt_boxes as _draw_prompt_boxes


def _resize(frame: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(
        frame,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )


def _cache_paths(output_root: Path, episode: EpisodeRecord) -> tuple[Path, Path]:
    output = output_root / episode.scene / episode.task / f"{episode.zarr_root.name}.npy"
    return output, output.with_suffix(".json")


def _valid_existing(
    array_path: Path,
    metadata_path: Path,
    stride: int,
    image_size: int,
    raw_indices: np.ndarray,
) -> bool:
    if not array_path.is_file() or not metadata_path.is_file():
        return False
    try:
        with open(metadata_path, "r") as stream:
            metadata = json.load(stream)
        frames = np.load(array_path, mmap_mode="r")
        return (
            int(metadata["stride"]) == stride
            and int(metadata["image_size"]) == image_size
            and int(metadata["n_strided_frames"]) == len(raw_indices)
            and metadata.get("raw_indices_sha256") == _indices_digest(raw_indices)
            and frames.shape == (len(raw_indices) + TASK_FRAME_COUNT, image_size, image_size, 3)
            and frames.dtype == np.uint8
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def _indices_digest(raw_indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(raw_indices, dtype="<i8").tobytes()).hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def _build_episode(
    episode: EpisodeRecord,
    output_root: str,
    stride: int,
    image_size: int,
    overwrite: bool,
) -> dict:
    cv2.setNumThreads(0)
    output_root_path = Path(output_root)
    array_path, metadata_path = _cache_paths(output_root_path, episode)
    root = zarr.open(str(episode.zarr_root), mode="r")
    valid = np.asarray(root["valid_indices"][:], dtype=bool)
    # Use the loader's alignment contract, including legacy compact arrays.
    for key in ("pose_interp", "gripper_widths"):
        align_valid_frames(root[key][:], valid)
    raw_indices = np.flatnonzero(valid)[::stride]
    if not len(raw_indices):
        raise ValueError(f"no valid strided frames for {episode.key}")
    if not overwrite and _valid_existing(
        array_path, metadata_path, stride, image_size, raw_indices
    ):
        return {"status": "skipped", "episode_key": episode.key}

    capture = cv2.VideoCapture(str(episode.video_path))
    if not capture.isOpened():
        raise OSError(f"cannot open {episode.video_path}")
    try:
        ok, first_bgr = capture.read()
        if not ok:
            raise OSError(f"cannot read frame 0 from {episode.video_path}")
        first = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
        source_hw = first.shape[:2]
        frames = np.empty(
            (len(raw_indices) + TASK_FRAME_COUNT, image_size, image_size, 3),
            dtype=np.uint8,
        )
        frames[0] = _resize(first, image_size)
        frames[1] = _resize(
            _draw_prompt_boxes(first, episode.object_box, episode.target_box),
            image_size,
        )
        frames[2] = _resize(
            _draw_prompt_boxes(first, object_box=episode.object_box), image_size
        )
        frames[3] = _resize(
            _draw_prompt_boxes(first, target_box=episode.target_box), image_size
        )

        selected = {int(raw): offset for offset, raw in enumerate(raw_indices)}
        if 0 in selected:
            frames[TASK_FRAME_COUNT + selected[0]] = _resize(first, image_size)
        maximum = int(raw_indices[-1])
        for raw_frame in range(1, maximum + 1):
            ok, frame_bgr = capture.read()
            if not ok:
                raise OSError(
                    f"cannot read frame {raw_frame} from {episode.video_path}"
                )
            offset = selected.get(raw_frame)
            if offset is not None:
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames[TASK_FRAME_COUNT + offset] = _resize(frame, image_size)
    finally:
        capture.release()

    array_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = array_path.with_suffix(array_path.suffix + f".tmp.{os.getpid()}")
    with open(temporary, "wb") as stream:
        np.save(stream, frames, allow_pickle=False)
    os.replace(temporary, array_path)
    _write_json_atomic(
        metadata_path,
        {
            "episode_key": episode.key,
            "source_video": str(episode.video_path),
            "source_hw": [int(source_hw[0]), int(source_hw[1])],
            "stride": stride,
            "image_size": image_size,
            "n_strided_frames": int(len(raw_indices)),
            "first_raw_frame": int(raw_indices[0]),
            "last_raw_frame": int(raw_indices[-1]),
            "raw_indices_sha256": _indices_digest(raw_indices),
        },
    )
    return {
        "status": "built",
        "episode_key": episode.key,
        "frames": int(len(raw_indices)),
        "bytes": int(frames.nbytes),
    }


def _verify_episode_cache(
    episode: EpisodeRecord,
    output_root: Path,
    stride: int,
    image_size: int,
) -> None:
    prompt_forms = (
        "none",
        "bbox",
        "point",
        "vision_bbox",
        "vision_bbox_and_bbox",
        "object_only",
        "target_only",
    )
    for prompt_form in prompt_forms:
        common = {
            "episodes": [episode],
            "action_horizon": 16,
            "history_horizon": 4,
            "image_size": image_size,
            "stride": stride,
            "prompt_form": prompt_form,
        }
        direct = VisionTrajDataset(**common)
        cached = VisionTrajDataset(**common, frame_cache_root=output_root)
        for sample_index in {0, len(direct) - 1}:
            direct_sample = direct[sample_index]
            cached_sample = cached[sample_index]
            for key in direct_sample:
                if not torch.equal(direct_sample[key], cached_sample[key]):
                    difference = float(
                        torch.max(
                            torch.abs(
                                direct_sample[key].float()
                                - cached_sample[key].float()
                            )
                        )
                    )
                    raise AssertionError(
                        f"cache mismatch episode={episode.key} prompt={prompt_form} "
                        f"sample={sample_index} key={key} max_abs={difference}"
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verify-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = build_episode_index(args.data_root, args.annotations)
    if args.limit > 0:
        episodes = episodes[: args.limit]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    counts = {"built": 0, "skipped": 0, "failed": 0}
    failures = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _build_episode,
                episode,
                str(output_root),
                args.stride,
                args.image_size,
                args.overwrite,
            ): episode
            for episode in episodes
        }
        for future in as_completed(futures):
            episode = futures[future]
            completed += 1
            try:
                result = future.result()
                counts[result["status"]] += 1
            except Exception as error:  # noqa: BLE001 - record and continue
                counts["failed"] += 1
                failures.append(
                    {
                        "episode_key": episode.key,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            if completed % 50 == 0 or completed == len(episodes):
                print(
                    f"[frame-cache] {completed}/{len(episodes)} "
                    f"built={counts['built']} skipped={counts['skipped']} "
                    f"failed={counts['failed']}",
                    flush=True,
                )

    summary = {
        "data_root": str(args.data_root),
        "output_root": str(output_root),
        "episodes": len(episodes),
        "stride": args.stride,
        "image_size": args.image_size,
        **counts,
        "failures": failures,
    }
    _write_json_atomic(output_root / "summary.json", summary)
    if failures:
        raise SystemExit(f"frame cache completed with {len(failures)} failures")
    verify_count = min(args.verify_samples, len(episodes))
    for episode in episodes[:verify_count]:
        _verify_episode_cache(episode, output_root, args.stride, args.image_size)
    if verify_count:
        print(
            f"[frame-cache] verified exact tensors for {verify_count} episodes",
            flush=True,
        )


if __name__ == "__main__":
    main()
