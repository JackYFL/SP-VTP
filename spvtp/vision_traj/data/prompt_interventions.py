"""Deterministic prompt interventions for causal evaluation.

The shuffled conditions transplant a normalized prompt box from a validation
episode in the same scene whose semantic role differs from the evaluated
episode.  The RGB observation, action history, and trajectory labels remain
unchanged.  Boxes are normalized in the donor frame and then mapped into the
recipient frame so the intervention is resolution independent.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from spvtp.vision_traj.data.dataset import EpisodeRecord


PROMPT_INTERVENTIONS = (
    "none",
    "shuffled_object",
    "shuffled_target",
    "role_swap",
    "jitter_2",
    "jitter_5",
    "jitter_10",
    "jitter_20",
    "scale_80",
    "scale_90",
    "scale_110",
    "scale_120",
    "missing_object",
    "missing_target",
)

JITTER_FRACTIONS = {
    "jitter_2": 0.02,
    "jitter_5": 0.05,
    "jitter_10": 0.10,
    "jitter_20": 0.20,
}
SCALE_FACTORS = {
    "scale_80": 0.80,
    "scale_90": 0.90,
    "scale_110": 1.10,
    "scale_120": 1.20,
}
MISSING_ROLE_INTERVENTIONS = {"missing_object", "missing_target"}


def parse_task_roles(task: str) -> tuple[str, str, str]:
    """Return object/target identifiers and how they were obtained.

    Scene 1/2 tasks expose semantic ``put_X_to_Y`` names.  Scene 3 uses opaque
    ``subsceneN`` identifiers, so only a different task/subscene can be
    guaranteed there; its object and target semantics are not inferred.
    """
    match = re.fullmatch(r"put_(.+)_to_(.+)", task)
    if match is not None:
        return match.group(1), match.group(2), "semantic_task_name"
    opaque_role = f"task:{task}"
    return opaque_role, opaque_role, "opaque_task_fallback"


def _video_frame_size(video_path: Path) -> tuple[int, int]:
    """Return ``(height, width)`` without decoding the full video."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise OSError(f"cannot open video to read frame size: {video_path}")
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid video frame size ({height}, {width}): {video_path}")
    return height, width


def _normalise_box(box: np.ndarray, frame_hw: tuple[int, int]) -> np.ndarray:
    height, width = frame_hw
    scale = np.asarray([width, height, width, height], dtype=np.float32)
    return np.asarray(box, dtype=np.float32) / scale


def _denormalise_box(box: np.ndarray, frame_hw: tuple[int, int]) -> np.ndarray:
    height, width = frame_hw
    scale = np.asarray([width, height, width, height], dtype=np.float32)
    return np.asarray(box, dtype=np.float32) * scale


def _stable_choice(candidates: Sequence[EpisodeRecord], token: str) -> EpisodeRecord:
    ordered = sorted(candidates, key=lambda episode: episode.key)
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], byteorder="big") % len(ordered)
    return ordered[index]


def _stable_uniform_pair(token: str) -> np.ndarray:
    """Return a deterministic 2-D sample in ``[-1, 1]``."""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    integers = np.frombuffer(digest[:16], dtype=np.uint64).astype(np.float64)
    return 2.0 * integers / np.iinfo(np.uint64).max - 1.0


def _jitter_box(box: np.ndarray, fraction: float, token: str) -> np.ndarray:
    """Shift a normalized box center while preserving its size and validity."""
    box = np.asarray(box, dtype=np.float32)
    delta = _stable_uniform_pair(token).astype(np.float32) * fraction
    low = box[:2]
    high = box[2:]
    delta = np.maximum(delta, -low)
    delta = np.minimum(delta, 1.0 - high)
    return np.concatenate([low + delta, high + delta]).astype(np.float32)


def _scale_box(box: np.ndarray, factor: float) -> np.ndarray:
    """Scale a normalized box about its center and clip it to the image."""
    box = np.asarray(box, dtype=np.float32)
    center = 0.5 * (box[:2] + box[2:])
    half_size = 0.5 * (box[2:] - box[:2]) * factor
    return np.concatenate(
        [np.clip(center - half_size, 0.0, 1.0), np.clip(center + half_size, 0.0, 1.0)]
    ).astype(np.float32)


def apply_prompt_intervention(
    episodes: Sequence[EpisodeRecord],
    intervention: str,
    *,
    seed: int = 0,
    frame_size_fn: Callable[[Path], tuple[int, int]] = _video_frame_size,
) -> tuple[list[EpisodeRecord], list[dict]]:
    """Return copied episode records with intervened prompt boxes.

    Shuffled donors are restricted to the same scene and are required to have
    a different object/target identifier.  Selection is stable across runs and
    does not depend on Python's randomized hash seed.
    """
    if intervention not in PROMPT_INTERVENTIONS:
        raise ValueError(
            f"unknown prompt intervention {intervention!r}; "
            f"expected one of {PROMPT_INTERVENTIONS}"
        )
    if intervention == "none":
        return list(episodes), []

    by_scene: dict[str, list[EpisodeRecord]] = {}
    roles: dict[str, tuple[str, str, str]] = {}
    for episode in episodes:
        by_scene.setdefault(episode.scene, []).append(episode)
        roles[episode.key] = parse_task_roles(episode.task)

    size_cache: dict[Path, tuple[int, int]] = {}

    def frame_size(episode: EpisodeRecord) -> tuple[int, int]:
        if episode.video_path not in size_cache:
            size_cache[episode.video_path] = frame_size_fn(episode.video_path)
        return size_cache[episode.video_path]

    output: list[EpisodeRecord] = []
    manifest: list[dict] = []
    for episode in episodes:
        object_id, target_id, role_source = roles[episode.key]
        new_object = np.asarray(episode.object_box, dtype=np.float32).copy()
        new_target = np.asarray(episode.target_box, dtype=np.float32).copy()
        donor: EpisodeRecord | None = None

        intervention_parameters: dict[str, float | str] = {}
        if intervention in MISSING_ROLE_INTERVENTIONS:
            # The dataset prompt form is overridden by ``infer.py`` so both
            # the rendered cue and coordinate tokens for this role disappear.
            intervention_parameters["missing_role"] = intervention.removeprefix("missing_")
        elif intervention == "role_swap":
            new_object, new_target = new_target.copy(), new_object.copy()
        elif intervention in JITTER_FRACTIONS:
            fraction = JITTER_FRACTIONS[intervention]
            frame_hw = frame_size(episode)
            object_normalized = _normalise_box(new_object, frame_hw)
            target_normalized = _normalise_box(new_target, frame_hw)
            new_object = _denormalise_box(
                _jitter_box(
                    object_normalized,
                    fraction,
                    f"{seed}:{intervention}:{episode.key}:object",
                ),
                frame_hw,
            )
            new_target = _denormalise_box(
                _jitter_box(
                    target_normalized,
                    fraction,
                    f"{seed}:{intervention}:{episode.key}:target",
                ),
                frame_hw,
            )
            intervention_parameters["max_center_shift_fraction"] = fraction
        elif intervention in SCALE_FACTORS:
            factor = SCALE_FACTORS[intervention]
            frame_hw = frame_size(episode)
            new_object = _denormalise_box(
                _scale_box(_normalise_box(new_object, frame_hw), factor), frame_hw
            )
            new_target = _denormalise_box(
                _scale_box(_normalise_box(new_target, frame_hw), factor), frame_hw
            )
            intervention_parameters["box_scale_factor"] = factor
        else:
            role_index = 0 if intervention == "shuffled_object" else 1
            current_role = object_id if role_index == 0 else target_id
            candidates = [
                candidate
                for candidate in by_scene[episode.scene]
                if roles[candidate.key][role_index] != current_role
            ]
            if not candidates:
                raise ValueError(
                    f"no same-scene donor with a different role for {episode.key} "
                    f"under {intervention}"
                )
            donor = _stable_choice(
                candidates,
                token=f"{seed}:{intervention}:{episode.key}",
            )
            donor_box = (
                donor.object_box
                if intervention == "shuffled_object"
                else donor.target_box
            )
            donor_box_normalised = _normalise_box(donor_box, frame_size(donor))
            transplanted_box = _denormalise_box(
                donor_box_normalised, frame_size(episode)
            )
            if intervention == "shuffled_object":
                new_object = transplanted_box
            else:
                new_target = transplanted_box

        output.append(
            replace(
                episode,
                object_box=new_object,
                target_box=new_target,
            )
        )
        manifest.append(
            {
                "episode_key": episode.key,
                "scene": episode.scene,
                "task": episode.task,
                "object_id": object_id,
                "target_id": target_id,
                "role_source": role_source,
                "intervention": intervention,
                "donor_episode_key": donor.key if donor is not None else None,
                "donor_task": donor.task if donor is not None else None,
                "prompt_object": new_object.tolist(),
                "prompt_target": new_target.tolist(),
                **intervention_parameters,
            }
        )

    return output, manifest
