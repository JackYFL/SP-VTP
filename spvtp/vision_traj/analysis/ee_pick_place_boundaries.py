"""Find per-video pick/place landmarks from EE translation and gripper width.

The detector uses the ARKit/world Y coordinate as height (positive upward).
Sustained gripper closing/opening localizes pick/place, low EE-height basins
refine the two contact frames, and the highest EE position between contacts
defines the pick/place split.  Position-only detection is retained as an
explicit fallback for episodes whose gripper signal cannot be calibrated.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import zarr


@dataclass(frozen=True)
class PositionLandmarks:
    n_steps: int
    pick_index: int
    split_index: int
    place_index: int
    pick_prominence_m: float
    place_prominence_m: float
    confidence: float
    quality: str
    detection: str


@dataclass(frozen=True)
class GripperTransitions:
    close_start: int
    close_complete: int
    release_start: int
    release_complete: int
    open_level_m: float
    closed_level_m: float
    amplitude_m: float


@dataclass(frozen=True)
class PickPlaceLandmarks:
    n_steps: int
    pick_index: int
    split_index: int
    place_index: int
    position_pick_index: int
    position_split_index: int
    position_place_index: int
    gripper_close_start: int | None
    gripper_close_complete: int | None
    gripper_release_start: int | None
    gripper_release_complete: int | None
    pick_prominence_m: float
    place_prominence_m: float
    confidence: float
    quality: str
    gripper_used: bool
    detection: str


def _odd_window(length: int, fraction: float, minimum: int, maximum: int) -> int:
    window = int(round(length * fraction))
    window = min(maximum, max(minimum, window))
    if window % 2 == 0:
        window += 1
    if window >= length:
        window = length if length % 2 else length - 1
    return max(1, window)


def _interpolate_finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    for axis in range(values.shape[1]):
        finite = np.isfinite(values[:, axis])
        if not finite.any():
            raise ValueError(f"EE position axis {axis} has no finite values")
        if not finite.all():
            indices = np.arange(len(values))
            values[:, axis] = np.interp(
                indices, indices[finite], values[finite, axis]
            )
    return values


def _median_smooth(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    return np.asarray(
        [np.median(padded[index : index + window], axis=0) for index in range(len(values))]
    )


def _mean_smooth(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(values.shape[1])],
        axis=1,
    )


def _contact_center(
    height: np.ndarray,
    speed: np.ndarray,
    start: int,
    stop: int,
    height_range: float,
) -> int:
    """Return the low-speed center of the deepest height basin in [start, stop)."""
    start = max(0, int(start))
    stop = min(len(height), int(stop))
    if stop - start < 3:
        return max(0, min(len(height) - 1, (start + stop) // 2))

    minimum = start + int(np.argmin(height[start:stop]))
    local_radius = max(3, int(round(0.055 * len(height))))
    low = max(start, minimum - local_radius)
    high = min(stop, minimum + local_radius + 1)
    threshold = float(height[minimum] + max(0.003, 0.035 * height_range))
    basin_mask = height[low:high] <= threshold
    relative_minimum = minimum - low

    left = relative_minimum
    while left > 0 and basin_mask[left - 1]:
        left -= 1
    right = relative_minimum
    while right + 1 < len(basin_mask) and basin_mask[right + 1]:
        right += 1
    candidates = np.arange(low + left, low + right + 1)
    if not len(candidates):
        return minimum

    candidate_speed = speed[candidates]
    best_speed = float(np.min(candidate_speed))
    near_best = candidates[candidate_speed <= best_speed + 1e-9]
    basin_midpoint = 0.5 * (candidates[0] + candidates[-1])
    return int(near_best[np.argmin(np.abs(near_best - basin_midpoint))])


def _position_signals(
    translations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    translations = _interpolate_finite(np.asarray(translations, dtype=np.float64))
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError(f"translations must have shape (T, 3), got {translations.shape}")
    n_steps = len(translations)
    median_window = _odd_window(n_steps, 0.025, 3, 9)
    mean_window = _odd_window(n_steps, 0.025, 3, 9)
    smooth = _mean_smooth(_median_smooth(translations, median_window), mean_window)
    height = smooth[:, 1]
    speed = np.r_[0.0, np.linalg.norm(np.diff(smooth, axis=0), axis=1)]
    speed = _mean_smooth(speed[:, None], _odd_window(n_steps, 0.02, 3, 7))[:, 0]
    return smooth, height, speed, float(np.ptp(height))


def _split_apex(height: np.ndarray, pick_index: int, place_index: int) -> int:
    n_steps = len(height)
    split_low = min(
        place_index - 1,
        pick_index + max(2, int(round(0.035 * n_steps))),
    )
    split_high = max(
        split_low + 1,
        place_index - max(2, int(round(0.035 * n_steps))),
    )
    split_high = min(place_index, split_high)
    if split_high <= split_low:
        return (pick_index + place_index) // 2
    apex = float(np.max(height[split_low : split_high + 1]))
    apex_tolerance = max(0.002, 0.015 * float(np.ptp(height)))
    candidates = np.flatnonzero(
        height[split_low : split_high + 1] >= apex - apex_tolerance
    ) + split_low
    midpoint = 0.5 * (pick_index + place_index)
    return int(candidates[np.argmin(np.abs(candidates - midpoint))])


def _contact_prominences(
    height: np.ndarray,
    pick_index: int,
    split_index: int,
    place_index: int,
) -> tuple[float, float]:
    pick_left_high = float(np.max(height[: pick_index + 1]))
    pick_right_high = float(np.max(height[pick_index : split_index + 1]))
    pick_prominence = max(
        0.0,
        min(pick_left_high, pick_right_high) - float(height[pick_index]),
    )
    place_left_high = float(np.max(height[split_index : place_index + 1]))
    place_prominence = max(0.0, place_left_high - float(height[place_index]))
    return pick_prominence, place_prominence


def detect_position_landmarks(translations: np.ndarray) -> PositionLandmarks:
    """Detect pick contact, pick/place split, and place contact from XYZ only."""
    translations = _interpolate_finite(np.asarray(translations, dtype=np.float64))
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError(f"translations must have shape (T, 3), got {translations.shape}")
    n_steps = len(translations)
    if n_steps < 24:
        raise ValueError(f"at least 24 valid EE positions are required, got {n_steps}")

    _, height, speed, height_range = _position_signals(translations)

    pick_start = max(2, int(round(0.12 * n_steps)))
    pick_stop = min(n_steps - 6, int(round(0.62 * n_steps)))
    pick_index = _contact_center(height, speed, pick_start, pick_stop, height_range)

    minimum_separation = max(8, int(round(0.18 * n_steps)))
    place_start = max(
        pick_index + minimum_separation,
        int(round(0.52 * n_steps)),
    )
    place_stop = min(n_steps - 2, int(round(0.97 * n_steps)))
    if place_stop - place_start < 3:
        place_start = min(n_steps - 5, pick_index + max(4, minimum_separation // 2))
        place_stop = n_steps - 1
    place_index = _contact_center(height, speed, place_start, place_stop, height_range)

    split_index = _split_apex(height, pick_index, place_index)
    pick_prominence, place_prominence = _contact_prominences(
        height, pick_index, split_index, place_index
    )

    positive_speed = speed[speed > 1e-8]
    typical_speed = float(np.median(positive_speed)) if len(positive_speed) else 1e-8
    dwell_score = 0.5 * (
        np.exp(-float(speed[pick_index]) / max(typical_speed, 1e-8))
        + np.exp(-float(speed[place_index]) / max(typical_speed, 1e-8))
    )
    prominence_score = 0.5 * (
        min(1.0, pick_prominence / 0.04)
        + min(1.0, place_prominence / 0.04)
    )
    pick_progress = pick_index / max(1, n_steps - 1)
    place_progress = place_index / max(1, n_steps - 1)
    timing_score = 0.5 * (
        np.exp(-((pick_progress - 0.42) / 0.22) ** 2)
        + np.exp(-((place_progress - 0.82) / 0.20) ** 2)
    )
    confidence = float(
        np.clip(0.65 * prominence_score + 0.20 * dwell_score + 0.15 * timing_score, 0.0, 1.0)
    )
    if pick_prominence >= 0.015 and place_prominence >= 0.015 and confidence >= 0.70:
        quality = "high"
    elif pick_prominence >= 0.008 and place_prominence >= 0.008 and confidence >= 0.45:
        quality = "medium"
    else:
        quality = "low"

    return PositionLandmarks(
        n_steps=n_steps,
        pick_index=pick_index,
        split_index=split_index,
        place_index=place_index,
        pick_prominence_m=float(pick_prominence),
        place_prominence_m=float(place_prominence),
        confidence=confidence,
        quality=quality,
        detection="ee_y_contact_basins_and_transport_apex",
    )


def _first_run(mask: np.ndarray, start: int, stop: int, length: int) -> int | None:
    stop = min(stop, len(mask))
    for index in range(max(0, start), max(0, stop - length + 1)):
        if bool(np.all(mask[index : index + length])):
            return index
    return None


def _gripper_transitions(gripper: np.ndarray) -> GripperTransitions | None:
    """Detect sustained close/open transition bounds from calibrated width."""
    gripper = np.asarray(gripper, dtype=np.float64).reshape(-1)
    if len(gripper) < 24:
        return None
    window = _odd_window(len(gripper), 0.025, 3, 9)
    smooth = _median_smooth(gripper[:, None], window)[:, 0]
    head = max(5, int(round(0.12 * len(smooth))))
    open_level = float(np.median(smooth[:head]))
    closed_level = float(np.percentile(smooth, 10))
    amplitude = open_level - closed_level
    if not np.isfinite(amplitude) or amplitude < 0.015:
        return None
    openness = np.clip((smooth - closed_level) / amplitude, -0.25, 1.5)
    run = max(2, min(5, int(round(0.025 * len(smooth)))))
    close_complete = _first_run(
        openness <= 0.25,
        int(round(0.10 * len(smooth))),
        int(round(0.72 * len(smooth))),
        run,
    )
    if close_complete is None:
        return None

    close_search_start = max(
        1, close_complete - max(6, int(round(0.20 * len(smooth))))
    )
    open_before_close = np.flatnonzero(
        openness[close_search_start:close_complete] >= 0.75
    )
    close_start = (
        close_search_start + int(open_before_close[-1]) + 1
        if len(open_before_close)
        else max(1, close_complete - max(2, int(round(0.05 * len(smooth)))))
    )

    release_complete = _first_run(
        openness >= 0.40,
        max(close_complete + run, int(round(0.55 * len(smooth)))),
        len(smooth),
        2,
    )
    if release_complete is None:
        return None
    release_search_start = max(
        close_complete + 1,
        release_complete - max(5, int(round(0.20 * len(smooth)))),
    )
    closed_before_release = np.flatnonzero(
        openness[release_search_start:release_complete] <= 0.20
    )
    release_start = (
        release_search_start + int(closed_before_release[-1]) + 1
        if len(closed_before_release)
        else max(
            close_complete + 2,
            release_complete - max(2, int(round(0.05 * len(smooth)))),
        )
    )
    return GripperTransitions(
        close_start=int(close_start),
        close_complete=int(close_complete),
        release_start=int(release_start),
        release_complete=int(release_complete),
        open_level_m=open_level,
        closed_level_m=closed_level,
        amplitude_m=float(amplitude),
    )


def detect_pick_place_landmarks(
    translations: np.ndarray,
    gripper_width: np.ndarray | None,
) -> PickPlaceLandmarks:
    """Fuse gripper transitions with EE-height contact geometry."""
    position = detect_position_landmarks(translations)
    transitions = (
        _gripper_transitions(gripper_width) if gripper_width is not None else None
    )
    if transitions is None:
        return PickPlaceLandmarks(
            n_steps=position.n_steps,
            pick_index=position.pick_index,
            split_index=position.split_index,
            place_index=position.place_index,
            position_pick_index=position.pick_index,
            position_split_index=position.split_index,
            position_place_index=position.place_index,
            gripper_close_start=None,
            gripper_close_complete=None,
            gripper_release_start=None,
            gripper_release_complete=None,
            pick_prominence_m=position.pick_prominence_m,
            place_prominence_m=position.place_prominence_m,
            confidence=min(position.confidence, 0.49),
            quality="low",
            gripper_used=False,
            detection="position_only_fallback_invalid_gripper",
        )

    _, height, speed, height_range = _position_signals(translations)
    n_steps = len(height)
    pick_margin = max(4, int(round(0.06 * n_steps)))
    place_margin_before = max(5, int(round(0.08 * n_steps)))
    place_margin_after = max(4, int(round(0.05 * n_steps)))
    pick_index = _contact_center(
        height,
        speed,
        transitions.close_start - pick_margin,
        transitions.close_complete + pick_margin + 1,
        height_range,
    )
    place_index = _contact_center(
        height,
        speed,
        transitions.release_start - place_margin_before,
        transitions.release_complete + place_margin_after + 1,
        height_range,
    )
    if place_index <= pick_index + max(6, int(round(0.12 * n_steps))):
        return PickPlaceLandmarks(
            n_steps=position.n_steps,
            pick_index=position.pick_index,
            split_index=position.split_index,
            place_index=position.place_index,
            position_pick_index=position.pick_index,
            position_split_index=position.split_index,
            position_place_index=position.place_index,
            gripper_close_start=transitions.close_start,
            gripper_close_complete=transitions.close_complete,
            gripper_release_start=transitions.release_start,
            gripper_release_complete=transitions.release_complete,
            pick_prominence_m=position.pick_prominence_m,
            place_prominence_m=position.place_prominence_m,
            confidence=min(position.confidence, 0.49),
            quality="low",
            gripper_used=False,
            detection="position_fallback_inconsistent_gripper_order",
        )

    split_index = _split_apex(height, pick_index, place_index)
    pick_prominence, place_prominence = _contact_prominences(
        height, pick_index, split_index, place_index
    )
    close_center = 0.5 * (transitions.close_start + transitions.close_complete)
    release_center = 0.5 * (transitions.release_start + transitions.release_complete)
    agreement = 0.5 * (
        np.exp(-abs(position.pick_index - close_center) / max(4.0, 0.08 * n_steps))
        + np.exp(-abs(position.place_index - release_center) / max(4.0, 0.08 * n_steps))
    )
    amplitude_score = min(1.0, transitions.amplitude_m / 0.05)
    confidence = float(
        np.clip(
            0.55 * position.confidence + 0.30 * agreement + 0.15 * amplitude_score,
            0.0,
            1.0,
        )
    )
    if pick_prominence >= 0.012 and place_prominence >= 0.012 and confidence >= 0.70:
        quality = "high"
    elif pick_prominence >= 0.006 and place_prominence >= 0.006 and confidence >= 0.50:
        quality = "medium"
    else:
        quality = "low"
    return PickPlaceLandmarks(
        n_steps=n_steps,
        pick_index=int(pick_index),
        split_index=int(split_index),
        place_index=int(place_index),
        position_pick_index=position.pick_index,
        position_split_index=position.split_index,
        position_place_index=position.place_index,
        gripper_close_start=transitions.close_start,
        gripper_close_complete=transitions.close_complete,
        gripper_release_start=transitions.release_start,
        gripper_release_complete=transitions.release_complete,
        pick_prominence_m=float(pick_prominence),
        place_prominence_m=float(place_prominence),
        confidence=confidence,
        quality=quality,
        gripper_used=True,
        detection="ee_y_contact_basins_refined_by_gripper_transitions",
    )


def _episode_identity(data_root: Path, video_path: Path) -> tuple[str, str, str, str]:
    relative = video_path.relative_to(data_root)
    scene, task, _, episode, _ = relative.parts
    key = f"{scene}/{task}/{episode}/camera_1.mp4"
    return key, scene, task, episode


def _load_episode(episode_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    root = zarr.open(str(episode_dir), mode="r")
    valid = np.asarray(root["valid_indices"][:], dtype=bool)
    pose = np.asarray(root["pose_interp"][:], dtype=np.float64)
    gripper = (
        np.asarray(root["gripper_widths"][:], dtype=np.float64).reshape(-1)
        if "gripper_widths" in root
        else None
    )
    lengths = [len(valid), len(pose)]
    if gripper is not None:
        lengths.append(len(gripper))
    common_length = min(lengths)
    valid_indices = np.flatnonzero(valid[:common_length])
    if len(valid_indices) < 24:
        raise ValueError(f"only {len(valid_indices)} valid frames")
    translations = pose[valid_indices, :3, 3]
    episode_gripper = gripper[valid_indices] if gripper is not None else None
    return valid_indices, translations, episode_gripper


def _row_for_episode(
    data_root: Path,
    video_path: Path,
    use_gripper: bool,
) -> dict:
    episode_dir = video_path.parent
    key, scene, task, episode = _episode_identity(data_root, video_path)
    raw_indices, translations, gripper = _load_episode(episode_dir)
    landmarks = detect_pick_place_landmarks(
        translations, gripper if use_gripper else None
    )
    row = {
        "episode_key": key,
        "scene": scene,
        "task": task,
        "episode": episode,
        "video_path": str(video_path),
        **asdict(landmarks),
    }
    for name, index in (
        ("pick", landmarks.pick_index),
        ("split", landmarks.split_index),
        ("place", landmarks.place_index),
    ):
        row[f"{name}_raw_frame"] = int(raw_indices[index])
        row[f"{name}_x_m"] = float(translations[index, 0])
        row[f"{name}_y_m"] = float(translations[index, 1])
        row[f"{name}_z_m"] = float(translations[index, 2])

    for name in (
        "gripper_close_start",
        "gripper_close_complete",
        "gripper_release_start",
        "gripper_release_complete",
    ):
        index = getattr(landmarks, name)
        row[f"{name}_raw_frame"] = "" if index is None else int(raw_indices[index])
    if landmarks.gripper_close_complete is None:
        row["position_pick_gripper_abs_error_frames"] = ""
    else:
        row["position_pick_gripper_abs_error_frames"] = int(
            abs(
                int(raw_indices[landmarks.position_pick_index])
                - int(raw_indices[landmarks.gripper_close_complete])
            )
        )
    if landmarks.gripper_release_start is None:
        row["position_place_gripper_abs_error_frames"] = ""
    else:
        row["position_place_gripper_abs_error_frames"] = int(
            abs(
                int(raw_indices[landmarks.position_place_index])
                - int(raw_indices[landmarks.gripper_release_start])
            )
        )
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_contact_sheet(
    output_path: Path,
    rows: list[dict],
    count: int,
    seed: int,
) -> None:
    if count <= 0 or not rows:
        return
    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    scenes = sorted({row["scene"] for row in rows})
    per_scene = max(1, int(np.ceil(count / len(scenes))))
    for scene in scenes:
        candidates = [row for row in rows if row["scene"] == scene]
        indices = rng.choice(
            len(candidates), size=min(per_scene, len(candidates)), replace=False
        )
        selected.extend(candidates[int(index)] for index in sorted(indices))
    selected = selected[:count]

    cell_width, cell_height, label_height = 300, 200, 48
    sheet = np.full(
        (len(selected) * (cell_height + label_height), 3 * cell_width, 3),
        245,
        dtype=np.uint8,
    )
    landmarks = ("pick", "split", "place")
    for row_index, row in enumerate(selected):
        capture = cv2.VideoCapture(row["video_path"])
        try:
            for column, name in enumerate(landmarks):
                raw_frame = int(row[f"{name}_raw_frame"])
                capture.set(cv2.CAP_PROP_POS_FRAMES, raw_frame)
                ok, frame = capture.read()
                if not ok:
                    frame = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
                else:
                    frame = cv2.resize(frame, (cell_width, cell_height))
                y = row_index * (cell_height + label_height)
                x = column * cell_width
                sheet[y : y + cell_height, x : x + cell_width] = frame
                cv2.putText(
                    sheet,
                    f"{name} frame={raw_frame}",
                    (x + 5, y + cell_height + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                if column == 0:
                    cv2.putText(
                        sheet,
                        f"{row['scene']}/{row['task']}/{row['episode']} {row['quality']}",
                        (x + 5, y + cell_height + 39),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
        finally:
            capture.release()
    cv2.imwrite(str(output_path), sheet)


def _summary(rows: list[dict], failures: list[dict]) -> dict:
    quality_counts = Counter(row["quality"] for row in rows)
    scene_counts = Counter(row["scene"] for row in rows)
    gripper_counts = Counter("used" if row["gripper_used"] else "fallback" for row in rows)
    summary = {
        "method": "gripper close/open transitions + EE-Y contact basins + inter-contact height apex",
        "height_axis": "world_y (positive upward)",
        "videos_discovered": len(rows) + len(failures),
        "videos_succeeded": len(rows),
        "videos_failed": len(failures),
        "quality_counts": dict(quality_counts),
        "gripper_counts": dict(gripper_counts),
        "scene_counts": dict(scene_counts),
    }
    if rows:
        summary["progress_statistics"] = {}
        for name in ("pick", "split", "place"):
            values = np.asarray(
                [row[f"{name}_index"] / max(1, row["n_steps"] - 1) for row in rows],
                dtype=np.float64,
            )
            summary["progress_statistics"][name] = {
                "median": float(np.median(values)),
                "q10": float(np.quantile(values, 0.10)),
                "q90": float(np.quantile(values, 0.90)),
            }

    references = [row for row in rows if row["gripper_used"]]
    if references:
        agreement = {"episodes": len(references)}
        for name in ("pick", "place"):
            errors = np.asarray(
                [row[f"position_{name}_gripper_abs_error_frames"] for row in references],
                dtype=np.float64,
            )
            agreement[name] = {
                "median_abs_error_frames": float(np.median(errors)),
                "q90_abs_error_frames": float(np.quantile(errors, 0.90)),
                "within_10_frames": float(np.mean(errors <= 10)),
                "within_20_frames": float(np.mean(errors <= 20)),
            }
        summary["position_gripper_agreement"] = agreement
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="spvtp/EgoSPT")
    parser.add_argument(
        "--output-dir",
        default="runs/vision_traj_experiments_final/ee_pick_place_boundaries",
    )
    parser.add_argument("--diagnostic-episodes", type=int, default=18)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--position-only",
        action="store_true",
        help="disable gripper fusion and force the EE-position fallback",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(
        data_root.glob("*/*/recording_output_processed/*/camera_1.mp4")
    )
    if args.limit > 0:
        videos = videos[: args.limit]

    def process(video_path: Path) -> tuple[dict | None, dict | None]:
        try:
            return (
                _row_for_episode(
                    data_root, video_path, use_gripper=not args.position_only
                ),
                None,
            )
        except (KeyError, OSError, ValueError, zarr.errors.PathNotFoundError) as error:
            return None, {"video_path": str(video_path), "error": str(error)}

    rows: list[dict] = []
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        results = executor.map(process, videos)
        for index, (row, failure) in enumerate(results, start=1):
            if row is not None:
                rows.append(row)
            if failure is not None:
                failures.append(failure)
            if index % 250 == 0 or index == len(videos):
                print(
                    f"[ee-boundaries] {index}/{len(videos)} "
                    f"succeeded={len(rows)} failed={len(failures)}",
                    flush=True,
                )

    _write_csv(output_dir / "ee_pick_place_boundaries.csv", rows)
    _write_csv(output_dir / "failures.csv", failures)
    summary = _summary(rows, failures)
    with open(output_dir / "summary.json", "w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    _make_contact_sheet(
        output_dir / "ee_pick_place_contact_sheet.jpg",
        rows,
        count=args.diagnostic_episodes,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
