"""Execution-phase segmentation and per-phase trajectory evaluation.

The episode timeline is segmented from gripper transitions, translational EE
speed, and progress constraints.  Per-step prediction errors are assigned by
the phase of their *future* timestep.  Chunk FDE is assigned by the phase of
the chunk endpoint, avoiding the ambiguity of assigning a 16-step chunk to a
single current-frame phase.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import zarr


PHASES = (
    "approach",
    "pre_grasp",
    "grasp_occlusion",
    "post_grasp_transport",
    "placement",
)
METRICS = (
    "pos_l2",
    "final_displacement_error",
    "rot6d_l2",
    "rotation_geodesic_deg",
    "grip_l1",
    "grip_state_accuracy",
    "endpoint_success_10cm",
    "rotation_success_20deg",
    "combined_success_10cm_20deg_grip",
)
ENDPOINT_METRICS = {
    "final_displacement_error",
    "endpoint_success_10cm",
    "rotation_success_20deg",
    "combined_success_10cm_20deg_grip",
}
HIGHER_IS_BETTER = {
    "grip_state_accuracy",
    "endpoint_success_10cm",
    "rotation_success_20deg",
    "combined_success_10cm_20deg_grip",
}
REPORT_METRICS = {
    "pos_l2": ("pos_l2 (cm)", 100.0),
    "final_displacement_error": ("final_displacement_error (cm)", 100.0),
    "rot6d_l2": ("rot6d_l2", 1.0),
    "rotation_geodesic_deg": ("rotation_geodesic_deg", 1.0),
    "grip_l1": ("grip_l1 (cm)", 100.0),
    "grip_state_accuracy": ("grip_state_accuracy", 1.0),
    "endpoint_success_10cm": ("endpoint_success_10cm", 1.0),
    "rotation_success_20deg": ("rotation_success_20deg", 1.0),
    "combined_success_10cm_20deg_grip": (
        "combined_success_10cm_20deg_grip",
        1.0,
    ),
}


@dataclass(frozen=True)
class PhaseBoundaries:
    n_steps: int
    pre_grasp_start: int
    grasp_start: int
    transport_start: int
    placement_start: int
    close_complete: int
    release_start: int
    release_complete: int
    open_level: float
    closed_level: float
    detection: str

    def phase_at(self, step: int) -> str:
        if step < self.pre_grasp_start:
            return "approach"
        if step < self.grasp_start:
            return "pre_grasp"
        if step < self.transport_start:
            return "grasp_occlusion"
        if step < self.placement_start:
            return "post_grasp_transport"
        return "placement"


def _canonical_key(key: str) -> str:
    return key.replace("\\", "/")


def _median_smooth(values: np.ndarray, kernel: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) < 3:
        return values.copy()
    kernel = min(kernel, len(values) if len(values) % 2 else len(values) - 1)
    kernel = max(kernel, 3)
    radius = kernel // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.asarray(
        [np.median(padded[i : i + kernel]) for i in range(len(values))]
    )


def _first_run(mask: np.ndarray, start: int, stop: int, length: int) -> int | None:
    start = max(0, start)
    stop = min(len(mask), stop)
    length = max(1, length)
    for index in range(start, max(start, stop - length + 1)):
        if bool(np.all(mask[index : index + length])):
            return index
    return None


def _progress_fallback(n_steps: int, open_level: float, closed_level: float) -> PhaseBoundaries:
    boundaries = [
        max(1, int(round(n_steps * fraction)))
        for fraction in (0.28, 0.38, 0.52, 0.78)
    ]
    for index in range(1, len(boundaries)):
        boundaries[index] = max(boundaries[index], boundaries[index - 1] + 1)
    boundaries[-1] = min(boundaries[-1], n_steps - 1)
    return PhaseBoundaries(
        n_steps=n_steps,
        pre_grasp_start=boundaries[0],
        grasp_start=boundaries[1],
        transport_start=boundaries[2],
        placement_start=boundaries[3],
        close_complete=boundaries[2],
        release_start=boundaries[3],
        release_complete=min(n_steps - 1, boundaries[3] + 1),
        open_level=open_level,
        closed_level=closed_level,
        detection="progress_fallback",
    )


def segment_episode_phases(
    gripper_width: np.ndarray,
    translations: np.ndarray,
) -> PhaseBoundaries:
    """Infer five ordered manipulation phases for one strided episode."""
    gripper = _median_smooth(np.asarray(gripper_width, dtype=np.float64))
    translations = np.asarray(translations, dtype=np.float64)
    n_steps = len(gripper)
    if n_steps < 20 or translations.shape != (n_steps, 3):
        return _progress_fallback(
            n_steps,
            float(np.max(gripper)) if n_steps else 0.0,
            float(np.min(gripper)) if n_steps else 0.0,
        )

    head_count = max(5, int(round(0.12 * n_steps)))
    open_level = float(np.median(gripper[:head_count]))
    closed_level = float(np.percentile(gripper, 10))
    amplitude = open_level - closed_level
    if not np.isfinite(amplitude) or amplitude < 0.015:
        return _progress_fallback(n_steps, open_level, closed_level)

    openness = np.clip((gripper - closed_level) / amplitude, -0.25, 1.5)
    run_length = max(2, min(5, int(round(0.04 * n_steps))))
    close_complete = _first_run(
        openness <= 0.25,
        start=max(2, int(round(0.08 * n_steps))),
        stop=max(3, int(round(0.78 * n_steps))),
        length=run_length,
    )
    if close_complete is None:
        return _progress_fallback(n_steps, open_level, closed_level)

    close_search_start = max(1, close_complete - max(6, int(round(0.20 * n_steps))))
    open_before_close = np.flatnonzero(
        openness[close_search_start:close_complete] >= 0.75
    )
    if len(open_before_close):
        grasp_start = close_search_start + int(open_before_close[-1]) + 1
    else:
        grasp_start = max(2, close_complete - max(3, int(round(0.06 * n_steps))))
    grasp_start = min(grasp_start, close_complete - 1)

    release_search_start = max(
        close_complete + run_length,
        int(round(0.55 * n_steps)),
    )
    release_complete = _first_run(
        openness >= 0.40,
        start=release_search_start,
        stop=n_steps,
        length=2,
    )
    if release_complete is None:
        derivative = np.diff(gripper)
        derivative_start = max(release_search_start, int(round(0.65 * n_steps)))
        if derivative_start >= len(derivative):
            return _progress_fallback(n_steps, open_level, closed_level)
        release_complete = derivative_start + int(
            np.argmax(derivative[derivative_start:])
        ) + 1

    release_back_start = max(
        close_complete + 1,
        release_complete - max(5, int(round(0.20 * n_steps))),
    )
    closed_before_release = np.flatnonzero(
        openness[release_back_start:release_complete] <= 0.20
    )
    if len(closed_before_release):
        release_start = release_back_start + int(closed_before_release[-1]) + 1
    else:
        release_start = max(
            close_complete + 2,
            release_complete - max(2, int(round(0.05 * n_steps))),
        )

    speed = np.r_[0.0, np.linalg.norm(np.diff(translations, axis=0), axis=1)]
    speed = _median_smooth(speed)

    # Pre-grasp begins near the final deceleration before closure.  Progress
    # constraints keep noisy velocity from moving this boundary too far.
    pre_length = max(4, int(round(0.10 * n_steps)))
    pre_nominal = grasp_start - pre_length
    pre_low = max(1, pre_nominal - 2)
    pre_high = min(grasp_start - 2, pre_nominal + 2)
    if pre_high >= pre_low:
        pre_grasp_start = pre_low + int(np.argmin(speed[pre_low : pre_high + 1]))
    else:
        pre_grasp_start = max(1, pre_nominal)
    pre_grasp_start = min(pre_grasp_start, grasp_start - 2)

    # Keep the immediate post-close/lift interval inside grasp/occlusion.
    transport_start = close_complete + max(2, int(round(0.05 * n_steps)))

    # Placement includes the final target approach, not just gripper release.
    # Refine a progress-based estimate using local EE deceleration.
    placement_nominal = release_start - max(5, int(round(0.18 * n_steps)))
    place_low = max(transport_start + 2, placement_nominal - 3)
    place_high = min(release_start - 2, placement_nominal + 3)
    if place_high >= place_low:
        deceleration = np.r_[0.0, speed[:-1] - speed[1:]]
        placement_start = place_low + int(
            np.argmax(deceleration[place_low : place_high + 1])
        )
    else:
        placement_start = max(transport_start + 2, placement_nominal)

    # Enforce strict ordering while retaining at least two timeline steps in
    # every transition-centered phase.
    pre_grasp_start = max(1, min(pre_grasp_start, n_steps - 8))
    grasp_start = max(pre_grasp_start + 2, min(grasp_start, n_steps - 6))
    transport_start = max(grasp_start + 2, min(transport_start, n_steps - 4))
    placement_start = max(
        transport_start + 2,
        min(placement_start, n_steps - 2),
    )

    return PhaseBoundaries(
        n_steps=n_steps,
        pre_grasp_start=pre_grasp_start,
        grasp_start=grasp_start,
        transport_start=transport_start,
        placement_start=placement_start,
        close_complete=close_complete,
        release_start=release_start,
        release_complete=release_complete,
        open_level=open_level,
        closed_level=closed_level,
        detection="gripper_velocity_progress",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        metavar="NAME=STEP_METRICS_CSV",
        help="repeat for each evaluated model/intervention",
    )
    parser.add_argument("--diagnostic-episodes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _parse_conditions(items: Iterable[str]) -> dict[str, Path]:
    conditions = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"condition must be NAME=PATH, got {item!r}")
        name, path = item.split("=", maxsplit=1)
        if not name or name in conditions:
            raise ValueError(f"invalid or duplicate condition name: {name!r}")
        conditions[name] = Path(path)
    if not conditions:
        raise ValueError("at least one --condition is required")
    return conditions


def _episode_dir(data_root: Path, episode_key: str) -> Path:
    parts = _canonical_key(episode_key).split("/")
    if len(parts) < 4:
        raise ValueError(f"invalid episode key: {episode_key!r}")
    return data_root / parts[0] / parts[1] / "recording_output_processed" / parts[2]


def _load_boundaries(
    manifest_path: Path,
    data_root: Path,
    stride: int,
    required_keys: set[str],
) -> tuple[dict[str, PhaseBoundaries], dict[str, dict], dict[str, np.ndarray]]:
    with open(manifest_path, "r") as stream:
        manifest = json.load(stream)
    manifest_rows = {
        _canonical_key(row["episode_key"]): row
        for row in manifest["val_episodes"]
    }
    missing_manifest = required_keys - set(manifest_rows)
    if missing_manifest:
        raise KeyError(f"metrics contain {len(missing_manifest)} keys absent from manifest")

    boundaries = {}
    metadata = {}
    raw_indices = {}
    for episode_key in sorted(required_keys):
        row = manifest_rows[episode_key]
        episode_dir = _episode_dir(data_root, episode_key)
        root = zarr.open(str(episode_dir), mode="r")
        valid = np.asarray(root["valid_indices"][:], dtype=bool)
        pose = np.asarray(root["pose_interp"][:], dtype=np.float64)
        gripper = np.asarray(root["gripper_widths"][:], dtype=np.float64).reshape(-1)
        common_length = min(len(valid), len(pose), len(gripper))
        valid_idx = np.where(valid[:common_length])[0][::stride]
        episode_gripper = gripper[valid_idx]
        translations = pose[valid_idx, :3, 3]
        boundaries[episode_key] = segment_episode_phases(
            episode_gripper,
            translations,
        )
        metadata[episode_key] = {
            "scene": row["scene"],
            "task": row["task"],
            "episode_dir": str(episode_dir),
            "video_path": str(episode_dir / "camera_1.mp4"),
        }
        raw_indices[episode_key] = valid_idx
    return boundaries, metadata, raw_indices


def _write_boundaries(
    output_dir: Path,
    boundaries: dict[str, PhaseBoundaries],
    metadata: dict[str, dict],
    raw_indices: dict[str, np.ndarray],
) -> None:
    rows = []
    for episode_key, boundary in sorted(boundaries.items()):
        row = {
            "episode_key": episode_key,
            "scene": metadata[episode_key]["scene"],
            "task": metadata[episode_key]["task"],
            **boundary.__dict__,
        }
        for name in (
            "pre_grasp_start",
            "grasp_start",
            "transport_start",
            "placement_start",
            "close_complete",
            "release_start",
            "release_complete",
        ):
            step = min(getattr(boundary, name), len(raw_indices[episode_key]) - 1)
            row[f"{name}_raw_frame"] = int(raw_indices[episode_key][step])
        rows.append(row)
    with open(output_dir / "phase_boundaries.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_step_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(row)
    return rows


def _aggregate(
    conditions: dict[str, Path],
    boundaries: dict[str, PhaseBoundaries],
    metadata: dict[str, dict],
) -> list[dict]:
    output_rows = []
    expected_identity = None
    for condition, path in conditions.items():
        rows = _read_step_rows(path)
        identity = [
            (
                _canonical_key(row["episode_key"]),
                int(row["sample_index"]),
                int(row["horizon_index"]),
            )
            for row in rows
        ]
        if expected_identity is None:
            expected_identity = identity
        elif identity != expected_identity:
            raise ValueError(
                f"step-metric rows for {condition} are not paired with the first condition"
            )

        buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            episode_key = _canonical_key(row["episode_key"])
            future_step = int(row["future_step"])
            phase = boundaries[episode_key].phase_at(future_step)
            scene = metadata[episode_key]["scene"]
            parsed = {
                "episode_key": episode_key,
                "pos_l2": float(row["pos_l2"]),
                "rot6d_l2": float(row["rot6d_l2"]),
                "rotation_geodesic_deg": float(row["rotation_geodesic_deg"]),
                "grip_l1": float(row["grip_l1"]),
                "grip_state_accuracy": float(row["grip_state_correct"]),
                "endpoint_success_10cm": float(row["pos_l2"]) <= 0.10,
                "rotation_success_20deg": float(row["rotation_geodesic_deg"]) <= 20.0,
                "combined_success_10cm_20deg_grip": (
                    float(row["pos_l2"]) <= 0.10
                    and float(row["rotation_geodesic_deg"]) <= 20.0
                    and bool(int(row["grip_state_correct"]))
                ),
                "is_chunk_endpoint": int(row["is_chunk_endpoint"]),
            }
            buckets[("all", phase)].append(parsed)
            buckets[(scene, phase)].append(parsed)

        for (scope, phase), bucket in sorted(buckets.items()):
            episode_buckets: dict[str, list[dict]] = defaultdict(list)
            for row in bucket:
                episode_buckets[row["episode_key"]].append(row)
            endpoints = [row for row in bucket if row["is_chunk_endpoint"]]
            endpoint_by_episode: dict[str, list[dict]] = defaultdict(list)
            for row in endpoints:
                endpoint_by_episode[row["episode_key"]].append(row)

            metric_values = {
                "pos_l2": [row["pos_l2"] for row in bucket],
                "rot6d_l2": [row["rot6d_l2"] for row in bucket],
                "rotation_geodesic_deg": [row["rotation_geodesic_deg"] for row in bucket],
                "grip_l1": [row["grip_l1"] for row in bucket],
                "grip_state_accuracy": [row["grip_state_accuracy"] for row in bucket],
                "final_displacement_error": [row["pos_l2"] for row in endpoints],
                "endpoint_success_10cm": [row["endpoint_success_10cm"] for row in endpoints],
                "rotation_success_20deg": [row["rotation_success_20deg"] for row in endpoints],
                "combined_success_10cm_20deg_grip": [
                    row["combined_success_10cm_20deg_grip"] for row in endpoints
                ],
            }
            result = {
                "condition": condition,
                "scope": scope,
                "phase": phase,
                "episodes": len(episode_buckets),
                "timestep_predictions": len(bucket),
                "endpoint_predictions": len(endpoints),
            }
            for metric, values in metric_values.items():
                result[f"mean_{metric}"] = float(np.mean(values)) if values else None
                source = endpoint_by_episode if metric in ENDPOINT_METRICS else episode_buckets
                per_episode = []
                for episode_rows in source.values():
                    episode_values = [
                        row["pos_l2"] if metric == "final_displacement_error" else row[metric]
                        for row in episode_rows
                    ]
                    if episode_values:
                        per_episode.append(float(np.mean(episode_values)))
                result[f"episode_balanced_{metric}"] = (
                    float(np.mean(per_episode)) if per_episode else None
                )
            output_rows.append(result)
    return output_rows


def _write_metrics(output_dir: Path, rows: list[dict]) -> None:
    with open(output_dir / "phase_metrics.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    all_rows = {
        (row["condition"], row["phase"]): row
        for row in rows
        if row["scope"] == "all"
    }
    conditions = []
    for row in rows:
        if row["condition"] not in conditions:
            conditions.append(row["condition"])
    with open(output_dir / "phase_metrics.md", "w") as stream:
        stream.write("# Execution-phase evaluation\n\n")
        stream.write(
            "Per-timestep errors are grouped by the phase of the future target "
            "timestep. Endpoint metrics are grouped by the phase of each chunk endpoint. "
            "Error metrics are lower-is-better; accuracy/success metrics are higher-is-better. "
            "All length errors are displayed in centimeters; phase_metrics.csv remains in meters.\n\n"
        )
        for metric in METRICS:
            report_name, scale = REPORT_METRICS[metric]
            stream.write(f"## {report_name}\n\n")
            stream.write("| Condition | " + " | ".join(PHASES) + " |\n")
            stream.write("|---|" + "---:|" * len(PHASES) + "\n")
            for condition in conditions:
                values = []
                for phase in PHASES:
                    value = all_rows[(condition, phase)][f"mean_{metric}"]
                    values.append("--" if value is None else f"{scale * value:.6f}")
                stream.write(f"| {condition} | " + " | ".join(values) + " |\n")
            stream.write("\n")

        if "bbox_visual" in conditions and "no_prompt" in conditions:
            stream.write("## BBox+Visual gain over No Prompt\n\n")
            stream.write("Positive values mean that BBox+Visual improves the metric.\n\n")
            stream.write("| Metric | " + " | ".join(PHASES) + " |\n")
            stream.write("|---|" + "---:|" * len(PHASES) + "\n")
            for metric in METRICS:
                gains = []
                for phase in PHASES:
                    baseline = all_rows[("no_prompt", phase)][f"mean_{metric}"]
                    prompted = all_rows[("bbox_visual", phase)][f"mean_{metric}"]
                    if metric in HIGHER_IS_BETTER:
                        gain = 100.0 * (prompted - baseline) / baseline
                    else:
                        gain = 100.0 * (baseline - prompted) / baseline
                    gains.append(f"{gain:+.2f}%")
                stream.write(f"| {metric} | " + " | ".join(gains) + " |\n")


def _make_contact_sheet(
    output_dir: Path,
    boundaries: dict[str, PhaseBoundaries],
    metadata: dict[str, dict],
    raw_indices: dict[str, np.ndarray],
    count: int,
    seed: int,
) -> None:
    if count <= 0:
        return
    rng = np.random.default_rng(seed)
    selected = []
    scenes = sorted({row["scene"] for row in metadata.values()})
    per_scene = max(1, int(np.ceil(count / max(1, len(scenes)))))
    for scene in scenes:
        candidates = sorted(
            key for key, row in metadata.items() if row["scene"] == scene
        )
        indices = rng.choice(
            len(candidates), size=min(per_scene, len(candidates)), replace=False
        )
        selected.extend(candidates[int(index)] for index in sorted(indices))
    selected = selected[:count]

    cell_h, cell_w = 180, 260
    label_h = 44
    sheet = np.full(
        (len(selected) * (cell_h + label_h), len(PHASES) * cell_w, 3),
        245,
        dtype=np.uint8,
    )
    for row_index, episode_key in enumerate(selected):
        boundary = boundaries[episode_key]
        ranges = [
            (0, boundary.pre_grasp_start),
            (boundary.pre_grasp_start, boundary.grasp_start),
            (boundary.grasp_start, boundary.transport_start),
            (boundary.transport_start, boundary.placement_start),
            (boundary.placement_start, boundary.n_steps),
        ]
        cap = cv2.VideoCapture(metadata[episode_key]["video_path"])
        try:
            for column, (phase, (start, stop)) in enumerate(zip(PHASES, ranges)):
                step = min(boundary.n_steps - 1, max(0, (start + max(start + 1, stop) - 1) // 2))
                raw_frame = int(raw_indices[episode_key][step])
                cap.set(cv2.CAP_PROP_POS_FRAMES, raw_frame)
                ok, frame = cap.read()
                if not ok:
                    frame = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                else:
                    frame = cv2.resize(frame, (cell_w, cell_h))
                y = row_index * (cell_h + label_h)
                x = column * cell_w
                sheet[y : y + cell_h, x : x + cell_w] = frame
                cv2.putText(
                    sheet,
                    f"{phase} s={step} f={raw_frame}",
                    (x + 4, y + cell_h + 17),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                if column == 0:
                    short_key = "/".join(episode_key.split("/")[:3])
                    cv2.putText(
                        sheet,
                        short_key[:38],
                        (x + 4, y + cell_h + 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
        finally:
            cap.release()
    cv2.imwrite(str(output_dir / "phase_boundary_contact_sheet.jpg"), sheet)


def main() -> None:
    args = parse_args()
    conditions = _parse_conditions(args.condition)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    first_rows = _read_step_rows(next(iter(conditions.values())))
    required_keys = {_canonical_key(row["episode_key"]) for row in first_rows}
    boundaries, metadata, raw_indices = _load_boundaries(
        Path(args.manifest),
        Path(args.data_root),
        args.stride,
        required_keys,
    )
    _write_boundaries(output_dir, boundaries, metadata, raw_indices)
    rows = _aggregate(conditions, boundaries, metadata)
    _write_metrics(output_dir, rows)
    _make_contact_sheet(
        output_dir,
        boundaries,
        metadata,
        raw_indices,
        args.diagnostic_episodes,
        args.seed,
    )

    detection_counts = defaultdict(int)
    for boundary in boundaries.values():
        detection_counts[boundary.detection] += 1
    summary = {
        "episodes": len(boundaries),
        "conditions": list(conditions),
        "phases": list(PHASES),
        "segmentation": {
            "signals": ["gripper_width", "ee_translation_speed", "episode_progress"],
            "detection_counts": dict(detection_counts),
        },
    }
    with open(output_dir / "summary.json", "w") as stream:
        json.dump(summary, stream, indent=2)
    print(
        f"[phase-eval] episodes={len(boundaries)} conditions={len(conditions)} "
        f"detection={dict(detection_counts)} output={output_dir}"
    )


if __name__ == "__main__":
    main()
