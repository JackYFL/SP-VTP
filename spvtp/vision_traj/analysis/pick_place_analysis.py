"""Compare trajectory metrics during pick and place portions of each episode.

This analysis joins existing per-sample evaluation metrics with the per-video
EE/gripper landmarks produced by ``ee_pick_place_boundaries.py``.  It reports
both a high-coverage split at the transport apex and contact-centered windows.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import zarr


BASE_METRICS = (
    "pos_l2",
    "final_displacement_error",
    "rot6d_l2",
    "grip_l1",
)
SUCCESS_THRESHOLDS = (0.02, 0.05, 0.10)
ANALYSES = ("task_half", "contact_window")
PHASES = ("pick", "place")


def _canonical_key(value: str) -> str:
    return value.replace("\\", "/")


def _read_csv(path: Path) -> list[dict]:
    with open(path, "r", newline="") as stream:
        return list(csv.DictReader(stream))


def _parse_conditions(items: list[str]) -> dict[str, Path]:
    conditions: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"condition must be NAME=METRICS.csv, got {item!r}")
        name, path = item.split("=", maxsplit=1)
        if not name or name in conditions:
            raise ValueError(f"invalid or duplicate condition name: {name!r}")
        conditions[name] = Path(path)
    if not conditions:
        raise ValueError("at least one --condition is required")
    return conditions


def _load_boundaries(path: Path) -> dict[str, dict]:
    boundaries = {}
    for row in _read_csv(path):
        key = _canonical_key(row["episode_key"])
        boundaries[key] = {
            "pick_raw_frame": int(row["pick_raw_frame"]),
            "split_raw_frame": int(row["split_raw_frame"]),
            "place_raw_frame": int(row["place_raw_frame"]),
            "video_path": row["video_path"],
            "n_steps": int(row["n_steps"]),
            "quality": row["quality"],
            "gripper_used": row["gripper_used"].lower() == "true",
        }
    return boundaries


def _metric_values(row: dict) -> dict[str, float]:
    values = {metric: float(row[metric]) for metric in BASE_METRICS}
    fde = values["final_displacement_error"]
    for threshold in SUCCESS_THRESHOLDS:
        values[f"success_at_{int(round(threshold * 100))}cm"] = float(fde <= threshold)
    return values


def _assignments(raw_frame: int, boundary: dict, contact_window: int) -> list[tuple[str, str]]:
    assignments = [
        (
            "task_half",
            "pick" if raw_frame < boundary["split_raw_frame"] else "place",
        )
    ]
    pick_distance = abs(raw_frame - boundary["pick_raw_frame"])
    place_distance = abs(raw_frame - boundary["place_raw_frame"])
    if min(pick_distance, place_distance) <= contact_window:
        assignments.append(
            (
                "contact_window",
                "pick" if pick_distance <= place_distance else "place",
            )
        )
    return assignments


def _load_condition_rows(
    path: Path,
    boundaries: dict[str, dict],
    contact_window: int,
    endpoint_cache: dict[str, np.ndarray],
    stride: int,
    action_horizon: int,
    boundary_filter: str,
) -> list[dict]:
    output = []
    missing = set()
    for row in _read_csv(path):
        episode_key = _canonical_key(row["episode_key"])
        boundary = boundaries.get(episode_key)
        if boundary is None:
            missing.add(episode_key)
            continue
        if boundary_filter == "high" and boundary["quality"] != "high":
            continue
        if boundary_filter == "gripper" and not boundary["gripper_used"]:
            continue
        identity = (episode_key, int(row["step"]))
        values = _metric_values(row)
        endpoint_frames = endpoint_cache.get(episode_key)
        if endpoint_frames is None:
            episode_dir = Path(boundary["video_path"]).parent
            root = zarr.open(str(episode_dir), mode="r")
            valid = np.asarray(root["valid_indices"][:], dtype=bool)
            pose_length = len(root["pose_interp"])
            gripper_length = len(root["gripper_widths"])
            common_length = min(len(valid), pose_length, gripper_length)
            endpoint_frames = np.flatnonzero(valid[:common_length])[::stride]
            endpoint_cache[episode_key] = endpoint_frames
        endpoint_step = int(row["step"]) + action_horizon
        if endpoint_step >= len(endpoint_frames):
            raise IndexError(
                f"endpoint step {endpoint_step} exceeds {episode_key} length "
                f"{len(endpoint_frames)}"
            )
        endpoint_raw_frame = int(endpoint_frames[endpoint_step])
        for analysis, phase in _assignments(
            endpoint_raw_frame, boundary, contact_window
        ):
            output.append(
                {
                    "identity": identity,
                    "episode_key": episode_key,
                    "analysis": analysis,
                    "phase": phase,
                    "endpoint_raw_frame": endpoint_raw_frame,
                    "boundary_quality": boundary["quality"],
                    "gripper_used": boundary["gripper_used"],
                    **values,
                }
            )
    if missing:
        raise KeyError(f"{path} contains {len(missing)} episodes absent from boundaries")
    return output


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    chunk = 512
    means = []
    for start in range(0, iterations, chunk):
        count = min(chunk, iterations - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means.append(np.mean(values[indices], axis=1))
    samples = np.concatenate(means)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _aggregate_condition(
    condition: str,
    rows: list[dict],
    rng: np.random.Generator,
    bootstrap_iterations: int,
) -> tuple[list[dict], list[dict]]:
    metric_names = BASE_METRICS + tuple(
        f"success_at_{int(round(threshold * 100))}cm"
        for threshold in SUCCESS_THRESHOLDS
    )
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["analysis"], row["phase"])].append(row)

    aggregate_rows = []
    episode_values: dict[tuple[str, str, str], dict[str, float]] = {}
    for analysis in ANALYSES:
        for phase in PHASES:
            bucket = grouped[(analysis, phase)]
            by_episode: dict[str, list[dict]] = defaultdict(list)
            for row in bucket:
                by_episode[row["episode_key"]].append(row)
            result = {
                "condition": condition,
                "analysis": analysis,
                "phase": phase,
                "samples": len(bucket),
                "episodes": len(by_episode),
                "high_quality_samples": sum(
                    row["boundary_quality"] == "high" for row in bucket
                ),
                "gripper_used_samples": sum(row["gripper_used"] for row in bucket),
            }
            for metric in metric_names:
                sample_values = np.asarray([row[metric] for row in bucket])
                result[f"mean_{metric}"] = float(np.mean(sample_values))
                per_episode = []
                for episode_key, episode_rows in by_episode.items():
                    value = float(np.mean([row[metric] for row in episode_rows]))
                    episode_values[(analysis, phase, episode_key, metric)] = value
                    per_episode.append(value)
                result[f"episode_balanced_{metric}"] = float(np.mean(per_episode))
            aggregate_rows.append(result)

    comparison_rows = []
    for analysis in ANALYSES:
        pick_episodes = {
            key[2]
            for key in episode_values
            if key[0] == analysis and key[1] == "pick"
        }
        place_episodes = {
            key[2]
            for key in episode_values
            if key[0] == analysis and key[1] == "place"
        }
        paired_episodes = sorted(pick_episodes & place_episodes)
        for metric in metric_names:
            pick_values = np.asarray(
                [episode_values[(analysis, "pick", episode, metric)] for episode in paired_episodes]
            )
            place_values = np.asarray(
                [episode_values[(analysis, "place", episode, metric)] for episode in paired_episodes]
            )
            differences = place_values - pick_values
            ci_low, ci_high = _bootstrap_mean_ci(
                differences, rng, bootstrap_iterations
            )
            pick_mean = float(np.mean(pick_values))
            place_mean = float(np.mean(place_values))
            comparison_rows.append(
                {
                    "condition": condition,
                    "analysis": analysis,
                    "metric": metric,
                    "paired_episodes": len(paired_episodes),
                    "pick_episode_mean": pick_mean,
                    "place_episode_mean": place_mean,
                    "place_minus_pick": place_mean - pick_mean,
                    "place_over_pick": (
                        place_mean / pick_mean if abs(pick_mean) > 1e-12 else float("nan")
                    ),
                    "paired_bootstrap_ci95_low": ci_low,
                    "paired_bootstrap_ci95_high": ci_high,
                }
            )
    return aggregate_rows, comparison_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _find_comparison(rows: list[dict], condition: str, analysis: str, metric: str) -> dict:
    return next(
        row
        for row in rows
        if row["condition"] == condition
        and row["analysis"] == analysis
        and row["metric"] == metric
    )


def _write_report(
    path: Path,
    conditions: list[str],
    comparison_rows: list[dict],
    contact_window: int,
    boundary_filter: str,
) -> None:
    with open(path, "w") as stream:
        stream.write("# Pick vs. place trajectory analysis\n\n")
        stream.write(f"Boundary subset: `{boundary_filter}`.\n\n")
        stream.write(
            "`task_half` assigns a prediction by whether its chunk endpoint is "
            "before or after the EE transport apex. `contact_window` compares chunk "
            f"endpoints within ±{contact_window} raw video frames of the detected pick/place "
            "contacts. FDE and endpoint success correspond exactly to that endpoint; "
            "Pos/Rot/Grip remain whole-chunk means and may straddle a boundary. Values are "
            "episode-balanced; confidence intervals are paired "
            "episode bootstrap intervals for place minus pick.\n\n"
        )
        for analysis in ANALYSES:
            stream.write(f"## {analysis}\n\n")
            stream.write(
                "| Condition | Pick FDE (cm) | Place FDE (cm) | Δ (cm) [95% CI] | "
                "Pick Pos (cm) | Place Pos (cm) | Pick S@10 | Place S@10 |\n"
            )
            stream.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for condition in conditions:
                fde = _find_comparison(
                    comparison_rows, condition, analysis, "final_displacement_error"
                )
                pos = _find_comparison(comparison_rows, condition, analysis, "pos_l2")
                success = _find_comparison(
                    comparison_rows, condition, analysis, "success_at_10cm"
                )
                stream.write(
                    f"| {condition} | {100*fde['pick_episode_mean']:.2f} | "
                    f"{100*fde['place_episode_mean']:.2f} | "
                    f"{100*fde['place_minus_pick']:+.2f} "
                    f"[{100*fde['paired_bootstrap_ci95_low']:+.2f}, "
                    f"{100*fde['paired_bootstrap_ci95_high']:+.2f}] | "
                    f"{100*pos['pick_episode_mean']:.2f} | "
                    f"{100*pos['place_episode_mean']:.2f} | "
                    f"{100*success['pick_episode_mean']:.1f}% | "
                    f"{100*success['place_episode_mean']:.1f}% |\n"
                )
            stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundaries", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        metavar="NAME=METRICS.csv",
    )
    parser.add_argument("--contact-window", type=int, default=15)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument(
        "--boundary-filter",
        choices=("all", "high", "gripper"),
        default="all",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = _parse_conditions(args.condition)
    boundaries = _load_boundaries(Path(args.boundaries))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    aggregate_rows = []
    comparison_rows = []
    expected_identities = None
    endpoint_cache: dict[str, np.ndarray] = {}
    for condition, path in conditions.items():
        rows = _load_condition_rows(
            path,
            boundaries,
            args.contact_window,
            endpoint_cache,
            args.stride,
            args.action_horizon,
            args.boundary_filter,
        )
        identities = {
            row["identity"] for row in rows if row["analysis"] == "task_half"
        }
        if expected_identities is None:
            expected_identities = identities
        elif identities != expected_identities:
            raise ValueError(
                f"{condition} is not paired with the first condition: "
                f"{len(identities)} vs {len(expected_identities)} samples"
            )
        aggregate, comparison = _aggregate_condition(
            condition, rows, rng, args.bootstrap_iterations
        )
        aggregate_rows.extend(aggregate)
        comparison_rows.extend(comparison)

    _write_csv(output_dir / "pick_place_metrics.csv", aggregate_rows)
    _write_csv(output_dir / "pick_vs_place_comparisons.csv", comparison_rows)
    _write_report(
        output_dir / "report.md",
        list(conditions),
        comparison_rows,
        args.contact_window,
        args.boundary_filter,
    )
    print(
        f"[pick-place] conditions={len(conditions)} "
        f"paired_samples={len(expected_identities or [])} output={output_dir}"
    )


if __name__ == "__main__":
    main()
