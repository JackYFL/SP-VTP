"""Summarize prompt-intervention metrics overall and by scene."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "pos_l2",
    "final_displacement_error",
    "rot6d_l2",
    "grip_l1",
    "rotation_geodesic_deg",
    "grip_state_accuracy",
    "combined_success_5cm_20deg_grip",
    "combined_success_10cm_20deg_grip",
)
SUMMARY_KEYS = {
    "pos_l2": "mean_pos_l2",
    "final_displacement_error": "mean_final_displacement_error",
    "rot6d_l2": "mean_rot6d_l2",
    "grip_l1": "mean_grip_l1",
    "rotation_geodesic_deg": "mean_rotation_geodesic_deg",
    "grip_state_accuracy": "mean_grip_state_accuracy",
    "combined_success_5cm_20deg_grip": "combined_success_5cm_20deg_grip",
    "combined_success_10cm_20deg_grip": "combined_success_10cm_20deg_grip",
}

CONDITION_ORDER = (
    "correct_prompt",
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
    "shuffled_object",
    "shuffled_target",
    "role_swap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervention-root", required=True)
    parser.add_argument("--clean-root", required=True)
    return parser.parse_args()


def _scene_from_key(key: str) -> str:
    return key.replace("\\", "/").split("/", maxsplit=1)[0]


def _read_json(path: Path) -> dict:
    with open(path, "r") as stream:
        return json.load(stream)


def _write_scene_summaries(condition_dir: Path) -> None:
    all_dir = condition_dir / "all"
    overall = _read_json(all_dir / "summary.json")
    by_scene: dict[str, list[dict]] = defaultdict(list)
    with open(all_dir / "metrics.csv", "r", newline="") as stream:
        for row in csv.DictReader(stream):
            by_scene[_scene_from_key(row["episode_key"])].append(row)

    for scene, rows in sorted(by_scene.items()):
        scene_dir = condition_dir / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "split": overall["split"],
            "samples": len(rows),
            "split_seed": overall["split_seed"],
            "val_ratio": overall["val_ratio"],
            "sample_seed": overall["sample_seed"],
            "prompt_intervention": overall["prompt_intervention"],
            "intervention_seed": overall["intervention_seed"],
        }
        for metric in METRICS:
            if metric not in rows[0]:
                continue
            summary[SUMMARY_KEYS[metric]] = float(
                np.mean([float(row[metric]) for row in rows])
            )
        with open(scene_dir / "summary.json", "w") as stream:
            json.dump(summary, stream, indent=2)


def _summaries_from_all(condition_root: Path) -> dict[str, dict]:
    """Build all/scene summaries from one globally indexed inference pass."""
    overall = _read_json(condition_root / "all" / "summary.json")
    summaries = {"all": overall}
    by_scene: dict[str, list[dict]] = defaultdict(list)
    with open(condition_root / "all" / "metrics.csv", "r", newline="") as stream:
        for row in csv.DictReader(stream):
            by_scene[_scene_from_key(row["episode_key"])].append(row)
    for scene, scene_rows in by_scene.items():
        summary = {"samples": len(scene_rows)}
        for metric in METRICS:
            if metric not in scene_rows[0]:
                continue
            summary[SUMMARY_KEYS[metric]] = float(
                np.mean([float(row[metric]) for row in scene_rows])
            )
        summaries[scene] = summary
    return summaries


def main() -> None:
    args = parse_args()
    intervention_root = Path(args.intervention_root)
    clean_root = Path(args.clean_root)

    condition_dirs = sorted(
        path
        for path in intervention_root.iterdir()
        if path.is_dir() and (path / "all" / "summary.json").is_file()
    )
    for condition_dir in condition_dirs:
        _write_scene_summaries(condition_dir)

    rows = []
    conditions = [("correct_prompt", clean_root)]
    conditions.extend((path.name, path) for path in condition_dirs if path.name != "none")
    for condition, condition_root in conditions:
        summaries = _summaries_from_all(condition_root)
        for split in ("all", "scene1", "scene2", "scene3"):
            if split not in summaries:
                continue
            summary = summaries[split]
            row = {"condition": condition, "split": split}
            for summary_key in SUMMARY_KEYS.values():
                if summary_key in summary:
                    row[summary_key] = summary[summary_key]
            rows.append(row)

    output_csv = intervention_root / "comparison.csv"
    fieldnames = ["condition", "split", *SUMMARY_KEYS.values()]
    with open(output_csv, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    all_rows = {row["condition"]: row for row in rows if row["split"] == "all"}
    clean = all_rows["correct_prompt"]
    output_md = intervention_root / "comparison.md"
    with open(output_md, "w") as stream:
        stream.write("# Prompt intervention results\n\n")
        stream.write(
            "All metrics are computed on the same validation split; lower is better. "
            "All length-error metrics are displayed in centimeters; raw CSV/JSON values remain in meters.\n\n"
        )
        stream.write("| Condition | Pos. L2 (cm) | FDE (cm) | Rot6D L2 | Grip L1 (cm) | FDE change vs. correct |\n")
        stream.write("|---|---:|---:|---:|---:|---:|\n")
        ordered_conditions = [
            condition for condition in CONDITION_ORDER if condition in all_rows
        ]
        ordered_conditions.extend(
            condition
            for condition in sorted(all_rows)
            if condition not in ordered_conditions
        )
        for condition in ordered_conditions:
            if condition not in all_rows:
                continue
            row = all_rows[condition]
            fde = float(row["mean_final_displacement_error"])
            clean_fde = float(clean["mean_final_displacement_error"])
            relative = 100.0 * (fde - clean_fde) / clean_fde
            stream.write(
                f"| {condition} | {100.0 * float(row['mean_pos_l2']):.4f} "
                f"| {100.0 * fde:.4f} | {float(row['mean_rot6d_l2']):.6f} "
                f"| {100.0 * float(row['mean_grip_l1']):.4f} | {relative:+.2f}% |\n"
            )

        task_metric_keys = {
            "mean_rotation_geodesic_deg",
            "mean_grip_state_accuracy",
            "combined_success_5cm_20deg_grip",
            "combined_success_10cm_20deg_grip",
        }
        if all(task_metric_keys <= row.keys() for row in all_rows.values()):
            stream.write("\n## Offline task-oriented metrics\n\n")
            stream.write(
                "Gripper state uses a 3 cm width threshold. Combined success requires "
                "endpoint translation, endpoint rotation $\\leq20^\\circ$, and the correct "
                "endpoint gripper state.\n\n"
            )
            stream.write(
                "| Condition | Rotation geo. (deg) | Grip state acc. | "
                "Combined@5cm | Combined@10cm |\n"
            )
            stream.write("|---|---:|---:|---:|---:|\n")
            for condition in ordered_conditions:
                row = all_rows[condition]
                stream.write(
                    f"| {condition} | {float(row['mean_rotation_geodesic_deg']):.2f} "
                    f"| {100.0 * float(row['mean_grip_state_accuracy']):.2f}% "
                    f"| {100.0 * float(row['combined_success_5cm_20deg_grip']):.2f}% "
                    f"| {100.0 * float(row['combined_success_10cm_20deg_grip']):.2f}% |\n"
                )

    print(f"[prompt-intervention] wrote {output_csv} and {output_md}")


if __name__ == "__main__":
    main()
