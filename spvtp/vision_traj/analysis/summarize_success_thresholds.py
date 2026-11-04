"""Compute endpoint-success rates from existing per-sample FDE files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CONFIGS = {
    "No Prompt": "baseline_no_prompt_all_scenes",
    "Target Only": "baseline_target_only_all_scenes",
    "Visual Prompt": "baseline_vision_bbox_all_scenes",
    "BBox Prompt": "baseline_bbox_all_scenes",
    "BBox+Visual": "baseline_vision_bbox_and_bbox_all_scenes",
}
THRESHOLDS_CM = (2, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs/vision_traj_experiments_final")
    parser.add_argument(
        "--output-dir",
        default="runs/vision_traj_experiments_final/success_thresholds",
    )
    return parser.parse_args()


def metric_path(root: Path, run_name: str, seed: int, split: str) -> Path:
    if seed == 42:
        run_root = root / run_name
    else:
        run_root = root / "multiseed" / f"seed_{seed}" / run_name
    return run_root / "eval_metrics" / split / "metrics.csv"


def read_fde(path: Path) -> np.ndarray:
    with open(path, newline="") as handle:
        return np.asarray(
            [float(row["final_displacement_error"]) for row in csv.DictReader(handle)],
            dtype=np.float64,
        )


def main() -> None:
    args = parse_args()
    root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, run_name in CONFIGS.items():
        for seed in (42, 0, 1):
            path = metric_path(root, run_name, seed, "all")
            if not path.exists():
                continue
            fde = read_fde(path)
            row = {
                "configuration": label,
                "seed": seed,
                "samples": len(fde),
                "mean_fde_cm": float(fde.mean() * 100.0),
            }
            for threshold in THRESHOLDS_CM:
                row[f"success_at_{threshold}cm"] = float(
                    (fde <= threshold / 100.0).mean() * 100.0
                )
            rows.append(row)

    summary = []
    for label in CONFIGS:
        group = [row for row in rows if row["configuration"] == label]
        if not group:
            continue
        item = {"configuration": label, "seeds": [row["seed"] for row in group]}
        for field in ["mean_fde_cm", *(f"success_at_{t}cm" for t in THRESHOLDS_CM)]:
            values = np.asarray([row[field] for row in group])
            item[field] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        summary.append(item)

    with open(output_dir / "success_thresholds.json", "w") as handle:
        json.dump({"per_seed": rows, "summary": summary}, handle, indent=2)
    with open(output_dir / "success_thresholds.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Endpoint success thresholds",
        "",
        "Success is the fraction of validation chunks whose final 3D position "
        "error is no larger than the stated threshold. Values are percentages "
        "and aggregate all currently completed training seeds.",
        "",
        "| Configuration | Seeds | FDE (cm) | Success@2cm | Success@5cm | Success@10cm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        def fmt(field: str) -> str:
            value = item[field]
            return f"{value['mean']:.2f} ± {value['std']:.2f}"

        lines.append(
            f"| {item['configuration']} | {len(item['seeds'])} | {fmt('mean_fde_cm')} | "
            f"{fmt('success_at_2cm')} | {fmt('success_at_5cm')} | "
            f"{fmt('success_at_10cm')} |"
        )
    with open(output_dir / "success_thresholds.md", "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
