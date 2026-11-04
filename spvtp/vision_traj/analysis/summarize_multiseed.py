"""Summarize E5 training seeds and paired episode-bootstrap comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CONFIGS = {
    "no_prompt": "baseline_no_prompt_all_scenes",
    "target_only": "baseline_target_only_all_scenes",
    "visual_prompt": "baseline_vision_bbox_all_scenes",
    "bbox_prompt": "baseline_bbox_all_scenes",
    "bbox_visual": "baseline_vision_bbox_and_bbox_all_scenes",
}
TRAJECTORY_METRICS = (
    "pos_l2",
    "final_displacement_error",
    "rot6d_l2",
    "grip_l1",
)
SUCCESS_THRESHOLDS = {
    "success_at_2cm": 0.02,
    "success_at_5cm": 0.05,
    "success_at_10cm": 0.10,
}
SUCCESS_METRICS = tuple(SUCCESS_THRESHOLDS)
METRICS = (*TRAJECTORY_METRICS, *SUCCESS_METRICS)
REPORT_METRICS = {
    "pos_l2": ("Pos. L2 (cm)", 100.0, 4),
    "final_displacement_error": ("FDE (cm)", 100.0, 4),
    "rot6d_l2": ("Rot6D L2", 1.0, 6),
    "grip_l1": ("Grip L1 (cm)", 100.0, 4),
    "success_at_2cm": ("Success@2cm (%)", 100.0, 2),
    "success_at_5cm": ("Success@5cm (%)", 100.0, 2),
    "success_at_10cm": ("Success@10cm (%)", 100.0, 2),
}
COMPARISONS = (
    ("bbox_visual", "no_prompt"),
    ("bbox_visual", "visual_prompt"),
    ("bbox_visual", "bbox_prompt"),
    ("bbox_visual", "target_only"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-root",
        default="runs/vision_traj_experiments_final",
        help="root containing the original seed-42 runs",
    )
    parser.add_argument(
        "--multiseed-root",
        default="runs/vision_traj_experiments_final/multiseed",
    )
    parser.add_argument("--seeds", default="42,0,1")
    parser.add_argument(
        "--output-dir",
        default="runs/vision_traj_experiments_final/multiseed/summary",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def metrics_path(
    config: str,
    seed: int,
    legacy_root: Path,
    multiseed_root: Path,
) -> Path:
    run_dir = CONFIGS[config]
    root = legacy_root if seed == 42 else multiseed_root / f"seed_{seed}"
    return root / run_dir / "eval_metrics" / "all" / "metrics.csv"


def load_metrics(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["episode_key"].replace("\\", "/"), int(row["step"]))
            if key in records:
                raise ValueError(f"duplicate sample key {key} in {path}")
            values = {metric: float(row[metric]) for metric in TRAJECTORY_METRICS}
            fde = values["final_displacement_error"]
            values.update(
                {
                    metric: float(fde <= threshold)
                    for metric, threshold in SUCCESS_THRESHOLDS.items()
                }
            )
            records[key] = values
    if not records:
        raise ValueError(f"no metric rows in {path}")
    return records


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def hierarchical_episode_bootstrap(
    lhs_by_seed: dict[int, dict],
    rhs_by_seed: dict[int, dict],
    metric: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    seeds = sorted(lhs_by_seed)
    grouped = {}
    point_deltas = []
    for seed in seeds:
        lhs = lhs_by_seed[seed]
        rhs = rhs_by_seed[seed]
        if lhs.keys() != rhs.keys():
            missing_lhs = sorted(rhs.keys() - lhs.keys())[:3]
            missing_rhs = sorted(lhs.keys() - rhs.keys())[:3]
            raise ValueError(
                f"unaligned paired metrics for seed {seed}: "
                f"missing_lhs={missing_lhs} missing_rhs={missing_rhs}"
            )
        by_episode = defaultdict(list)
        for (episode_key, step), lhs_values in lhs.items():
            by_episode[episode_key].append(
                lhs_values[metric] - rhs[(episode_key, step)][metric]
            )
        grouped[seed] = {
            episode: np.asarray(values, dtype=np.float64)
            for episode, values in by_episode.items()
        }
        point_deltas.append(
            np.mean([value for values in grouped[seed].values() for value in values])
        )

    draws = np.empty(n_bootstrap, dtype=np.float64)
    for draw_idx in range(n_bootstrap):
        sampled_seed_ids = rng.choice(seeds, size=len(seeds), replace=True)
        seed_deltas = []
        for seed in sampled_seed_ids:
            episode_map = grouped[int(seed)]
            episodes = list(episode_map)
            sampled_episodes = rng.choice(episodes, size=len(episodes), replace=True)
            sampled_values = [episode_map[str(episode)] for episode in sampled_episodes]
            seed_deltas.append(np.concatenate(sampled_values).mean())
        draws[draw_idx] = float(np.mean(seed_deltas))

    point = float(np.mean(point_deltas))
    low, high = np.quantile(draws, [0.025, 0.975])
    return point, float(low), float(high)


def main() -> None:
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    legacy_root = Path(args.legacy_root)
    multiseed_root = Path(args.multiseed_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: dict[str, dict[int, dict]] = defaultdict(dict)
    seed_rows = []
    for config in CONFIGS:
        for seed in seeds:
            path = metrics_path(config, seed, legacy_root, multiseed_root)
            records = load_metrics(path)
            all_records[config][seed] = records
            row = {
                "config": config,
                "seed": seed,
                "samples": len(records),
                "metrics_path": str(path),
            }
            for metric in METRICS:
                row[metric] = float(np.mean([values[metric] for values in records.values()]))
            seed_rows.append(row)

    aggregate_rows = []
    for config in CONFIGS:
        row = {"config": config, "num_seeds": len(seeds)}
        for metric in METRICS:
            values = np.asarray(
                [
                    next(
                        seed_row[metric]
                        for seed_row in seed_rows
                        if seed_row["config"] == config and seed_row["seed"] == seed
                    )
                    for seed in seeds
                ],
                dtype=np.float64,
            )
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate_rows.append(row)

    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_rows = []
    for lhs, rhs in COMPARISONS:
        for metric in METRICS:
            point, low, high = hierarchical_episode_bootstrap(
                all_records[lhs],
                all_records[rhs],
                metric,
                args.bootstrap_samples,
                rng,
            )
            bootstrap_rows.append(
                {
                    "comparison": f"{lhs}_minus_{rhs}",
                    "metric": metric,
                    "delta": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "favors_lhs": bool(
                        high < 0.0 if metric in TRAJECTORY_METRICS else low > 0.0
                    ),
                }
            )

    write_csv(
        output_dir / "seed_metrics.csv",
        seed_rows,
        ["config", "seed", "samples", *METRICS, "metrics_path"],
    )
    aggregate_fields = ["config", "num_seeds"]
    for metric in METRICS:
        aggregate_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows, aggregate_fields)
    write_csv(
        output_dir / "paired_bootstrap.csv",
        bootstrap_rows,
        ["comparison", "metric", "delta", "ci95_low", "ci95_high", "favors_lhs"],
    )

    lines = [
        "# E5 Multi-seed Summary",
        "",
        f"Training seeds: {', '.join(map(str, seeds))}; split seed: 42.",
        "Values are mean ± sample standard deviation across training seeds. "
        "All length-error metrics are displayed in centimeters; machine-readable CSV/JSON values remain in meters.",
        "",
        "| Configuration | Pos. L2 (cm) | FDE (cm) | Rot6D L2 | Grip L1 (cm) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        cells = []
        for metric in TRAJECTORY_METRICS:
            _, scale, precision = REPORT_METRICS[metric]
            mean = row[f"{metric}_mean"] * scale
            std = row[f"{metric}_std"] * scale
            cells.append(f"{mean:.{precision}f} ± {std:.{precision}f}")
        lines.append(f"| {row['config']} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Endpoint Success uses each sample's final 3-D displacement and reports the fraction at or below each threshold.",
            "",
            "| Configuration | Success@2cm (%) | Success@5cm (%) | Success@10cm (%) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        cells = []
        for metric in SUCCESS_METRICS:
            _, scale, precision = REPORT_METRICS[metric]
            mean = row[f"{metric}_mean"] * scale
            std = row[f"{metric}_std"] * scale
            cells.append(f"{mean:.{precision}f} ± {std:.{precision}f}")
        lines.append(f"| {row['config']} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Paired 95% CIs use a hierarchical bootstrap over training seeds and validation episodes. "
            "A negative delta favors the first method for error metrics; a positive delta favors it for success metrics.",
            "",
            "| Comparison | Metric | Delta | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for row in bootstrap_rows:
        label, scale, precision = REPORT_METRICS[row["metric"]]
        lines.append(
            f"| {row['comparison']} | {label} | {row['delta'] * scale:.{precision}f} | "
            f"[{row['ci95_low'] * scale:.{precision}f}, "
            f"{row['ci95_high'] * scale:.{precision}f}] |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "split_seed": 42,
                "bootstrap_samples": args.bootstrap_samples,
                "aggregate_metrics": aggregate_rows,
                "paired_bootstrap": bootstrap_rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[multiseed summary] wrote {output_dir}")


if __name__ == "__main__":
    main()
