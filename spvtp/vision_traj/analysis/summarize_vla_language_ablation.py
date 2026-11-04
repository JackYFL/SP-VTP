"""Summarize the inference-time VLA language ablation across seeds and scenes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MODELS = {
    "smolvla": {
        "label": "SmolVLA",
        "run": "baseline_smolvla_language_only_all_scenes",
        "step": 10_000,
        "seeds": (42, 0, 1),
    },
    "openvla": {
        "label": "OpenVLA-OFT",
        "run": "baseline_openvla_oft_language_only_all_scenes",
        "step": 5_000,
        "seeds": (42, 0, 1),
    },
    "openvla_no_history": {
        "label": "OpenVLA-OFT No History",
        "run": "baseline_openvla_oft_language_no_history_all_scenes",
        "step": 5_000,
        "seeds": (42,),
    },
    "groot": {
        "label": "GR00T N1.6",
        "run": "baseline_groot_n1_6_language_only_all_scenes",
        "step": 5_000,
        "seeds": (42, 0, 1),
    },
    "pi05": {
        "label": "π0.5 (generic language + Visual Prompt)",
        "run": "baseline_pi05_all_scenes",
        "step": 10_000,
        "seeds": (42, 0, 1),
        "object_id_in_language": False,
    },
}
SCENES = (("scene1", 4_333), ("scene2", 3_079), ("scene3", 2_612))
ERROR_METRICS = ("pos_l2", "final_displacement_error", "rot6d_l2", "grip_l1")
SUCCESS_THRESHOLDS = {
    "success_at_2cm": 0.02,
    "success_at_5cm": 0.05,
    "success_at_10cm": 0.10,
}
METRICS = (*ERROR_METRICS, *SUCCESS_THRESHOLDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root", default="runs/vision_traj_experiments_final"
    )
    parser.add_argument(
        "--ablation-root",
        default=(
            "runs/vision_traj_experiments_final/"
            "language_prompt_ablations/no_object_id"
        ),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def original_run(runs_root: Path, run: str, seed: int) -> Path:
    if seed == 42:
        return runs_root / run
    return runs_root / "multiseed_vla" / f"seed_{seed}" / run


def metrics_path(run_root: Path, step: int) -> Path:
    return run_root / "eval_metrics" / f"step_{step:06d}" / "metrics.csv"


def load_metrics(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(float(row["sample_index"])))
    arrays = {
        metric: np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        for metric in ERROR_METRICS
    }
    fde = arrays["final_displacement_error"]
    arrays.update(
        {
            metric: (fde <= threshold).astype(np.float64)
            for metric, threshold in SUCCESS_THRESHOLDS.items()
        }
    )
    expected = sum(count for _, count in SCENES)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} samples in {path}, found {len(rows)}")
    return arrays


def means(arrays: dict[str, np.ndarray], start: int = 0, end: int | None = None) -> dict:
    return {metric: float(arrays[metric][start:end].mean()) for metric in METRICS}


def aggregate(rows: list[dict], field: str) -> dict:
    result = {}
    for metric in METRICS:
        values = np.asarray([row[field][metric] for row in rows], dtype=np.float64)
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return result


def fmt(stats: dict, metric: str, *, delta: bool = False) -> str:
    scale = 100.0 if metric != "rot6d_l2" else 1.0
    precision = 2 if metric.startswith("success_") else (6 if metric == "rot6d_l2" else 4)
    mean = stats[metric]["mean"] * scale
    std = stats[metric]["std"] * scale
    prefix = "+" if delta and mean > 0 else ""
    if std == 0.0:
        return f"{prefix}{mean:.{precision}f}"
    return f"{prefix}{mean:.{precision}f} ± {std:.{precision}f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    ablation_root = Path(args.ablation_root)
    output_dir = ablation_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_payload = []
    seed_csv = []
    scene_csv = []
    aggregate_payload = {}
    for method, spec in MODELS.items():
        method_rows = []
        for seed in spec["seeds"]:
            original_path = metrics_path(
                original_run(runs_root, spec["run"], seed), spec["step"]
            )
            ablation_applicable = spec.get("object_id_in_language", True)
            ablated_path = (
                metrics_path(ablation_root / method / f"seed_{seed}", spec["step"])
                if ablation_applicable
                else original_path
            )
            if not ablated_path.is_file() and args.allow_incomplete:
                continue
            original = load_metrics(original_path)
            ablated = load_metrics(ablated_path)
            original_all = means(original)
            ablated_all = means(ablated)
            delta_all = {
                metric: ablated_all[metric] - original_all[metric]
                for metric in METRICS
            }
            row = {
                "method": method,
                "label": spec["label"],
                "seed": seed,
                "samples": len(ablated["pos_l2"]),
                "ablation_applicable": ablation_applicable,
                "original": original_all,
                "no_object_id": ablated_all,
                "delta": delta_all,
                "scenes": {},
                "original_metrics": str(original_path),
                "ablated_metrics": str(ablated_path),
            }
            start = 0
            for scene, count in SCENES:
                end = start + count
                original_scene = means(original, start, end)
                ablated_scene = means(ablated, start, end)
                delta_scene = {
                    metric: ablated_scene[metric] - original_scene[metric]
                    for metric in METRICS
                }
                row["scenes"][scene] = {
                    "samples": count,
                    "original": original_scene,
                    "no_object_id": ablated_scene,
                    "delta": delta_scene,
                }
                scene_row = {
                    "method": method,
                    "seed": seed,
                    "scene": scene,
                    "samples": count,
                }
                for condition, values in (
                    ("original", original_scene),
                    ("no_object_id", ablated_scene),
                    ("delta", delta_scene),
                ):
                    for metric in METRICS:
                        scene_row[f"{condition}_{metric}"] = values[metric]
                scene_csv.append(scene_row)
                start = end
            seed_payload.append(row)
            method_rows.append(row)
            flat = {
                "method": method,
                "seed": seed,
                "samples": row["samples"],
            }
            for condition in ("original", "no_object_id", "delta"):
                for metric in METRICS:
                    flat[f"{condition}_{metric}"] = row[condition][metric]
            seed_csv.append(flat)

        if not method_rows:
            continue
        aggregate_payload[method] = {
            "label": spec["label"],
            "seeds": [row["seed"] for row in method_rows],
            "ablation_applicable": method_rows[0]["ablation_applicable"],
            "original": aggregate(method_rows, "original"),
            "no_object_id": aggregate(method_rows, "no_object_id"),
            "delta": aggregate(method_rows, "delta"),
            "scenes": {},
        }
        for scene, _ in SCENES:
            scene_rows = [
                {
                    condition: row["scenes"][scene][condition]
                    for condition in ("original", "no_object_id", "delta")
                }
                for row in method_rows
            ]
            aggregate_payload[method]["scenes"][scene] = {
                condition: aggregate(scene_rows, condition)
                for condition in ("original", "no_object_id", "delta")
            }

    payload = {
        "protocol": {
            "intervention": "remove trailing object instance ID at inference only",
            "example_original": "pick fork3 and place to plate2",
            "example_ablated": "pick fork and place to plate2",
            "target_instance_id_retained": True,
            "training_checkpoint_unchanged": True,
            "samples_per_seed": sum(count for _, count in SCENES),
        },
        "aggregate": aggregate_payload,
        "per_seed": seed_payload,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    if seed_csv:
        write_csv(output_dir / "seed_comparison.csv", seed_csv)
    if scene_csv:
        write_csv(output_dir / "scene_comparison.csv", scene_csv)

    lines = [
        "# VLA Language Ablation: No Object Instance ID",
        "",
        "At inference only, `pick fork3 and place to plate2` becomes "
        "`pick fork and place to plate2`. Target IDs, observations, history, "
        "checkpoints, split, and evaluation samples are unchanged.",
        "",
        "| Model | Seeds | Original FDE (cm) | No-ID FDE (cm) | ΔFDE (cm) | Original S@10 | No-ID S@10 | ΔS@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in MODELS:
        if method not in aggregate_payload:
            continue
        row = aggregate_payload[method]
        if row["ablation_applicable"]:
            delta_fde = fmt(row["delta"], "final_displacement_error", delta=True)
            delta_s10 = f"{fmt(row['delta'], 'success_at_10cm', delta=True)} points"
            no_id_fde = fmt(row["no_object_id"], "final_displacement_error")
            no_id_s10 = f"{fmt(row['no_object_id'], 'success_at_10cm')}%"
        else:
            delta_fde = "N/A"
            delta_s10 = "N/A"
            no_id_fde = "same input"
            no_id_s10 = "same input"
        lines.append(
            f"| {row['label']} | {len(row['seeds'])} | "
            f"{fmt(row['original'], 'final_displacement_error')} | {no_id_fde} | "
            f"{delta_fde} | {fmt(row['original'], 'success_at_10cm')}% | "
            f"{no_id_s10} | {delta_s10} |"
        )
    lines.extend(
        [
            "",
            "| Model | Pos L2 (cm) | FDE (cm) | Rot6D L2 | Grip L1 (cm) | S@2/5/10cm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in MODELS:
        if method not in aggregate_payload:
            continue
        row = aggregate_payload[method]
        no_id = row["no_object_id"]
        lines.append(
            f"| {row['label']} | {fmt(no_id, 'pos_l2')} | "
            f"{fmt(no_id, 'final_displacement_error')} | "
            f"{fmt(no_id, 'rot6d_l2')} | {fmt(no_id, 'grip_l1')} | "
            f"{fmt(no_id, 'success_at_2cm')} / {fmt(no_id, 'success_at_5cm')} / "
            f"{fmt(no_id, 'success_at_10cm')}% |"
        )
    lines.extend(
        [
            "",
            "| Model | Scene | No-ID FDE (cm) | ΔFDE (cm) | No-ID S@10 | ΔS@10 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for method in MODELS:
        if method not in aggregate_payload:
            continue
        row = aggregate_payload[method]
        for scene, _ in SCENES:
            scene_row = row["scenes"][scene]
            delta_fde = (
                fmt(scene_row["delta"], "final_displacement_error", delta=True)
                if row["ablation_applicable"]
                else "N/A"
            )
            delta_s10 = (
                f"{fmt(scene_row['delta'], 'success_at_10cm', delta=True)} points"
                if row["ablation_applicable"]
                else "N/A"
            )
            lines.append(
                f"| {row['label']} | {scene} | "
                f"{fmt(scene_row['no_object_id'], 'final_displacement_error')} | "
                f"{delta_fde} | "
                f"{fmt(scene_row['no_object_id'], 'success_at_10cm')}% | "
                f"{delta_s10} |"
            )
    lines.extend(
        [
            "",
            "Values are mean ± sample standard deviation across training seeds. "
            "For single-seed No-History, no cross-seed deviation is reported. "
            "π0.5 already uses generic language with no instance ID, so its existing "
            "three-seed result is included as an unchanged reference rather than a new intervention.",
        ]
    )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"[VLA language ablation] methods={len(aggregate_payload)} output={output_dir}")


if __name__ == "__main__":
    main()
