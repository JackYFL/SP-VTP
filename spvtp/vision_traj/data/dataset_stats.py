"""Statistical report for the vision trajectory dataset.

Example:

    python -m spvtp.vision_traj.data.dataset_stats \
        --data-root spvtp/output_data \
        --annotations annotations_merged.json \
        --out-dir runs/vision_traj_dataset_stats
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from omegaconf import OmegaConf

from spvtp.vision_traj.data.dataset import (
    EpisodeRecord,
    VisionTrajDataset,
    build_episode_index,
    filter_episodes_by_scenes,
    resolve_split_seed,
    split_episodes,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--annotations", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--scenes", default=None, help="comma-separated scene names; default uses config/all")
    p.add_argument("--val-ratio", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--action-horizon", type=int, default=None)
    p.add_argument("--history-horizon", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--prompt-form", default=None)
    p.add_argument(
        "--max-action-samples",
        type=int,
        default=0,
        help="limit action-stat samples per split for quick debugging; 0 uses all",
    )
    p.add_argument(
        "--image-stats-samples",
        type=int,
        default=256,
        help="number of current frames sampled for RGB stats; 0 disables image reads",
    )
    p.add_argument("--outlier-top-k", type=int, default=50)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def _as_list(x) -> list[str] | None:
    if x is None:
        return None
    if isinstance(x, str):
        return [s.strip() for s in x.split(",") if s.strip()]
    return list(x)


def _describe(values: np.ndarray | list[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    qs = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p25": float(qs[2]),
        "median": float(qs[3]),
        "p75": float(qs[4]),
        "p95": float(qs[5]),
        "p99": float(qs[6]),
        "max": float(arr.max()),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@lru_cache(maxsize=None)
def _video_metadata_for_path(video_path: str) -> dict[str, int | float | None]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"video_width": None, "video_height": None, "video_frames": None, "video_fps": None}
    meta = {
        "video_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "video_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "video_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "video_fps": float(cap.get(cv2.CAP_PROP_FPS)),
    }
    cap.release()
    return meta


def _video_metadata(ep: EpisodeRecord) -> dict[str, int | float | None]:
    return dict(_video_metadata_for_path(str(ep.video_path)))


def _counts(
    datasets: dict[str, VisionTrajDataset],
    split_by_key: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rows = []
    for split_name, dataset in datasets.items():
        per_episode_samples = Counter(ei for ei, _ in dataset._samples)
        for ei, ep in enumerate(dataset.episodes):
            meta = _video_metadata(ep)
            episode_rows.append(
                {
                    "split": split_name,
                    "episode_index": ei,
                    "episode_key": ep.key,
                    "scene": ep.scene,
                    "task": ep.task,
                    "valid_frames_raw": ep.n_frames,
                    "valid_frames_strided": len(dataset._valid_raw_idx[ei]),
                    "samples": per_episode_samples.get(ei, 0),
                    **meta,
                }
            )

    count_rows = []
    for split_name, dataset in datasets.items():
        per_episode_samples = Counter(ei for ei, _ in dataset._samples)
        buckets: dict[tuple[str, str], dict[str, Any]] = {}
        for ei, ep in enumerate(dataset.episodes):
            for level, key in (
                ("scene", ep.scene),
                ("task", ep.task),
                ("scene_task", f"{ep.scene}/{ep.task}"),
            ):
                row = buckets.setdefault(
                    (level, key),
                    {"split": split_name, "level": level, "key": key, "episodes": 0, "samples": 0},
                )
                row["episodes"] += 1
                row["samples"] += per_episode_samples.get(ei, 0)
        count_rows.extend(sorted(buckets.values(), key=lambda r: (r["level"], r["key"])))

    split_rows = [
        {"episode_key": ep.key, "scene": ep.scene, "task": ep.task, "split": split_by_key[ep.key]}
        for dataset in [datasets["all"]]
        for ep in dataset.episodes
    ]
    return episode_rows, count_rows, split_rows


def _prompt_stats(episodes: list[EpisodeRecord]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for ep in episodes:
        meta = _video_metadata(ep)
        width = meta["video_width"]
        height = meta["video_height"]
        obj = ep.object_box.astype(np.float64)
        tgt = ep.target_box.astype(np.float64)
        obj_wh = np.maximum(obj[2:4] - obj[0:2], 0)
        tgt_wh = np.maximum(tgt[2:4] - tgt[0:2], 0)
        obj_center = (obj[0:2] + obj[2:4]) / 2
        tgt_center = (tgt[0:2] + tgt[2:4]) / 2
        img_area = float(width * height) if width and height else math.nan
        rows.append(
            {
                "episode_key": ep.key,
                "scene": ep.scene,
                "task": ep.task,
                "object_area_px": float(obj_wh[0] * obj_wh[1]),
                "target_area_px": float(tgt_wh[0] * tgt_wh[1]),
                "object_area_frac": float((obj_wh[0] * obj_wh[1]) / img_area) if img_area else math.nan,
                "target_area_frac": float((tgt_wh[0] * tgt_wh[1]) / img_area) if img_area else math.nan,
                "object_width_px": float(obj_wh[0]),
                "object_height_px": float(obj_wh[1]),
                "target_width_px": float(tgt_wh[0]),
                "target_height_px": float(tgt_wh[1]),
                "object_center_x_frac": float(obj_center[0] / width) if width else math.nan,
                "object_center_y_frac": float(obj_center[1] / height) if height else math.nan,
                "target_center_x_frac": float(tgt_center[0] / width) if width else math.nan,
                "target_center_y_frac": float(tgt_center[1] / height) if height else math.nan,
                "object_target_center_dist_frac": (
                    float(np.linalg.norm((obj_center - tgt_center) / np.asarray([width, height], dtype=np.float64)))
                    if width and height
                    else math.nan
                ),
                "object_out_of_bounds": bool(width and height and (obj[0] < 0 or obj[1] < 0 or obj[2] > width or obj[3] > height)),
                "target_out_of_bounds": bool(width and height and (tgt[0] < 0 or tgt[1] < 0 or tgt[2] > width or tgt[3] > height)),
            }
        )

    summary = {
        key: _describe([row[key] for row in rows])
        for key in [
            "object_area_frac",
            "target_area_frac",
            "object_width_px",
            "object_height_px",
            "target_width_px",
            "target_height_px",
            "object_target_center_dist_frac",
        ]
    }
    summary["object_out_of_bounds"] = int(sum(row["object_out_of_bounds"] for row in rows))
    summary["target_out_of_bounds"] = int(sum(row["target_out_of_bounds"] for row in rows))
    return summary, rows


def _action_stats(
    dataset: VisionTrajDataset,
    split_name: str,
    max_samples: int,
    outlier_top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_indices = np.arange(len(dataset), dtype=np.int64)
    if max_samples > 0:
        sample_indices = sample_indices[:max_samples]

    action_chunks = []
    sample_rows = []
    for sample_index in sample_indices:
        ei, step = dataset._samples[int(sample_index)]
        ep = dataset.episodes[ei]
        valid_idx = dataset._valid_raw_idx[ei]
        pose = dataset._poses[ei]
        gripper = dataset._grippers[ei]
        cur_raw = int(valid_idx[step])
        fut_raw = valid_idx[step + 1 : step + 1 + dataset.action_horizon]
        rel_pose = dataset._relative_action(pose[cur_raw], pose[fut_raw])
        grip = gripper[fut_raw].astype(np.float32).reshape(-1, 1)
        actions = np.concatenate([rel_pose, grip], axis=-1)
        action_chunks.append(actions)

        pos_l2 = np.linalg.norm(actions[:, :3], axis=-1)
        step_disp = np.linalg.norm(np.diff(pose[[cur_raw, *fut_raw], :3, 3], axis=0), axis=-1)
        sample_rows.append(
            {
                "split": split_name,
                "sample_index": int(sample_index),
                "episode_index": int(ei),
                "episode_key": ep.key,
                "scene": ep.scene,
                "task": ep.task,
                "step": int(step),
                "raw_frame": cur_raw,
                "mean_position_l2": float(pos_l2.mean()),
                "final_position_l2": float(pos_l2[-1]),
                "max_position_l2": float(pos_l2.max()),
                "mean_step_displacement": float(step_disp.mean()),
                "max_step_displacement": float(step_disp.max()),
                "mean_rotation6d_l2": float(np.linalg.norm(actions[:, 3:9], axis=-1).mean()),
                "mean_gripper": float(actions[:, 9].mean()),
                "min_gripper": float(actions[:, 9].min()),
                "max_gripper": float(actions[:, 9].max()),
            }
        )

    if not action_chunks:
        return {"split": split_name, "samples": 0}, []

    all_actions = np.concatenate(action_chunks, axis=0)
    pos_l2_all = np.linalg.norm(all_actions[:, :3], axis=-1)
    rot_l2_all = np.linalg.norm(all_actions[:, 3:9], axis=-1)
    gripper = all_actions[:, 9]
    gripper_threshold = float(np.nanmedian(gripper))
    summary: dict[str, Any] = {
        "split": split_name,
        "samples": int(len(sample_indices)),
        "action_steps": int(all_actions.shape[0]),
        "position_l2": _describe(pos_l2_all),
        "rotation6d_l2": _describe(rot_l2_all),
        "gripper": _describe(gripper),
        "gripper_below_or_equal_median": int((gripper <= gripper_threshold).sum()),
        "gripper_above_median": int((gripper > gripper_threshold).sum()),
        "action_dim": {f"dim_{i}": _describe(all_actions[:, i]) for i in range(all_actions.shape[1])},
        "per_sample": {
            "mean_position_l2": _describe([r["mean_position_l2"] for r in sample_rows]),
            "final_position_l2": _describe([r["final_position_l2"] for r in sample_rows]),
            "max_step_displacement": _describe([r["max_step_displacement"] for r in sample_rows]),
        },
    }
    outliers = sorted(sample_rows, key=lambda r: r["final_position_l2"], reverse=True)[:outlier_top_k]
    summary["_plot_values"] = {
        "position_l2": pos_l2_all,
        "final_position_l2": [r["final_position_l2"] for r in sample_rows],
        "gripper": gripper,
    }
    return summary, outliers


def _image_stats(dataset: VisionTrajDataset, n_samples: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if n_samples <= 0 or len(dataset) == 0:
        return {"samples": 0}, []
    indices = np.linspace(0, len(dataset) - 1, num=min(n_samples, len(dataset)), dtype=np.int64)
    rows = []
    channel_means = []
    channel_stds = []
    brightness = []
    for sample_index in indices:
        ei, step = dataset._samples[int(sample_index)]
        ep = dataset.episodes[ei]
        raw_frame = int(dataset._valid_raw_idx[ei][step])
        try:
            frame = dataset._cache.read_frame(str(ep.video_path), raw_frame)
        except Exception as exc:
            rows.append(
                {
                    "sample_index": int(sample_index),
                    "episode_key": ep.key,
                    "raw_frame": raw_frame,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        arr = frame.astype(np.float32) / 255.0
        mean = arr.reshape(-1, 3).mean(axis=0)
        std = arr.reshape(-1, 3).std(axis=0)
        bright = arr.mean()
        channel_means.append(mean)
        channel_stds.append(std)
        brightness.append(float(bright))
        rows.append(
            {
                "sample_index": int(sample_index),
                "episode_key": ep.key,
                "scene": ep.scene,
                "task": ep.task,
                "raw_frame": raw_frame,
                "height": int(frame.shape[0]),
                "width": int(frame.shape[1]),
                "mean_r": float(mean[0]),
                "mean_g": float(mean[1]),
                "mean_b": float(mean[2]),
                "std_r": float(std[0]),
                "std_g": float(std[1]),
                "std_b": float(std[2]),
                "brightness": float(bright),
                "near_black": bool(bright < 0.03),
                "near_white": bool(bright > 0.97),
                "error": "",
            }
        )
    if not channel_means:
        return {"samples": 0, "failed_reads": len(rows)}, rows
    means = np.stack(channel_means)
    stds = np.stack(channel_stds)
    summary = {
        "samples": int(len(channel_means)),
        "failed_reads": int(sum(1 for row in rows if row.get("error"))),
        "rgb_mean": [float(x) for x in means.mean(axis=0)],
        "rgb_std": [float(x) for x in stds.mean(axis=0)],
        "brightness": _describe(brightness),
        "near_black": int(sum(row.get("near_black", False) for row in rows)),
        "near_white": int(sum(row.get("near_white", False) for row in rows)),
    }
    return summary, rows


def _distribution_distance(rows: list[dict[str, Any]], key: str) -> float | None:
    train = Counter(row[key] for row in rows if row["split"] == "train")
    val = Counter(row[key] for row in rows if row["split"] == "val")
    if not train or not val:
        return None
    all_keys = sorted(set(train) | set(val))
    train_total = sum(train.values())
    val_total = sum(val.values())
    return float(0.5 * sum(abs(train[k] / train_total - val[k] / val_total) for k in all_keys))


def _make_plots(out_dir: Path, episode_rows: list[dict[str, Any]], action_summaries: dict[str, Any]) -> None:
    try:
        import matplotlib
    except ImportError:
        return

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def hist(values, title, xlabel, filename, bins=50):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(arr, bins=bins, color="tab:blue", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.2)
        fig.savefig(plot_dir / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)

    hist([r["valid_frames_strided"] for r in episode_rows if r["split"] == "all"], "Episode Length", "strided valid frames", "episode_length.png")
    hist([r["samples"] for r in episode_rows if r["split"] == "all"], "Samples Per Episode", "samples", "samples_per_episode.png")
    all_action = action_summaries.get("all", {}).get("_plot_values", {})
    hist(all_action.get("position_l2", []), "Future Position L2", "L2", "position_l2.png")
    hist(all_action.get("final_position_l2", []), "Final Displacement", "L2", "final_displacement.png")
    hist(all_action.get("gripper", []), "Gripper Values", "gripper", "gripper.png")

    scene_counts = Counter(r["scene"] for r in episode_rows if r["split"] == "all")
    if scene_counts:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = sorted(scene_counts)
        ax.bar(labels, [scene_counts[x] for x in labels], color="tab:green")
        ax.set_title("Episodes By Scene")
        ax.set_ylabel("episodes")
        fig.savefig(plot_dir / "episodes_by_scene.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def _clean_summary_for_json(summary: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in summary.items():
        if key == "_plot_values":
            continue
        if isinstance(value, dict):
            cleaned[key] = _clean_summary_for_json(value)
        else:
            cleaned[key] = value
    return cleaned


def _write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Vision Trajectory Dataset Statistics",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Data root: `{summary['data_root']}`",
        f"- Annotations: `{summary['annotations']}`",
        f"- Action horizon: `{summary['params']['action_horizon']}`",
        f"- History horizon: `{summary['params']['history_horizon']}`",
        f"- Stride: `{summary['params']['stride']}`",
        "",
        "## Scale",
        "",
        f"- Episodes: `{summary['scale']['episodes_all']}`",
        f"- Samples: `{summary['scale']['samples_all']}`",
        f"- Train episodes / samples: `{summary['scale']['episodes_train']}` / `{summary['scale']['samples_train']}`",
        f"- Val episodes / samples: `{summary['scale']['episodes_val']}` / `{summary['scale']['samples_val']}`",
        "",
        "## Key Distributions",
        "",
        f"- Episode length mean: `{summary['episode_length']['mean']}`",
        f"- Samples per episode mean: `{summary['samples_per_episode']['mean']}`",
        f"- Position L2 mean (cm): `{100.0 * summary['actions']['all']['position_l2']['mean']}`",
        f"- Final displacement mean (cm): `{100.0 * summary['actions']['all']['per_sample']['final_position_l2']['mean']}`",
        f"- Gripper width mean (cm): `{100.0 * summary['actions']['all']['gripper']['mean']}`",
        "",
        "## Split Quality",
        "",
        f"- Scene distribution total variation: `{summary['split_quality']['scene_total_variation']}`",
        f"- Task distribution total variation: `{summary['split_quality']['task_total_variation']}`",
        "",
        "## Files",
        "",
        "- `summary.json`: machine-readable summary",
        "- `report.md`: this report",
        "- `episode_stats.csv`: per-episode lengths, sample counts, and video metadata",
        "- `count_stats.csv`: scene/task/scene-task counts",
        "- `prompt_stats.csv`: bbox size and location statistics",
        "- `image_stats.csv`: sampled RGB statistics",
        "- `outliers_*.csv`: largest final-displacement samples per split",
        "- `plots/`: basic histograms and count charts",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    from ..runtime import load_config
    cfg = load_config(args.config)
    data_root = Path(args.data_root or cfg.data.root)
    annotations = Path(args.annotations or cfg.data.annotations_json)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"runs/vision_traj_dataset_stats_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    stride = int(args.stride or cfg.data.stride)
    action_horizon = int(args.action_horizon or cfg.policy.action_horizon)
    history_horizon = int(args.history_horizon or cfg.policy.history_horizon)
    image_size = int(args.image_size or cfg.data.image_size)
    prompt_form = str(args.prompt_form or cfg.data.prompt_form)
    seed = int(args.seed if args.seed is not None else resolve_split_seed(cfg))
    val_ratio = float(args.val_ratio if args.val_ratio is not None else cfg.data.val_ratio)

    if args.scenes is not None:
        scenes = _as_list(args.scenes)
    else:
        cfg_scenes = cfg.data.get("scenes", None)
        scenes = _as_list(OmegaConf.to_container(cfg_scenes)) if cfg_scenes is not None else None

    episodes = build_episode_index(data_root, annotations)
    episodes = filter_episodes_by_scenes(episodes, scenes)
    train_eps, val_eps = split_episodes(episodes, val_ratio, seed)
    split_by_key = {ep.key: "train" for ep in train_eps}
    split_by_key.update({ep.key: "val" for ep in val_eps})

    def make_dataset(eps: list[EpisodeRecord]) -> VisionTrajDataset:
        return VisionTrajDataset(
            episodes=list(eps),
            action_horizon=action_horizon,
            history_horizon=history_horizon,
            image_size=image_size,
            stride=stride,
            prompt_form=prompt_form,
        )

    datasets = {
        "all": make_dataset(episodes),
        "train": make_dataset(train_eps),
        "val": make_dataset(val_eps),
    }

    episode_rows, count_rows, split_rows = _counts(datasets, split_by_key)
    prompt_summary, prompt_rows = _prompt_stats(datasets["all"].episodes)

    action_summaries = {}
    for split_name, dataset in datasets.items():
        action_summary, outliers = _action_stats(dataset, split_name, args.max_action_samples, args.outlier_top_k)
        action_summaries[split_name] = action_summary
        _write_csv(out_dir / f"outliers_{split_name}.csv", outliers)

    image_summary, image_rows = _image_stats(datasets["all"], args.image_stats_samples)

    _write_csv(out_dir / "episode_stats.csv", episode_rows)
    _write_csv(out_dir / "count_stats.csv", count_rows)
    _write_csv(out_dir / "split_stats.csv", split_rows)
    _write_csv(out_dir / "prompt_stats.csv", prompt_rows)
    _write_csv(out_dir / "image_stats.csv", image_rows)

    all_episode_rows = [row for row in episode_rows if row["split"] == "all"]
    summary = {
        "generated_at": timestamp,
        "data_root": str(data_root),
        "annotations": str(annotations),
        "out_dir": str(out_dir),
        "params": {
            "scenes": scenes,
            "val_ratio": val_ratio,
            "seed": seed,
            "stride": stride,
            "action_horizon": action_horizon,
            "history_horizon": history_horizon,
            "image_size": image_size,
            "prompt_form": prompt_form,
            "max_action_samples": args.max_action_samples,
            "image_stats_samples": args.image_stats_samples,
        },
        "scale": {
            "episodes_all": len(datasets["all"].episodes),
            "episodes_train": len(datasets["train"].episodes),
            "episodes_val": len(datasets["val"].episodes),
            "samples_all": len(datasets["all"]),
            "samples_train": len(datasets["train"]),
            "samples_val": len(datasets["val"]),
        },
        "episode_length": _describe([row["valid_frames_strided"] for row in all_episode_rows]),
        "samples_per_episode": _describe([row["samples"] for row in all_episode_rows]),
        "prompt": prompt_summary,
        "image": image_summary,
        "actions": {
            split_name: _clean_summary_for_json(action_summaries[split_name])
            for split_name in ["all", "train", "val"]
        },
        "split_quality": {
            "scene_total_variation": _distribution_distance(split_rows, "scene"),
            "task_total_variation": _distribution_distance(split_rows, "task"),
        },
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _write_report(out_dir, summary)

    if not args.no_plots:
        _make_plots(out_dir, episode_rows, action_summaries)

    print(f"[dataset_stats] wrote report to {out_dir}")


if __name__ == "__main__":
    main()
