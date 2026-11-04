"""Visualize stitched full-episode trajectories from predicted action chunks.

This is intentionally separate from training-time wandb visualization. It runs
open-loop inference on samples from one or more episodes, aligns each predicted
action chunk by raw future frame, and aggregates overlapping predictions into a
single episode-level trajectory.

Example:

    python -m spvtp.vision_traj.scripts.vis_stitched_traj \
        --ckpt runs/vision_traj_default/ckpt_final.pt \
        --data-root spvtp/output_data \
        --annotations annotations_merged.json \
        --split val \
        --n-episodes 3 \
        --out-dir runs/vision_traj_default/stitched_val
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


from ..runtime import make_initial_noise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--split",
        choices=("val", "train", "tail", "all"),
        default="val",
        help="which episode split to visualize",
    )
    p.add_argument(
        "--n-episodes",
        type=int,
        default=3,
        help="number of episodes from the selected split to visualize; 0 keeps all",
    )
    p.add_argument(
        "--episode-indices",
        type=str,
        default=None,
        help="comma-separated episode indices within the selected split; overrides --n-episodes",
    )
    p.add_argument("--seed", type=int, default=None,
                   help="split seed override; defaults to cfg.data.split_seed, then cfg.train.seed")
    p.add_argument("--val-ratio", type=float, default=None,
                   help="validation ratio override; defaults to cfg.data.val_ratio")
    p.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="comma-separated scene names to restrict evaluation",
    )
    p.add_argument("--sample-seed", type=int, default=0,
                   help="base seed for deterministic initial noise")
    p.add_argument(
        "--stitch",
        choices=("mean_overlap", "first_action"),
        default="mean_overlap",
        help=(
            "mean_overlap averages all chunk predictions that land on the same raw frame; "
            "first_action uses only the first action from each chunk"
        ),
    )
    p.add_argument(
        "--max-samples-per-episode",
        type=int,
        default=0,
        help="debug limit on samples per episode; 0 uses all samples",
    )
    p.add_argument(
        "--figure-format",
        choices=("png", "pdf"),
        default="png",
        help="file format for stitched trajectory figures",
    )
    return p.parse_args()


def _select_episodes(args: argparse.Namespace, cfg, episodes):
    from omegaconf import OmegaConf
    from spvtp.vision_traj.data.dataset import (
        filter_episodes_by_scenes,
        resolve_split_seed,
        split_episodes,
    )

    split_seed = resolve_split_seed(cfg) if args.seed is None else args.seed
    val_ratio = cfg.data.val_ratio if args.val_ratio is None else args.val_ratio

    if args.scenes is not None:
        scenes_filter = [s.strip() for s in args.scenes.split(",") if s.strip()] or None
    else:
        cfg_scenes = cfg.data.get("scenes", None) if hasattr(cfg.data, "get") else None
        scenes_filter = OmegaConf.to_container(cfg_scenes) if cfg_scenes else None
    if scenes_filter:
        episodes = filter_episodes_by_scenes(episodes, scenes_filter)

    if args.split in ("train", "val"):
        train_eps, val_eps = split_episodes(episodes, val_ratio, split_seed)
        episodes = train_eps if args.split == "train" else val_eps
    elif args.split == "tail":
        episodes = episodes[-args.n_episodes:] if args.n_episodes > 0 else episodes

    if args.episode_indices:
        indices = [int(x.strip()) for x in args.episode_indices.split(",") if x.strip()]
        episodes = [episodes[i] for i in indices]
    elif args.split != "tail" and args.n_episodes > 0:
        episodes = episodes[:args.n_episodes]

    return episodes, split_seed, val_ratio


def _relative_pos_to_world(pos_rel: np.ndarray, t_cur: np.ndarray) -> np.ndarray:
    return (t_cur[:3, :3] @ pos_rel.astype(np.float64)) + t_cur[:3, 3]


def _set_equal_3d_axes(ax, points: np.ndarray) -> None:
    center = points.mean(axis=0)
    radius = max(float(np.abs(points - center).max()), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _save_episode_plot(
    path: Path,
    *,
    first_frame: torch.Tensor,
    prompt_object: torch.Tensor,
    prompt_target: torch.Tensor,
    prompt_form: str,
    raw_frames: list[int],
    gt_pos: np.ndarray,
    pred_pos: np.ndarray,
    title: str,
    subtitle: str,
) -> None:
    from spvtp.vision_traj.utils.viz import _FIGURE_TEXT, _apply_axis_text_sizes, overlay_prompts

    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for stitched trajectory visualization") from exc

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    prompt_img = overlay_prompts(first_frame, prompt_object, prompt_target, prompt_form)
    gt_pos = 100.0 * gt_pos
    pred_pos = 100.0 * pred_pos
    pos_err = np.linalg.norm(pred_pos - gt_pos, axis=-1)
    x = np.arange(len(raw_frames))

    fig = plt.figure(figsize=(17, 8), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.25, h_pad=0.12, wspace=0.08, hspace=0.02)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0])

    ax_prompt = fig.add_subplot(gs[0, 0])
    ax_prompt.imshow(prompt_img)
    ax_prompt.set_title("First Frame Task Prompt")
    ax_prompt.axis("off")
    _apply_axis_text_sizes(ax_prompt)

    ax_traj = fig.add_subplot(gs[0, 1], projection="3d")
    ax_traj.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], "-o", color="tab:blue", label="GT")
    ax_traj.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], "-^", color="tab:orange", label="Pred stitched")
    ax_traj.scatter(gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2], c="black", marker="x", s=64, label="Start")
    _set_equal_3d_axes(ax_traj, np.vstack([gt_pos, pred_pos]))
    ax_traj.view_init(elev=25, azim=135)
    ax_traj.set_title("Stitched Episode Trajectory")
    ax_traj.set_xlabel("x (cm)")
    ax_traj.set_ylabel("y (cm)")
    ax_traj.set_zlabel("z (cm)")
    ax_traj.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.9),
        bbox_transform=ax_traj.transAxes,
        borderaxespad=0.0,
        framealpha=0.85,
    )
    ax_traj.set_box_aspect((1.0, 1.0, 1.0))
    _apply_axis_text_sizes(ax_traj)

    ax_err = fig.add_subplot(gs[0, 2])
    ax_err.plot(x, pos_err, color="tab:green", marker="o")
    ax_err.set_title(f"Position Error (mean={pos_err.mean():.2f} cm, final={pos_err[-1]:.2f} cm)")
    ax_err.set_xlabel("stitched trajectory index")
    ax_err.set_ylabel("L2 (cm)")
    ax_err.grid(True, alpha=0.25)
    ax_err.set_box_aspect(1.0)
    _apply_axis_text_sizes(ax_err)

    fig.suptitle(f"{title}\n{subtitle}", fontsize=_FIGURE_TEXT["suptitle"])
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _predict_episode_stitched(
    *,
    dataset: VisionTrajDataset,
    episode_index: int,
    policy: VisionTrajPolicy,
    device: torch.device,
    sample_seed: int,
    stitch: str,
    max_samples: int,
) -> dict:
    sample_refs = [
        (sample_index, step)
        for sample_index, (ei, step) in enumerate(dataset._samples)
        if ei == episode_index
    ]
    if max_samples > 0:
        sample_refs = sample_refs[:max_samples]
    if not sample_refs:
        raise ValueError(f"episode_index={episode_index} has no samples")

    pred_by_raw: dict[int, list[np.ndarray]] = defaultdict(list)
    gt_by_raw: dict[int, np.ndarray] = {}
    valid_raw_idx = dataset._valid_raw_idx[episode_index]
    poses = dataset._poses[episode_index]

    first_sample = dataset[int(sample_refs[0][0])]
    start_raw = int(valid_raw_idx[sample_refs[0][1]])
    gt_by_raw[start_raw] = poses[start_raw, :3, 3].astype(np.float64)
    pred_by_raw[start_raw].append(gt_by_raw[start_raw])

    for sample_index, step in sample_refs:
        sample = dataset[int(sample_index)]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
        noise = make_initial_noise(
            [int(sample_index)],
            horizon=policy.cfg.action_horizon,
            action_dim=policy.cfg.action_dim,
            base_seed=sample_seed,
        )
        pred = policy.predict_action(batch, initial_noise=noise)[0].detach().cpu().numpy()
        cur_raw = int(valid_raw_idx[step])
        t_cur = poses[cur_raw]

        offsets = [0] if stitch == "first_action" else range(policy.cfg.action_horizon)
        for h in offsets:
            raw = int(valid_raw_idx[step + 1 + h])
            pred_by_raw[raw].append(_relative_pos_to_world(pred[h, :3], t_cur))
            gt_by_raw[raw] = poses[raw, :3, 3].astype(np.float64)

    raw_frames = sorted(raw for raw in gt_by_raw if raw in pred_by_raw)
    gt_pos = np.stack([gt_by_raw[raw] for raw in raw_frames], axis=0)
    pred_pos = np.stack(
        [np.stack(pred_by_raw[raw], axis=0).mean(axis=0) for raw in raw_frames],
        axis=0,
    )
    err = np.linalg.norm(pred_pos - gt_pos, axis=-1)

    return {
        "episode": dataset.episodes[episode_index],
        "first_sample": first_sample,
        "raw_frames": raw_frames,
        "gt_pos": gt_pos,
        "pred_pos": pred_pos,
        "pos_err": err,
        "num_samples": len(sample_refs),
    }


def _write_csv(path: Path, raw_frames: list[int], gt_pos: np.ndarray, pred_pos: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    err = np.linalg.norm(pred_pos - gt_pos, axis=-1)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "raw_frame",
                "gt_x",
                "gt_y",
                "gt_z",
                "pred_x",
                "pred_y",
                "pred_z",
                "pos_l2",
            ],
        )
        writer.writeheader()
        for raw, gt, pred, e in zip(raw_frames, gt_pos, pred_pos, err):
            writer.writerow(
                {
                    "raw_frame": raw,
                    "gt_x": gt[0],
                    "gt_y": gt[1],
                    "gt_z": gt[2],
                    "pred_x": pred[0],
                    "pred_y": pred[1],
                    "pred_z": pred[2],
                    "pos_l2": e,
                }
            )


def main() -> None:
    args = parse_args()
    import torch
    from omegaconf import OmegaConf

    from spvtp.vision_traj.data.dataset import VisionTrajDataset, build_episode_index
    from spvtp.vision_traj.models.policy import VisionTrajPolicy, VisionTrajPolicyConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from ..runtime import load_checkpoint, make_dataset
    policy, cfg, _ = load_checkpoint(args.ckpt, device)
    policy.eval()

    episodes = build_episode_index(args.data_root, args.annotations)
    episodes, split_seed, val_ratio = _select_episodes(args, cfg, episodes)
    dataset = make_dataset(cfg, episodes)

    summaries = []
    for episode_index, ep in enumerate(dataset.episodes):
        with torch.no_grad():
            result = _predict_episode_stitched(
                dataset=dataset,
                episode_index=episode_index,
                policy=policy,
                device=device,
                sample_seed=args.sample_seed,
                stitch=args.stitch,
                max_samples=args.max_samples_per_episode,
            )
        stem = f"episode_{episode_index:04d}_{ep.scene}_{ep.task}"
        figure_path = out_dir / f"{stem}_stitched.{args.figure_format}"
        csv_path = out_dir / f"{stem}_stitched.csv"
        _save_episode_plot(
            figure_path,
            first_frame=result["first_sample"]["first_frame"],
            prompt_object=result["first_sample"]["prompt_object"],
            prompt_target=result["first_sample"]["prompt_target"],
            prompt_form=cfg.data.prompt_form,
            raw_frames=result["raw_frames"],
            gt_pos=result["gt_pos"],
            pred_pos=result["pred_pos"],
            title=f"{ep.key}",
            subtitle=(
                f"split={args.split} stitch={args.stitch} "
                f"samples={result['num_samples']} frames={len(result['raw_frames'])}"
            ),
        )
        _write_csv(csv_path, result["raw_frames"], result["gt_pos"], result["pred_pos"])
        summaries.append(
            {
                "episode_index": episode_index,
                "episode_key": ep.key,
                "figure": figure_path.name,
                "csv": csv_path.name,
                "num_samples": result["num_samples"],
                "num_frames": len(result["raw_frames"]),
                "mean_pos_l2": float(result["pos_err"].mean()),
                "final_pos_l2": float(result["pos_err"][-1]),
            }
        )
        print(
            f"[stitched] {ep.key} "
            f"mean_pos={100.0 * summaries[-1]['mean_pos_l2']:.2f}cm "
            f"final_pos={100.0 * summaries[-1]['final_pos_l2']:.2f}cm "
            f"-> {figure_path}"
        )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "split": args.split,
                "split_seed": split_seed,
                "val_ratio": float(val_ratio),
                "sample_seed": args.sample_seed,
                "stitch": args.stitch,
                "episodes": summaries,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
