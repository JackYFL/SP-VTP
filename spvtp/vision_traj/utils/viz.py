"""Visualization helpers for wandb logging and offline inspection."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch

from spvtp.vision_traj.utils.rotation_utils import rot6d_to_rotmat


_FIGURE_TEXT = {
    "title": 20,
    "axis_label": 18,
    "tick_label": 14,
    "legend": 16,
    "suptitle": 18,
}


def _apply_axis_text_sizes(ax) -> None:
    ax.title.set_fontsize(_FIGURE_TEXT["title"])
    ax.xaxis.label.set_size(_FIGURE_TEXT["axis_label"])
    ax.yaxis.label.set_size(_FIGURE_TEXT["axis_label"])
    if hasattr(ax, "zaxis"):
        ax.zaxis.label.set_size(_FIGURE_TEXT["axis_label"])
    ax.tick_params(axis="both", labelsize=_FIGURE_TEXT["tick_label"])
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(_FIGURE_TEXT["legend"])


def _tensor_to_rgb(img: torch.Tensor) -> np.ndarray:
    """(3,H,W) float in [0,1] -> (H,W,3) uint8 RGB."""
    arr = img.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (arr * 255.0).astype(np.uint8)


def overlay_prompts(
    first_frame: torch.Tensor,
    prompt_object: torch.Tensor,
    prompt_target: torch.Tensor,
    prompt_form: str,
) -> np.ndarray:
    """Draw object (green) and target (red) prompts onto a first frame.

    Args:
        first_frame: (3, H, W) float tensor in [0, 1].
        prompt_object/prompt_target: (4,) bbox or (2,) point in normalized [0, 1].
        prompt_form: 'bbox', 'point', 'none', 'vision_bbox',
            'vision_bbox_and_bbox', 'object_only', or 'target_only'.

    Returns:
        (H, W, 3) uint8 RGB array with overlays.
    """
    import cv2

    img = _tensor_to_rgb(first_frame).copy()
    H, W = img.shape[:2]

    if (
        prompt_form in (
            "none",
            "vision_bbox",
            "vision_bbox_and_bbox",
            "object_only",
            "target_only",
        )
        or prompt_object.numel() == 0
        or prompt_target.numel() == 0
    ):
        return img

    def _denorm_box(b):
        x1, y1, x2, y2 = b
        return int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

    def _denorm_pt(p):
        return int(p[0] * W), int(p[1] * H)

    obj = prompt_object.detach().cpu().numpy()
    tgt = prompt_target.detach().cpu().numpy()

    if prompt_form == "bbox":
        x1, y1, x2, y2 = _denorm_box(obj)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "obj", (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        x1, y1, x2, y2 = _denorm_box(tgt)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 64, 64), 2)
        cv2.putText(img, "tgt", (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 64, 64), 2)
    elif prompt_form == "point":
        x, y = _denorm_pt(obj)
        cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(img, "obj", (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        x, y = _denorm_pt(tgt)
        cv2.circle(img, (x, y), 6, (255, 64, 64), -1)
        cv2.putText(img, "tgt", (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 64, 64), 2)
    else:
        raise ValueError(f"unknown prompt_form: {prompt_form!r}")

    return img


def sample_prompt_overlays(batch: dict, prompt_form: str, n: int = 4) -> List[np.ndarray]:
    """Produce up to ``n`` overlay images from a batch for wandb logging."""
    n = min(n, batch["first_frame"].shape[0])
    return [
        overlay_prompts(
            batch["first_frame"][i],
            batch["prompt_object"][i],
            batch["prompt_target"][i],
            prompt_form,
        )
        for i in range(n)
    ]


def _set_equal_3d_axes(ax, points: np.ndarray) -> None:
    center = points.mean(axis=0)
    radius = max(float(np.abs(points - center).max()), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _draw_orientation_arrow(ax, pos: np.ndarray, rot: np.ndarray, color: str, scale: float) -> None:
    direction = rot[:, 2]
    ax.quiver(
        pos[0],
        pos[1],
        pos[2],
        direction[0],
        direction[1],
        direction[2],
        length=scale,
        normalize=True,
        color=color,
        linewidth=1.4,
        arrow_length_ratio=0.2,
    )


def save_trajectory_comparison(
    path: str | Path,
    *,
    first_frame: torch.Tensor,
    current_frame: torch.Tensor,
    prompt_object: torch.Tensor,
    prompt_target: torch.Tensor,
    prompt_form: str,
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    title: str | None = None,
    subtitle: str | None = None,
) -> None:
    """Save a 3D trajectory comparison figure for one sample.

    The figure contains:
    - current frame, whose camera coordinates define the 3D trajectory frame
    - first frame with prompt overlay, used as the task specification
    - 3D end-effector trajectory (prediction vs ground truth) in current-frame coordinates
    - per-step position / rotation / gripper errors
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError(
            "save_trajectory_comparison requires matplotlib. Reinstall "
            "`spvtp/vision_traj` or install `matplotlib` into the active "
            "environment before using --viz-dir."
        ) from exc

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    pred = pred_actions.detach().cpu().float()
    gt = gt_actions.detach().cpu().float()
    pred_pos = 100.0 * pred[:, :3].numpy()
    gt_pos = 100.0 * gt[:, :3].numpy()
    pred_rot = rot6d_to_rotmat(pred[:, 3:9]).numpy()
    gt_rot = rot6d_to_rotmat(gt[:, 3:9]).numpy()

    pos_err = np.linalg.norm(pred_pos - gt_pos, axis=-1)
    rot_err = np.linalg.norm(pred[:, 3:9].numpy() - gt[:, 3:9].numpy(), axis=-1)
    grip_err = 100.0 * np.abs(pred[:, 9].numpy() - gt[:, 9].numpy())
    horizon = np.arange(1, pred.shape[0] + 1)

    prompt_img = overlay_prompts(first_frame, prompt_object, prompt_target, prompt_form)
    current_img = _tensor_to_rgb(current_frame)

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.18, wspace=0.02, hspace=0.12)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5], width_ratios=[1.0, 1.0])

    ax_current = fig.add_subplot(gs[0, 0])
    ax_current.imshow(current_img, aspect="auto")
    ax_current.set_title("Current Frame (origin for 3D trajectory)")
    ax_current.axis("off")
    _apply_axis_text_sizes(ax_current)

    ax_prompt = fig.add_subplot(gs[0, 1])
    ax_prompt.imshow(prompt_img, aspect="auto")
    ax_prompt.set_title("First Frame Task Prompt")
    ax_prompt.axis("off")
    _apply_axis_text_sizes(ax_prompt)

    ax_traj = fig.add_subplot(gs[1, 0], projection="3d")
    origin = np.zeros((1, 3), dtype=np.float32)
    gt_path = np.vstack([origin, gt_pos])
    pred_path = np.vstack([origin, pred_pos])
    ax_traj.plot(gt_path[:, 0], gt_path[:, 1], gt_path[:, 2], "-o", color="tab:blue", label="GT")
    ax_traj.plot(pred_path[:, 0], pred_path[:, 1], pred_path[:, 2], "-^", color="tab:orange", label="Pred")
    ax_traj.scatter([0.0], [0.0], [0.0], c="black", marker="x", s=64, label="Current")
    draw_steps = sorted({0, pred.shape[0] // 2, pred.shape[0] - 1})
    arrow_scale = max(
        float(np.ptp(np.vstack([pred_pos, gt_pos, np.zeros((1, 3), dtype=np.float32)]), axis=0).max()) * 0.18,
        5e-1,
    )
    for step in draw_steps:
        _draw_orientation_arrow(ax_traj, gt_pos[step], gt_rot[step], "tab:blue", arrow_scale)
        _draw_orientation_arrow(ax_traj, pred_pos[step], pred_rot[step], "tab:orange", arrow_scale)
    all_points = np.vstack([pred_pos, gt_pos, origin])
    _set_equal_3d_axes(ax_traj, all_points)
    ax_traj.view_init(elev=25, azim=135)
    ax_traj.set_title("3D End-Effector Trajectory (current camera frame)")
    ax_traj.set_xlabel("x (cm)")
    ax_traj.set_ylabel("y (cm)")
    ax_traj.set_zlabel("z (cm)")
    ax_traj.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.78),
        bbox_transform=ax_traj.transAxes,
        borderaxespad=0.0,
        framealpha=0.85,
    )
    _apply_axis_text_sizes(ax_traj)

    err_gs = gs[1, 1].subgridspec(3, 1, hspace=0.15)
    ax_pos = fig.add_subplot(err_gs[0, 0])
    ax_rot = fig.add_subplot(err_gs[1, 0], sharex=ax_pos)
    ax_grip = fig.add_subplot(err_gs[2, 0], sharex=ax_pos)

    ax_pos.plot(horizon, pos_err, color="tab:green", marker="o")
    ax_pos.set_ylabel("L2 (cm)")
    ax_pos.set_title(f"Position Error (mean={pos_err.mean():.2f} cm)")
    ax_pos.grid(True, alpha=0.25)
    _apply_axis_text_sizes(ax_pos)

    ax_rot.plot(horizon, rot_err, color="tab:red", marker="o")
    ax_rot.set_ylabel("L2")
    ax_rot.set_title(f"Rotation6D Error (mean={rot_err.mean():.4f})")
    ax_rot.grid(True, alpha=0.25)
    _apply_axis_text_sizes(ax_rot)

    ax_grip.plot(horizon, grip_err, color="tab:purple", marker="o")
    ax_grip.set_xlabel("Horizon Step")
    ax_grip.set_ylabel("L1 (cm)")
    ax_grip.set_title(f"Gripper Error (mean={grip_err.mean():.2f} cm)")
    ax_grip.grid(True, alpha=0.25)
    _apply_axis_text_sizes(ax_grip)

    header = title or "Validation Trajectory Comparison"
    if subtitle:
        header = f"{header}\n{subtitle}"
    fig.suptitle(header, fontsize=_FIGURE_TEXT["suptitle"])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
