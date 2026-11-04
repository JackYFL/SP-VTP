import os
import numpy as np
import cv2
import zarr
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import open3d as o3d
from spvtp.common.trajectory_util import (
    _step_time_aligned_trajectories_incremental,
    world_align_poses,
)
import pathlib

def plot_gripper_width(gripper_widths, frame_timestamps, output_path):
    """
    Plot gripper widths over time and save as an image.

    Args:
        gripper_widths: Array of gripper widths for each frame
        frame_timestamps: Timestamps or frame indices corresponding to gripper widths
        output_path: Path to save the generated plot image
    """
    # Create a plot of the gripper width over time
    plt.figure(figsize=(10, 6))
    plt.plot(frame_timestamps, gripper_widths, label="Gripper Width", color='blue')
    plt.xlabel('Time (frames)')
    plt.ylabel('Gripper Width')
    plt.title('Gripper Width Over Time')
    plt.grid(True)
    plt.legend()

    # Save the plot as a PNG image
    plt.savefig(output_path, format='png')
    plt.close()

    print(f"Gripper width plot saved to {output_path}")


def visualize_flying_umi_processed_data_episode(demonstration_dir, output_dir, cfg):
    """Visualize the processed data for a single episode of flying UMI"""
    oak_camera_video_path = os.path.join(demonstration_dir, 'camera_1.mp4')
    demo_basename = os.path.basename(demonstration_dir)
    output_zarr = zarr.open(os.path.join(output_dir,demo_basename),mode="a")
    episode = os.path.basename(demonstration_dir)
    demo_session_name = demonstration_dir.split('/')[-2].split('.')[0]
 
    video_out_dir = os.path.join(pathlib.Path(__file__).parent.parent.parent.parent, cfg.video_out_dir, demo_session_name)
    os.makedirs(video_out_dir, exist_ok=True)
    video_out_path = os.path.join(video_out_dir, f'{episode}_processed_vis.mp4')

    vis_video_aligned_trajectories(
        video_path=oak_camera_video_path,
        poses = output_zarr['pose_interp'][:],
        gripper_widths=output_zarr['gripper_widths'][:],
        valids=output_zarr['valid_indices'][:],
        video_out_path=video_out_path, 
        fps=cfg.fps,
        max_frames=cfg.max_frames,
        downsample =cfg.downsample,
        offscreen=cfg.offscreen,
        world_align=getattr(cfg, 'world_align', True),
        include_base_frame=getattr(cfg, 'include_base_frame', True),
        include_grid=getattr(cfg, 'include_grid', True),
        fix_camera=getattr(cfg, 'fix_camera', True),
        camera_mode=getattr(cfg, 'camera_mode', 'isometric'),
        axis_size=getattr(cfg, 'axis_size', 0.06),
        grid_step=getattr(cfg, 'grid_step', None),
    )

    print(f"Visualized episode {demonstration_dir} and saved to {video_out_path}")

 
def vis_video_aligned_trajectories(
    video_path,
    gripper_widths,
    valids,
    video_out_path,
    poses,
    fps=None,                 # if None, use source fps
    max_frames=-1,
    target_height=360,        # video/trajectory row height
    offscreen=True,
    downsample=1,             # visualize every Nth frame
    # New clarity controls
    world_align=True,
    include_base_frame=True,
    include_grid=True,
    fix_camera=True,
    camera_mode='isometric',  # 'isometric' | 'top' | 'side'
    axis_size=0.06,
    grid_step=None,
):
    """
    Layout:
      TOP ROW: [video (left, scaled to target_height) | trajectory render (right, square)]
      BOTTOM ROW: gripper-width plot spanning full width (letterboxed; aspect ratio preserved)

    Notes:
      - Only writes a frame if valids[i] is True (still advances pose stepper).
      - Downsampling applies to video, trajectory stepping, and gripper plotting.
      - X-axis for the plot uses *source video timestamps in seconds*.
      - Requires:
          create_gripper_width_plot_up_to_current_timestamp_x(series, frame_timestamps, idx)
          _step_time_aligned_trajectories_incremental(offscreen, scene, present_poses, material, include_base_frame=False)
    """
    assert os.path.exists(video_path), f"Missing: {video_path}"
    assert downsample >= 1, "downsample must be >= 1"
    Path(video_out_path).parent.mkdir(parents=True, exist_ok=True)

    # --- helpers ---
    def _safe_len(x, default_len):
        try:
            return len(x)
        except Exception:
            return default_len

    def _fit_preserve_ar(img_bgr, dst_w, dst_h, bg_color=(255, 255, 255)):
        """Letterbox img_bgr into (dst_h, dst_w) preserving aspect ratio."""
        h, w = img_bgr.shape[:2]
        if h == 0 or w == 0:
            return np.full((dst_h, dst_w, 3), bg_color, dtype=np.uint8)
        scale = min(dst_w / w, dst_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
        canvas = np.full((dst_h, dst_w, 3), bg_color, dtype=np.uint8)
        y0 = (dst_h - new_h) // 2
        x0 = (dst_w - new_w) // 2
        canvas[y0:y0+new_h, x0:x0+new_w] = resized
        return canvas

    # --- video capture ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    T_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    # --- align lengths ---
    T_candidates = [
        T_src,
        _safe_len(gripper_widths, T_src),
        _safe_len(valids, T_src),
        _safe_len(poses, T_src),
    ]
    T_data = min([t for t in T_candidates if t is not None and t > 0])
    T = T_data if max_frames <= 0 else min(T_data, max_frames)
    if T <= 0:
        raise ValueError("No frames to process (after alignment).")

    # fps (unchanged => time-lapse when downsample > 1)
    video_fps = float(fps) if fps is not None else src_fps

    # --- sizing: top row side-by-side; bottom plot full width ---
    scale = float(target_height) / float(max(H_src, 1))
    vid_h = out_h = int(round(H_src * scale))      # top-row height
    vid_w = out_w = int(round(W_src * scale))      # video width in top row

    traj_W = traj_H = out_h                        # square, same height as video
    plot_h = out_h
    plot_w = vid_w + traj_W

    combined_h = out_h + plot_h                    # top row + bottom row
    combined_w = plot_w

    # --- downsampled timeline ---
    indices = np.arange(0, T, downsample, dtype=np.int32)
    # Use *source* timeline for timestamps (seconds); avoids mixing with output fps
    frame_timestamps_ds = indices.astype(np.float64) / max(src_fps, 1e-6)
    gw_full = np.asarray(gripper_widths)[:T]
    gripper_widths_ds = gw_full[indices]

    # --- pre-render gripper plot base with axes (fast OpenCV) ---
    # Compute y-range with margin once
    if len(gripper_widths_ds) == 0:
        raise ValueError("No gripper widths to visualize after downsampling.")
    # Fixed Y-axis: always from 0.0 to 0.1 (meters) with 10 increments
    y_min = 0.0
    y_max = 0.1
    increments = 10

    # Time mapping
    t0 = float(frame_timestamps_ds[0])
    tN = float(frame_timestamps_ds[-1])
    t_span = max(tN - t0, 1e-6)

    # Layout margins for axes
    left_margin = max(50, int(round(0.06 * plot_w)))
    right_margin = int(round(0.02 * plot_w))
    top_margin = int(round(0.08 * plot_h))
    bottom_margin = max(28, int(round(0.12 * plot_h)))
    data_w = max(1, plot_w - left_margin - right_margin)
    data_h = max(1, plot_h - top_margin - bottom_margin)

    # Map timestamps and widths to pixel positions within data rect
    x_coords_px = (left_margin + np.clip(((frame_timestamps_ds - t0) / t_span), 0, 1) * (data_w - 1)).astype(np.int32)
    norm_y = (gripper_widths_ds - y_min) / max(y_max - y_min, 1e-6)
    y_coords_px = (top_margin + (1.0 - np.clip(norm_y, 0, 1)) * (data_h - 1)).astype(np.int32)
    poly_pts = np.stack([x_coords_px, y_coords_px], axis=1).astype(np.int32).reshape(-1, 1, 2)

    # Create base canvas
    base_plot = np.full((plot_h, plot_w, 3), 255, dtype=np.uint8)

    # Draw axes
    axis_color = (0, 0, 0)
    tick_color = (0, 0, 0)
    text_color = (0, 0, 0)
    # Y-axis
    y_axis_x = left_margin
    y_axis_y0 = top_margin
    y_axis_y1 = top_margin + data_h
    cv2.line(base_plot, (y_axis_x, y_axis_y0), (y_axis_x, y_axis_y1), axis_color, 1, cv2.LINE_AA)
    # X-axis
    x_axis_y = top_margin + data_h
    x_axis_x0 = left_margin
    x_axis_x1 = left_margin + data_w
    cv2.line(base_plot, (x_axis_x0, x_axis_y), (x_axis_x1, x_axis_y), axis_color, 1, cv2.LINE_AA)

    # Ticks and labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(0.6, plot_h / 600.0))
    thickness = 1
    # X ticks (time in s)
    num_xticks = 6
    for i_tick in range(num_xticks):
        frac = 0.0 if num_xticks == 1 else i_tick / (num_xticks - 1)
        tx = int(round(x_axis_x0 + frac * (data_w - 1)))
        cv2.line(base_plot, (tx, x_axis_y), (tx, x_axis_y + 5), tick_color, 1, cv2.LINE_AA)
        t_val = t0 + frac * t_span
        label = f"{t_val:.1f}s"
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.putText(base_plot, label, (tx - tw // 2, x_axis_y + 5 + th + 2), font, font_scale, text_color, thickness, cv2.LINE_AA)
    # Y ticks (width) fixed to 0.00 .. 0.10 in 10 increments
    num_yticks = increments + 1
    for i_tick in range(num_yticks):
        frac = 0.0 if num_yticks == 1 else i_tick / (num_yticks - 1)
        ty = int(round(y_axis_y1 - frac * (data_h - 1)))
        cv2.line(base_plot, (y_axis_x - 5, ty), (y_axis_x, ty), tick_color, 1, cv2.LINE_AA)
        y_val = y_min + frac * (y_max - y_min)
        label = f"{y_val:.2f}"
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.putText(base_plot, label, (max(2, y_axis_x - 8 - tw), ty + th // 2 - 2), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Draw gripper width polyline
    if len(poly_pts) >= 2:
        cv2.polylines(base_plot, [poly_pts], isClosed=False, color=(0, 191, 255), thickness=2, lineType=cv2.LINE_AA)

    # Align poses to a consistent world (first pose becomes identity)
    if world_align:
        poses_aligned, _ = world_align_poses(poses)
    else:
        poses_aligned = poses

    # --- Open3D renderer + stepper ---
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    if offscreen:
        renderer = o3d.visualization.rendering.OffscreenRenderer(traj_W, out_h)
        scene = renderer.scene
        scene.set_background([1, 1, 1, 1])
    else:
        renderer = o3d.visualization.Visualizer()
        renderer.create_window(visible=True, width=traj_W, height=traj_H)
        scene = renderer

    # Optionally add a static XY grid at z=0 for depth cues
    def _add_static_geometry(name, geometry):
        if offscreen:
            scene.add_geometry(name, geometry, material)
        else:
            renderer.add_geometry(geometry)

    if include_grid:
        # Determine grid extents from pose translations
        pts = np.asarray(poses_aligned)[:T, :3, 3]
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        extent = np.maximum(maxs - mins, 1e-6)
        size_xy = float(np.max(extent[:2]))
        step = float(grid_step) if grid_step is not None else max(size_xy / 20.0, 1e-3)
        grid = _make_xy_grid_lines(center=(mins[:2] + maxs[:2]) / 2.0, size=size_xy * 1.2, step=step, z=0.0)
        _add_static_geometry("xy_grid", grid)

    present_poses = [poses_aligned]
    geom_stepper = _step_time_aligned_trajectories_incremental(
        offscreen, scene, present_poses, material, axis_size=axis_size, include_base_frame=include_base_frame
    )

    # --- VideoWriter with fallbacks ---
    def _open_writer(codec_fourcc: str):
        fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)
        return cv2.VideoWriter(str(video_out_path), fourcc, video_fps, (combined_w, combined_h), True)

    writer = _open_writer("mp4v")
    if not writer.isOpened():
        writer = _open_writer("avc1")  # requires system H.264
    if not writer.isOpened():
        writer = _open_writer("XVID")
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter. Try a different codec or container (e.g., .avi with 'XVID').")

    # Precompute fixed camera if requested
    def _compute_fixed_camera(center_hint=None):
        pts = np.asarray(poses_aligned)[:T, :3, 3]
        c = pts.mean(axis=0) if center_hint is None else center_hint
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        ext = maxs - mins
        scale = float(np.linalg.norm(ext) or 1.0)
        scale = max(scale, 1e-3)
        if camera_mode == 'top':
            eye = c + np.array([0.0, 0.0, 1.5]) * scale
            up = [1.0, 0.0, 0.0]
        elif camera_mode == 'side':
            eye = c + np.array([1.5, 0.0, 0.3]) * scale
            up = [0.0, 0.0, 1.0]
        else:  # isometric
            eye = c + np.array([1.2, 1.2, 0.8]) * scale
            up = [0.0, 0.0, 1.0]
        return c, eye, up

    if offscreen and fix_camera:
        cam_center, cam_eye, cam_up = _compute_fixed_camera()
        scene.camera.look_at(center=cam_center, eye=cam_eye, up=cam_up)

    # --- main loop (downsampled) ---
    with tqdm(total=len(indices), desc="vis_video_aligned_trajectories", leave=False) as pbar:
        current_pos = -1
        valid_frame_index = 0  # Track position in pose_interp array
        for k, i in enumerate(indices):
            # Read sequentially up to desired frame i to avoid random seeks
            steps = int(i - current_pos)
            last_frame = None
            for _ in range(steps):
                ok, frame = cap.read()
                if not ok:
                    last_frame = None
                    break
                current_pos += 1
                last_frame = frame
                # Only step trajectory for valid frames that have corresponding poses
                if current_pos < len(valids) and bool(valids[current_pos]):
                    next(geom_stepper)

            if last_frame is None:
                break

            # top-left: video
            if vid_w < W_src or out_h < H_src:
                interp_method = cv2.INTER_AREA
            else:
                interp_method = cv2.INTER_CUBIC
            frame_rs = cv2.resize(last_frame, (vid_w, out_h), interpolation=interp_method)

            # top-right: trajectory render (already stepped above for valid frames only)
            if offscreen:
                if not fix_camera:
                    bbox = scene.bounding_box
                    center = bbox.get_center()
                    eye = center + 0.6 * bbox.get_extent() + np.array([0, 0, 0.1])
                    scene.camera.look_at(center=center, eye=eye, up=[0, 0, 1])

                rendered = renderer.render_to_image()
                rendered_np = np.asarray(rendered)
                traj_bgr = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)
            else:
                renderer.poll_events()
                renderer.update_renderer()
                trajectory_image = renderer.capture_screen_float_buffer(do_render=False)
                trajectory_image = (255 * np.asarray(trajectory_image)).astype(np.uint8)
                traj_bgr = cv2.cvtColor(trajectory_image, cv2.COLOR_RGB2BGR)

            if (traj_bgr.shape[0] != out_h) or (traj_bgr.shape[1] != traj_W):
                traj_bgr = cv2.resize(traj_bgr, (traj_W, out_h), interpolation=cv2.INTER_AREA)

            top_row = np.hstack([frame_rs, traj_bgr])  # (out_h, plot_w, 3)

            # bottom: fast gripper plot with moving time cursor
            plot_img = base_plot.copy()
            x = int(x_coords_px[k]) if k < len(x_coords_px) else int(x_coords_px[-1])
            cv2.line(plot_img, (x, 0), (x, plot_h - 1), (180, 105, 255), 2, lineType=cv2.LINE_AA)

            # final compose
            combined = np.vstack([top_row, plot_img])  # (combined_h, combined_w, 3)

            # sanity + write
            if combined.shape[0] != combined_h or combined.shape[1] != combined_w:
                raise RuntimeError(
                    f"Frame size {combined.shape[1]}x{combined.shape[0]} != expected {combined_w}x{combined_h}"
                )
            if i < len(valids) and bool(valids[i]):
                writer.write(combined)
                valid_frame_index += 1

            pbar.update(1)

    writer.release()
    cap.release()
    if not offscreen:
        cv2.destroyAllWindows()
        renderer.destroy_window()

    print(f"Saved video to {video_out_path}")


def create_gripper_width_plot_up_to_current_timestamp_x(gripper_widths, timestamps, frame_index):
    """
    Create a plot for gripper widths up to the current frame, with fixed Y-axis range and elapsed time-based X-axis.

    Args:
        gripper_widths: numpy array of shape (N) containing gripper widths.
        timestamps: numpy array of shape (N,) containing timestamps as floats in datetime.timestamp() format.
        frame_index: Current frame index.

    Returns:
        Plotly figure object.
    """
    # Ensure inputs are valid
    assert len(gripper_widths) == len(timestamps), "Gripper widths and timestamps must have the same length."

    # Normalize timestamps to start from 0 ms
    elapsed_time = (timestamps - timestamps[0])  # plot in seconds
    current_elapsed_time = elapsed_time[:frame_index + 1]  # Up to current frame

    # Compute fixed Y-axis range with margins
    gripper_min, gripper_max = gripper_widths.min(), gripper_widths.max()
    margin = 0.1 * (gripper_max - gripper_min) if gripper_max > gripper_min else 0.1
    y_min = gripper_min - margin
    y_max = gripper_max + margin

    # Create the figure
    fig = go.Figure()

    # Add the gripper width line plot
    fig.add_trace(go.Scatter(
        x=current_elapsed_time,
        y=gripper_widths[:frame_index + 1],
        mode='lines',
        name='Gripper Width',
        line=dict(color='gold')
    ))

    # Add a vertical line indicating the current elapsed time
    fig.add_vline(x=elapsed_time[frame_index], line=dict(color='palevioletred', dash='dash'))

    # Update layout with fixed axis ranges and labels
    fig.update_layout(
        title="Gripper Width Over Time",
        xaxis_title="Elapsed Time (s)",
        yaxis_title="Gripper Width (m)",
        xaxis=dict(
            range=[0, elapsed_time[-1]],  # Elapsed time from 0 ms to the end
        ),
        yaxis=dict(range=[y_min, y_max]),
        height=300,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    return fig


def _make_xy_grid_lines(center, size, step, z=0.0, color=(0.85, 0.85, 0.85)):
    """Create a square XY grid as an Open3D LineSet centered at `center` with total `size` and spacing `step`."""
    cx, cy = float(center[0]), float(center[1])
    half = float(size) / 2.0
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half

    # Generate grid lines along X and Y
    xs = np.arange(x0, x1 + 1e-8, step)
    ys = np.arange(y0, y1 + 1e-8, step)

    points = []
    lines = []
    col = []

    # Vertical lines (constant x)
    for x in xs:
        p_start = [x, y0, z]
        p_end = [x, y1, z]
        idx0 = len(points)
        points.extend([p_start, p_end])
        lines.append([idx0, idx0 + 1])
        col.append(list(color))

    # Horizontal lines (constant y)
    for y in ys:
        p_start = [x0, y, z]
        p_end = [x1, y, z]
        idx0 = len(points)
        points.extend([p_start, p_end])
        lines.append([idx0, idx0 + 1])
        col.append(list(color))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(col, dtype=np.float64))
    return line_set

 