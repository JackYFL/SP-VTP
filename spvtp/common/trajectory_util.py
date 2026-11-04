"""Utilities for aligning and visualizing trajectories"""

import numpy as np
import open3d as o3d
from spvtp.common.latency_util import regular_sample, get_latency
from spvtp.common.transform_util import pose_4x4_to_6d, pose_6d_to_4x4, pose_4x4_to_quat_xyzw
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
import os
from spvtp.common.transform_util import pose_6d_to_4x4

import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.graph_objects as go

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
    elapsed_time = (timestamps - timestamps[0]) * 1000  # Convert to milliseconds
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
        xaxis_title="Elapsed Time (ms)",
        yaxis_title="Gripper Width (m)",
        xaxis=dict(
            range=[0, elapsed_time[-1]],  # Elapsed time from 0 ms to the end
        ),
        yaxis=dict(range=[y_min, y_max]),
        height=300,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    return fig


def plot_video_aligned_trajectory_horizontal_plot_offscreen(frames, poses,
                                                gripper_widths,
                                                  video_output_path, fps=60, max_frames=-1, 
                                                  skip_start_frames=0, skip_end_frames=0):
    """
    Given poses and F/T data synchronized with video frames, plot everything together.
    
    Args:
        frames: numpy array of shape (N, H, W, 3) containing video frames in BGR format
        poses: list of poses synchronized with the frames
        ft_data_left: numpy array of shape (M, 6) containing force/torque measurements for left coinft
        ft_data_right: numpy array of shape (M, 6) containing force/torque measurements for right coinft
        ft_timestamps_left: numpy array of shape (M,) containing F/T timestamps for left
        ft_timestamps_right: numpy array of shape (M,) containing F/T timestamps for right
        frame_timestamps: numpy array of shape (N,) containing frame timestamps
        video_output_path: output path for the video
        fps: frames per second for output video
        max_frames: maximum number of frames to process (-1 for all)
        skip_start_frames: number of frames to skip at start
        skip_end_frames: number of frames to skip at end
    """

    # Get video dimensions from frames array
    T, H, W, _ = frames.shape
    assert T - skip_start_frames - skip_end_frames == len(poses)
    frame_timestamps = np.arange(T)  

    scale = 360 / H
    out_H = int(H * scale)
    out_W = int(W * scale)

    # Create Offscreen Renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(out_W, out_H)
    scene = renderer.scene

    if gripper_widths is not None:
        gripper_plot_height = out_H  # Smaller plot for gripper width
        combined_width = out_W * 2
    else:
        combined_width = out_W * 2

    # Calculate the widths and heights for the combined image
    combined_width = out_W * 2  # Left column + Right column
    left_height = out_H * 3
    right_height = out_H + (out_H * 2 if gripper_widths is not None else 0)
    total_combined_height = max(left_height, right_height)
    total_combined_width = out_W * 2 * 2  # Expand horizontally for a wider format

    # Initialize video writer
    video_writer = cv2.VideoWriter(
        video_output_path,
        cv2.VideoWriter_fourcc(*'avc1'),
        fps,
        (total_combined_width, total_combined_height)
    )

    num_frames = T if max_frames == -1 else min(max_frames + skip_start_frames, T)

    for i in tqdm(range(num_frames)):
        if i < skip_start_frames:
            continue
        if i == num_frames - skip_end_frames:
            break

        # # Clear previous geometries
        # scene.clear_geometry()

        # # Add the trajectory points up to the current frame
        # current_trajectory = poses[:i+1]
        # geometries = plot_time_aligned_trajectories([current_trajectory], include_base_frame=False, colors=[[0, 0, 0]])
        
        # for j, geometry in enumerate(geometries):
        #     scene.add_geometry(f"trajectory_{j}", geometry, o3d.visualization.rendering.MaterialRecord())

        # # Set up camera
        # scene.camera.look_at([1, 1, 1], [0, 0, 0], [0, 1, 0])
        # scene.camera.set_projection(60, out_W / out_H, 0.1, 10.0)

        # Render trajectory image
        image = renderer.render_to_image()
        trajectory_image = np.asarray(image).astype(np.uint8)

        # Resize trajectory image
        trajectory_image_resized = cv2.resize(trajectory_image, (out_W, out_H), interpolation=cv2.INTER_AREA)

        # Get the video frame and resize
        video_frame = frames[i]
        video_frame = cv2.resize(video_frame, dsize=(out_W, out_H), interpolation=cv2.INTER_CUBIC)
 
        # Convert the Plotly figure to an image
        ft_plot_image_left = ft_plot_left.to_image(format="png")
        ft_plot_left = cv2.imdecode(np.frombuffer(ft_plot_image_left, np.uint8), cv2.IMREAD_COLOR)
        ft_plot_left = cv2.resize(ft_plot_left, (combined_width, out_H * 2), interpolation=cv2.INTER_AREA)

        ft_plot_image_right = ft_plot_right.to_image(format="png")
        ft_plot_right = cv2.imdecode(np.frombuffer(ft_plot_image_right, np.uint8), cv2.IMREAD_COLOR)
        ft_plot_right = cv2.resize(ft_plot_right, (combined_width, out_H * 2), interpolation=cv2.INTER_AREA)

        if gripper_widths is not None:
            # Create gripper width plot
            gripper_plot = create_gripper_width_plot_up_to_current_timestamp_x(gripper_widths, frame_timestamps, i)
            gripper_plot_image = gripper_plot.to_image(format="png")
            gripper_plot = cv2.imdecode(np.frombuffer(gripper_plot_image, np.uint8), cv2.IMREAD_COLOR)
            gripper_plot = cv2.resize(gripper_plot, (total_combined_width // 2, gripper_plot_height), interpolation=cv2.INTER_AREA)

        # Combine video and trajectory side by side
        upper_half = np.hstack((video_frame, trajectory_image_resized[:, :, ::-1]))  # Convert RGB to BGR

        # Assemble the left and right columns
        left_column = np.vstack((upper_half, ft_plot_left)) if gripper_widths is not None else upper_half
        right_column = np.vstack((gripper_plot, ft_plot_right))

        # Pad the left and right columns to match height
        left_height = left_column.shape[0]
        right_height = right_column.shape[0]
        total_height = max(left_height, right_height)

        if left_height < total_height:
            left_column = np.pad(left_column, ((0, total_height - left_height), (0, 0), (0, 0)), mode='constant', constant_values=0)
        if right_height < total_height:
            right_column = np.pad(right_column, ((0, total_height - right_height), (0, 0), (0, 0)), mode='constant', constant_values=0)

        # Combine the columns side by side
        combined_image = np.hstack((left_column, right_column))
        video_writer.write(combined_image)

    # Release resources
    cv2.destroyAllWindows()
    video_writer.release()
    renderer = None  # Free memory


def plot_time_aligned_trajectories(time_aligned_trajectories, colors=None, axis_size=0.04, include_base_frame=True):
    # time_aligned_trajectories is a list of matrices with shape (T, 4, 4), where T is the number of time steps. Returns list of Open3D geometry objects that can be visualized.
    T = len(time_aligned_trajectories[0])  # Number of time steps

    if colors is None:
        colors = [[1,0,0], [0,1,0], [0,0,1], [0.6,0,0.6]]  # Default colors

    assert len(colors) >= len(time_aligned_trajectories)

    geometries = []

    for i, trajectory in enumerate(time_aligned_trajectories):
        assert trajectory.shape == (T, 4, 4)

        trajectory_1 = trajectory[:,:3,3]

        # Create LineSet for trajectory 1
        lines_1 = [[i, i + 1] for i in range(T - 1)]
        colors_1 = [colors[i] for j in range(T - 1)]

        line_set_1 = o3d.geometry.LineSet()
        line_set_1.points = o3d.utility.Vector3dVector(trajectory_1)
        line_set_1.lines = o3d.utility.Vector2iVector(lines_1)
        line_set_1.colors = o3d.utility.Vector3dVector(colors_1)

        def plot_trajectory_with_rgb_axes(poses, K=1):
            """
            Plots the trajectory with RGB axes using Open3D.

            Parameters:
            - poses: numpy array of shape (T, 4, 4), where T is the number of trajectories.
            - K: Subsampling factor, use every K-th frame in the trajectory.
            """
            # Create a list to hold the frames for visualization
            frames = []
            
            # Subsample the poses
            subsampled_poses = poses[::K]
            
            for i, pose in enumerate(subsampled_poses):
                # Extract the translation vector
                translation = pose[:3, 3]
                
                # Extract the rotation matrix
                rotation = pose[:3, :3]
                
                # Create the coordinate frame
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=translation)
                
                # Apply the rotation to the frame
                frame.rotate(rotation, center=translation)
                
                # Append the frame to the list
                frames.append(frame)
            
            # Create an Open3D visualization object
            return frames

        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2*axis_size)
        frames = plot_trajectory_with_rgb_axes(trajectory, K=30)

        if len(trajectory_1) > 1:
            geometries.append(line_set_1)
        if include_base_frame:
            geometries.append(base_frame)
        geometries.extend(frames)
    
    return geometries

def align_poses_different_timescales(pose1, pose2, time1, time2, show_alignment=False):
    # Use if pose timestamps are with respect to different timescales (their clocks are not aligned). This function aligns them using cross correlation. Returns aligned poses and times.

    # start both timescales at 0 (but at this point they are not aligned still)
    time1 -= min(time1)
    time2 -= min(time2)

    # Align trajectories using cross correlation
    t2 = time2
    x2 = pose_4x4_to_6d(pose2)
    t1 = time1
    x1 = pose_4x4_to_6d(pose1)

    n_dims = x1.shape[1]
    fig, axes = plt.subplots(n_dims, 3)
    fig.set_size_inches(15, 15, forward=True)

    # get independent latency for each dimension of position and rotation
    latencies = []
    for i in range(n_dims):
        latency, info = get_latency(x2[...,i], t2, x1[...,i], t1, force_positive=False)

        row = axes[i]
        ax = row[0]
        ax.plot(info['lags'], info['correlation'])
        ax.set_xlabel('lag')
        ax.set_ylabel('cross-correlation')
        ax.set_title(f"Action Dim {i} Cross Correlation")

        ax = row[1]
        ax.plot(t2, x2[...,i], label='target')
        ax.plot(t1, x1[...,i], label='actual')
        ax.set_xlabel('time')
        ax.set_ylabel('gripper-width')
        ax.legend()
        ax.set_title(f"Action Dim {i} Raw observation")

        ax = row[2]
        t_samples = info['t_samples'] - info['t_samples'][0]
        ax.plot(t_samples, info['x_target'], label='target')
        ax.plot(t_samples-latency, info['x_actual'], label='actual-latency')
        ax.set_xlabel('time')
        
        ax.set_ylabel('gripper-width')
        ax.legend()
        ax.set_title(f"Action Dim {i} Aligned with latency={latency:.04f}")
        latencies.append(latency)

    fig.tight_layout()
    if show_alignment:
        plt.show()

    avg_latency = np.mean(sorted(latencies)[1:-1]) # cut outliers (assumed to be first and last entries)

    t1 -= avg_latency # shift actual time to align with target time

    pose1 = pose_6d_to_4x4(x1)
    pose2 = pose_6d_to_4x4(x2)
    
    return pose1, pose2, t1, t2

def error_between_poses(poses1, poses2):
    # compute position error
    pos_delta = poses1[:,:3,3] - poses2[:,:3,3] 
    pos_err = np.linalg.norm(pos_delta, axis=1).mean()

    # compute rotation error (pre cut)
    delta_rotation_mat = poses1[:, :3, :3] @ np.linalg.inv(poses2[:, :3, :3])
    delta_rotation_rotvec = Rotation.from_matrix(delta_rotation_mat).as_rotvec()
    rot_err = np.linalg.norm(delta_rotation_rotvec, axis=1).mean()

    return pos_err, rot_err

def world_align_poses(poses, world_align_transform=None):
    # normalize poses with respect to the specified world transform or, if None, such that the first pose is the identity transformation
    if world_align_transform is None:
        world_align_transform = np.linalg.inv(poses[0])
    poses = np.array([world_align_transform @ pose for pose in poses]) # normalize with respect to initial frame pose
    return poses, world_align_transform

def get_time_aligned_poses(pose1, pose2, time1, time2, resample_dt=1/60):
    # required that time1 and time2 are on the same time scale (meaning that the times are recorded with respect to the same reference time). It's not required that time1 and time2 represent the same time intervals. Returns 6d poses that are sampled at the same times. Poses are in 4x4 format.
    t_start = max(time1[0],time2[0])
    t_end = min(time1[-1],time2[-1])

    n_samples = int((t_end - t_start) / resample_dt)
    t_samples = np.arange(n_samples) * resample_dt + t_start

    pose1_samples = sample_poses_at_times(pose1, time1, t_samples)
    pose2_samples = sample_poses_at_times(pose2, time2, t_samples)

    return pose1_samples, pose2_samples, t_samples

def sample_poses_at_times(poses, pose_times, sample_times):
    poses = pose_4x4_to_6d(poses)
    pose_samples = np.array([regular_sample(poses[:,i], pose_times, sample_times) for i in range(6)]).T
    pose_samples = pose_6d_to_4x4(pose_samples)
    return pose_samples

def _step_time_aligned_trajectories_incremental(offscreen, scene_or_vis, time_aligned_trajectories, material, axis_size=0.04, include_base_frame=True, line_width=3, colors=None):
    """You can either provide an offscreen scene or a visualizer. This handles both properly"""
    T = len(time_aligned_trajectories[0])
    if colors is None:
        colors = [[0,0,0], [0,1,0], [0,0,1], [0.6,0,0.6]]

    line_material = o3d.visualization.rendering.MaterialRecord()
    line_material.shader = "unlitLine"
    line_material.line_width = line_width

    def add_geo(name, geo, mat):
        if offscreen:
            scene = scene_or_vis
            scene.add_geometry(name, geo, mat)
        else:
            vis = scene_or_vis
            vis.add_geometry(geo)

    yield

    end_spheres = [None] * len(time_aligned_trajectories)
    traj_started = [False] * len(time_aligned_trajectories)
    for step_i in range(T-1):
        for traj_i, trajectory in enumerate(time_aligned_trajectories):            
            trajectory_1 = trajectory[step_i:step_i+2,:3,3]

            if np.all(trajectory_1[0] == trajectory_1[1]):
                # if the trajectory doesn't move at all, then don't plot this segment
                continue

            # add line
            points = o3d.utility.Vector3dVector(trajectory_1)
            lines = o3d.utility.Vector2iVector([[0, 1]])
            colors_1 = o3d.utility.Vector3dVector([colors[traj_i % len(colors)]])
            line_set = o3d.geometry.LineSet(points=points, lines=lines)
            line_set.colors = colors_1
            add_geo(f"line_{traj_i}_{step_i}", line_set, line_material)

            # Frame every N steps
            if step_i % 30 == 0 or not traj_started[traj_i]:
                translation = trajectory[step_i][:3, 3]
                rotation = trajectory[step_i][:3, :3]
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=translation)
                frame.rotate(rotation, center=translation)
                add_geo(f"frame_{traj_i}_{step_i}", frame, material)

            if not traj_started[traj_i]:
                # Green start sphere
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_size/4)
                sphere.translate(trajectory_1[0])
                sphere.paint_uniform_color([0, 1, 0])
                add_geo(f"start_sphere_{traj_i}", sphere, material)

                if include_base_frame:
                    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
                    add_geo("base_frame", base_frame, material)

            # Red moving end sphere
            if end_spheres[traj_i] is not None:
                if offscreen:
                    scene_or_vis.remove_geometry(f"end_sphere_{traj_i}")
                else:
                    scene_or_vis.remove_geometry(end_spheres[traj_i])
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_size/4)
            sphere.translate(trajectory_1[-1])
            sphere.paint_uniform_color([1, 0, 0])
            add_geo(f"end_sphere_{traj_i}", sphere, material)
            end_spheres[traj_i] = sphere

            traj_started[traj_i] = True
        
        yield

def get_open3d_geo_trajectories(time_aligned_trajectories, colors=None, axis_size=0.04, include_base_frame=True):
    """
    Gets geometry objects for one or more trajectories when can be then passed into an open3d visualizer. This function is not suitable for headless rendering.
    
    `time_aligned_trajectories` is a list of matrices with shape (T, 4, 4), where T is the number of time steps. Returns list of Open3D geometry objects that can be visualized.
    """
    T = len(time_aligned_trajectories[0])  # Number of time steps

    if colors is None:
        colors = [[1,0,0], [0,1,0], [0,0,1], [0.6,0,0.6]]  # Default colors

    assert len(colors) >= len(time_aligned_trajectories)

    geometries = []

    for i, trajectory in enumerate(time_aligned_trajectories):
        assert trajectory.shape == (T, 4, 4)

        trajectory_1 = trajectory[:,:3,3]

        # Create LineSet for trajectory 1
        lines_1 = [[i, i + 1] for i in range(T - 1)]
        colors_1 = [colors[i] for j in range(T - 1)]

        line_set_1 = o3d.geometry.LineSet()
        line_set_1.points = o3d.utility.Vector3dVector(trajectory_1)
        line_set_1.lines = o3d.utility.Vector2iVector(lines_1)
        line_set_1.colors = o3d.utility.Vector3dVector(colors_1)

        def plot_trajectory_with_rgb_axes(poses, K=1):
            """
            Plots the trajectory with RGB axes using Open3D.

            Parameters:
            - poses: numpy array of shape (T, 4, 4), where T is the number of trajectories.
            - K: Subsampling factor, use every K-th frame in the trajectory.
            """
            # Create a list to hold the frames for visualization
            frames = []
            
            # Subsample the poses
            subsampled_poses = poses[::K]
            
            for i, pose in enumerate(subsampled_poses):
                # Extract the translation vector
                translation = pose[:3, 3]
                
                # Extract the rotation matrix
                rotation = pose[:3, :3]
                
                # Create the coordinate frame
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=translation)
                
                # Apply the rotation to the frame
                frame.rotate(rotation, center=translation)
                
                # Append the frame to the list
                frames.append(frame)
            
            # Create an Open3D visualization object
            return frames

        frames = plot_trajectory_with_rgb_axes(trajectory, K=30)

        if len(trajectory_1) > 1:
            geometries.append(line_set_1)
        if include_base_frame:
            base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
            geometries.append(base_frame)
        
        # green sphere at the start of the trajectory
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_size/4)
        sphere.translate(trajectory_1[0])
        sphere.paint_uniform_color([0, 1, 0])
        geometries.append(sphere)

        # red sphere at the end of the trajectory
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_size/4)
        sphere.translate(trajectory_1[-1])
        sphere.paint_uniform_color([1, 0, 0])
        geometries.append(sphere)

        geometries.extend(frames)
    
    return geometries


def vis_trajectories(video_out_path, poses, is_lost=None, out_width=480, out_height=360, offscreen=True, fps=20):
    """
    Plots a series of one or more trajectories by saving to a video file.

    `offscreen` flag should be true if you are on a headless machine and want to use OffscreenRenderer and false if you are on a machine with a display and want to use Visualizer.
    Note that line width changes and the setting of up axis only apply if offscreen is true.
    """
    video_out_path = os.path.abspath(video_out_path)

    if type(poses) != list:
        poses = [poses]

    T = None
    for cur_poses in poses:
        if T is None:
            T = len(cur_poses)
        else:
            assert T == len(cur_poses)
    
    out_H = out_height
    out_W = out_width

    if is_lost is None:
        is_lost = np.zeros(T, dtype=bool)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    # Setup renderer
    if offscreen:
        renderer = o3d.visualization.rendering.OffscreenRenderer(out_W, out_H)
        scene = renderer.scene
        scene.set_background([1, 1, 1, 1])  # white background
    else:
        renderer = o3d.visualization.Visualizer()
        renderer.create_window(visible=True, width=out_W, height=out_H)
        scene = renderer

    video_writer = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc('a','v','c','1'), fps, (out_W, out_H))

    present_poses = [cur_poses[np.where(~is_lost)] for cur_poses in poses]
    geom_stepper = _step_time_aligned_trajectories_incremental(offscreen, scene, present_poses, material, include_base_frame=False)
    
    num_present_poses = len(present_poses[0])

    with tqdm(total=num_present_poses, leave=False, desc='vis_trajectories') as pbar:
        for i in range(T):
            if is_lost[i]:
                continue

            next(geom_stepper)

            if offscreen:
                # update camera position
                bbox = scene.bounding_box
                scene.camera.look_at(
                    center=bbox.get_center(),
                    eye=bbox.get_center() + 0.6 * bbox.get_extent() + np.array([0, 0, 0.1]),  # or scale based on bbox size
                    up=[0, 0, 1]
                )

                # Render to image
                rendered = renderer.render_to_image()
                rendered_np = np.asarray(rendered)
                rendered_bgr = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)
            else:
                # Update the visualizer with the current frame
                renderer.poll_events()
                renderer.update_renderer()

                # Capture the visualizer screen image
                trajectory_image = renderer.capture_screen_float_buffer(do_render=False)
                trajectory_image = (255 * np.asarray(trajectory_image)).astype(np.uint8)
                rendered_bgr = cv2.cvtColor(trajectory_image, cv2.COLOR_RGB2BGR)

            video_writer.write(rendered_bgr)

            pbar.update()

    video_writer.release()

    if not offscreen:
        cv2.destroyAllWindows()
        renderer.destroy_window()

    print(f"Saved video to {video_out_path}")

# def vis_video_aligned_trajectories(video_path, video_out_path, poses, gripper_widths, max_frames=-1, trajectory_render_size=360, offscreen=True):
#     """
#     Plots a video alongside one or more trajectories, saving the result as a video file.

#     - The input video is scaled so its height matches trajectory_render_size, and its width is scaled proportionally.
#     - The trajectory is rendered as a square of size (trajectory_render_size, trajectory_render_size).
#     - Each output frame consists of:
#         - Left: input video frame, resized to (scaled_W, trajectory_render_size), maintaining aspect ratio.
#         - Right: trajectory rendering, always (trajectory_render_size, trajectory_render_size).
#     - The output video has frame size (scaled_W + trajectory_render_size, trajectory_render_size).

#     Args:
#         video_path (str): Path to the input video.
#         video_out_path (str): Path to save the output video.
#         poses (list or np.ndarray): List of pose arrays, each of length T.
#         is_lost (np.ndarray or None): Boolean mask for lost frames.
#         max_frames (int): Maximum number of frames to visualize.
#         trajectory_render_size (int): Height (and width) of the square trajectory rendering.
#         offscreen (bool): Use offscreen rendering.
#     """
#     assert os.path.exists(video_path)
#     cap = cv2.VideoCapture(video_path)
#     T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     fps = cap.get(cv2.CAP_PROP_FPS)
 
#     # Compute scaled width for input video
#     scale = trajectory_render_size / H
#     scaled_W = int(W * scale)
#     out_H = trajectory_render_size
#     traj_W = trajectory_render_size  # Trajectory render is always square
#     out_W = scaled_W + traj_W

#     material = o3d.visualization.rendering.MaterialRecord()
#     material.shader = "defaultUnlit"

#     # Setup renderer for trajectory
#     if offscreen:
#         renderer = o3d.visualization.rendering.OffscreenRenderer(traj_W, out_H)
#         scene = renderer.scene
#         scene.set_background([1, 1, 1, 1])  # white background
#     else:
#         renderer = o3d.visualization.Visualizer()
#         renderer.create_window(visible=True, width=traj_W, height=out_H)
#         scene = renderer

#     video_writer = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc('a','v','c','1'), fps, (out_W, out_H))

#     present_poses = [poses]
#     geom_stepper = _step_time_aligned_trajectories_incremental(offscreen, scene, present_poses, material, include_base_frame=False)
    
#     num_present_poses = len(present_poses[0])
#     max_frames = num_present_poses if max_frames == -1 else min(num_present_poses, max_frames)
#     num_frames_visualized = 0

#     with tqdm(total=max_frames, leave=False, desc='vis_video_aligned_trajectories') as pbar:
#         for i in range(T):
#             found_frame, video_frame = cap.read()
#             assert found_frame

#             next(geom_stepper)

#             if offscreen:
#                 # update camera position
#                 bbox = scene.bounding_box
#                 scene.camera.look_at(
#                     center=bbox.get_center(),
#                     eye=bbox.get_center() + 0.6 * bbox.get_extent() + np.array([0, 0, 0.1]),  # or scale based on bbox size
#                     up=[0, 0, 1]
#                 )

#                 # Render to image
#                 rendered = renderer.render_to_image()
#                 rendered_np = np.asarray(rendered)
#                 rendered_bgr = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)
#             else:
#                 # Update the visualizer with the current frame
#                 renderer.poll_events()
#                 renderer.update_renderer()

#                 # Capture the visualizer screen image
#                 trajectory_image = renderer.capture_screen_float_buffer(do_render=False)
#                 trajectory_image = (255 * np.asarray(trajectory_image)).astype(np.uint8)
#                 rendered_bgr = cv2.cvtColor(trajectory_image, cv2.COLOR_RGB2BGR)

#             # Resize video frame to (scaled_W, out_H), maintaining aspect ratio
#             video_frame_resized = cv2.resize(video_frame, dsize=(scaled_W, out_H), interpolation=cv2.INTER_CUBIC)

#             # Ensure trajectory render is (traj_W, out_H)
#             if rendered_bgr.shape[1] != traj_W or rendered_bgr.shape[0] != out_H:
#                 rendered_bgr = cv2.resize(rendered_bgr, dsize=(traj_W, out_H), interpolation=cv2.INTER_CUBIC)

#             # Concatenate and write
#             combined_image = np.hstack((video_frame_resized, rendered_bgr))
#             video_writer.write(combined_image)

#             num_frames_visualized += 1
#             pbar.update()
#             if num_frames_visualized >= max_frames:
#                 break

#     cap.release()
#     video_writer.release()

#     if not offscreen:
#         cv2.destroyAllWindows()
#         renderer.destroy_window()
#     print(f"Saved video to {video_out_path}")

 
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import open3d as o3d

def vis_video_aligned_trajectories(
    video_path,
    gripper_widths,
    valids,
    video_out_path,
    poses,
    fps=None,                 # if None, use source fps
    max_frames=-1,
    target_height=360,        # height of the *video* frame (plot matches this; traj is double-height)
    offscreen=True,
):
    """
    Output layout:
      - LEFT (stacked): video (target_height) on top, gripper-width plot (target_height) below.
      - RIGHT: trajectory render, square with height = 2 * target_height.

    Notes:
      - Writes a frame only if valids[i] is True (still advances video + trajectory each step).
      - Requires:
          create_gripper_width_plot_up_to_current_timestamp_x(gripper_widths, frame_timestamps, i)
          _step_time_aligned_trajectories_incremental(offscreen, scene, present_poses, material, include_base_frame=False)
    """
    assert os.path.exists(video_path), f"Missing: {video_path}"
    Path(video_out_path).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    T_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W_src = int(cap.get(cv2.CAP_PROP_WIDTH))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    def _safe_len(x):
        try:
            return len(x)
        except Exception:
            return T_src

    # Use all available aligned lengths
    T_candidates = [T_src, _safe_len(gripper_widths), _safe_len(valids), _safe_len(poses)]
    T_data = min([t for t in T_candidates if t is not None and t > 0])
    T = T_data if max_frames <= 0 else min(T_data, max_frames)
    if T <= 0:
        raise ValueError("No frames to process (after alignment with provided arrays).")

    # Decide output fps
    video_fps = float(fps) if fps is not None else src_fps

    # Video (left-top) resize
    scale = float(target_height) / float(max(H_src, 1))
    out_h = int(round(H_src * scale))         # = target_height
    out_w = int(round(W_src * scale))

    # Plot (left-bottom) matches video size
    left_col_h = out_h * 2                    # stacked video + plot
    left_col_w = out_w

    # Trajectory (right) is a square, height = left column height
    traj_W = left_col_h
    traj_H = left_col_h

    # Final frame size
    combined_h = left_col_h
    combined_w = left_col_w + traj_W

    # Precompute timestamps for plotting
    frame_timestamps = np.arange(_safe_len(gripper_widths), dtype=np.int32)

    # ---- Setup Open3D renderer + stepper ----
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    if offscreen:
        renderer = o3d.visualization.rendering.OffscreenRenderer(traj_W, traj_H)
        scene = renderer.scene
        scene.set_background([1, 1, 1, 1])  # white
    else:
        renderer = o3d.visualization.Visualizer()
        renderer.create_window(visible=True, width=traj_W, height=traj_H)
        scene = renderer

    present_poses = [poses]
    geom_stepper = _step_time_aligned_trajectories_incremental(
        offscreen, scene, present_poses, material, include_base_frame=False
    )

    # ---- VideoWriter (fallback across codecs) ----
    def _open_writer(codec_fourcc: str):
        fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)
        writer = cv2.VideoWriter(str(video_out_path), fourcc, video_fps, (combined_w, combined_h), True)
        return writer

    writer = _open_writer("mp4v")
    if not writer.isOpened():
        writer = _open_writer("avc1")  # needs system H.264
    if not writer.isOpened():
        writer = _open_writer("XVID")
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter. Try a different codec or container (e.g., .avi with 'XVID').")

    # ---- Main loop ----
    with tqdm(total=T, desc="vis_video_aligned_trajectories", leave=False) as pbar:
        for i in range(T):
            ok, frame = cap.read()
            if not ok:
                break

            # LEFT-TOP: video
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

            # LEFT-BOTTOM: gripper width plot up to i
            plot_bytes = create_gripper_width_plot_up_to_current_timestamp_x(
                gripper_widths, frame_timestamps, i
            ).to_image(format="png")
            plot_img = cv2.imdecode(np.frombuffer(plot_bytes, np.uint8), cv2.IMREAD_COLOR)
            plot_img = cv2.resize(plot_img, (out_w, out_h), interpolation=cv2.INTER_AREA)

            left_col = np.vstack([frame, plot_img])  # (2*out_h, out_w, 3)

            # RIGHT: trajectory render
            next(geom_stepper)

            if offscreen:
                # Auto camera
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
                trajectory_image = renderer.capture_screen_float_buffer(do_render=False)  # <- keyword kept
                trajectory_image = (255 * np.asarray(trajectory_image)).astype(np.uint8)
                traj_bgr = cv2.cvtColor(trajectory_image, cv2.COLOR_RGB2BGR)

            # Ensure right square is exactly (traj_W, traj_H)
            if traj_bgr.shape[0] != traj_H or traj_bgr.shape[1] != traj_W:
                traj_bgr = cv2.resize(traj_bgr, (traj_W, traj_H), interpolation=cv2.INTER_CUBIC)

            # Combine: [ left_col | traj_bgr ]
            combined = np.hstack([left_col, traj_bgr])

            # Sanity check
            if combined.shape[0] != combined_h or combined.shape[1] != combined_w:
                raise RuntimeError(
                    f"Frame size {combined.shape[1]}x{combined.shape[0]} != expected {combined_w}x{combined_h}"
                )

            # Write only if valid
            if i < _safe_len(valids) and bool(valids[i]):
                writer.write(combined)

            pbar.update(1)

    writer.release()
    cap.release()

    if not offscreen:
        cv2.destroyAllWindows()
        renderer.destroy_window()

    print(f"Saved video to {video_out_path}")
 
 
def save_trajectory_umi_format(out_csv_path, poses, times, is_lost):
    # output the poses as csv (matching the format in UMI pipeline) which is in quat_xyzw and with relative timestamps starting at 0
    pos_quat_xyzw = pose_4x4_to_quat_xyzw(poses)

    csv_data = OrderedDict({
        'frame_idx': np.arange(len(times)),
        'timestamp': [time - times[0] for time in times],
        'state': [2] * len(times),
        'is_lost': is_lost,
        'is_keyframe': [False] * len(times),
        'x': pos_quat_xyzw[:, 0],
        'y': pos_quat_xyzw[:, 1],
        'z': pos_quat_xyzw[:, 2],
        'q_x': pos_quat_xyzw[:, 3],
        'q_y': pos_quat_xyzw[:, 4],
        'q_z': pos_quat_xyzw[:, 5],
        'q_w': pos_quat_xyzw[:, 6],
    })
    df = pd.DataFrame(csv_data)

    formats = {'timestamp': '{:.6f}'}
    for key in ['x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w']:
        formats[key] = '{:.9f}'

    for col, f in formats.items():
        df[col] = df[col].map(lambda x: f.format(x))

    df.to_csv(out_csv_path, index=False)

if __name__ == "__main__":
    # Create linearly interpolated poses from 0 to 2pi rotation around z and 0 to 10 in position
    angles = np.linspace(0.1, 2*np.pi, 100)
    positions = np.linspace(0, 0.5, 100)
    poses = np.zeros((100, 6))  # [x,y,z, rx,ry,rz] format
    poses[:, 0] = positions  # Linear motion along x-axis
    poses[:, 5] = angles  # Rotate around z-axis
    
    # Convert to 4x4 pose matrices
    poses_4x4 = pose_6d_to_4x4(poses)

    vis_trajectories("tmp_test.mp4", poses_4x4)
