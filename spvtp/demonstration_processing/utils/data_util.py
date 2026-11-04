import logging
import os
import shutil
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy import interpolate
from scipy.spatial import transform as st
from scipy.spatial.transform import Rotation
# from spvtp.diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
import os

import cv2
import ffmpeg
from spvtp.common.episode_arrays import align_valid_frames


# register_codecs()


def convert_homogeneous_matrix(R, p):
    """Create a homogeneous transformation matrix from rotation matrix R and translation vector p."""
    T = np.eye(4)  # Initialize a 4x4 identity matrix
    T[:3, :3] = R  # Set the rotation part
    T[:3, 3] = p  # Set the translation part
    return T


def extract_frames_videos(video_path: str, BGR2RGB=False, resize_ratio=None):
    """
    Extract frames from a video with optional BGR to RGB conversion and resizing

    Parameters:
    video_path (str): Path to the video file
    BGR2RGB (bool): Convert BGR to RGB if True
    resize_ratio (float): Ratio to resize frames (e.g., 0.5 for half size, 2 for double size)
                         None means no resizing

    Returns:
    list: List of extracted frames (optionally processed)
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Apply BGR to RGB conversion if requested
            if BGR2RGB:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply resizing if ratio is provided
            if resize_ratio is not None and resize_ratio > 0:
                new_width = int(frame.shape[1] * resize_ratio)
                new_height = int(frame.shape[0] * resize_ratio)
                frame = cv2.resize(
                    frame,
                    (new_width, new_height),
                    interpolation=(
                        cv2.INTER_AREA if resize_ratio < 1 else cv2.INTER_LINEAR
                    ),
                )

            frames.append(frame)
        else:
            break

    cap.release()
    return frames


def load_images_from_folder(folder_path, BGR2RGB=False, resize_ratio=None):
    """
    Load images from a folder with optional BGR to RGB conversion and resizing

    Parameters:
    folder_path (str): Path to the folder containing images
    BGR2RGB (bool): Convert BGR to RGB if True
    resize_ratio (float): Ratio to resize images (e.g., 0.5 for half size, 2 for double size)
                         None means no resizing

    Returns:
    list: List of loaded (and optionally processed) images
    """
    # Get list of files and sort them numerically
    images = []
    files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
    # Sort files based on the numeric part
    sorted_files = sorted(files, key=lambda x: int(x.replace(".jpg", "")))

    for filename in sorted_files:
        img_path = os.path.join(folder_path, filename)
        # Read image using cv2
        img = cv2.imread(img_path)
        if img is not None:
            if BGR2RGB:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply resizing if ratio is provided
            if resize_ratio is not None and resize_ratio > 0:
                new_width = int(img.shape[1] * resize_ratio)
                new_height = int(img.shape[0] * resize_ratio)
                img = cv2.resize(
                    img,
                    (new_width, new_height),
                    interpolation=(
                        cv2.INTER_AREA if resize_ratio < 1 else cv2.INTER_LINEAR
                    ),
                )

            images.append(img)

    return images


def convert_video_to_images(
    input_video, output_directory, format="jpg", quality=2, start_number=0
):
    """Convert video file to sequence of images (JPG/PNG) using ffmpeg.

    Args:
        input_video: Path to input video file
        output_directory: Directory to save output images
        format: Output format ('jpg' or 'png')
        quality: Image quality (1-31, lower is better)
        start_number: Starting number for image sequence
    """
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Validate format
    format = format.lower()
    if format not in ["jpg", "png"]:
        raise ValueError("Format must be 'jpg' or 'png'")

    # Construct output pattern
    output_pattern = os.path.join(output_directory, f"%05d.{format}")

    try:
        # Setup ffmpeg stream
        stream = ffmpeg.input(input_video)

        # Configure output based on format
        output_args = {"start_number": start_number}
        if format == "jpg":
            output_args["q:v"] = quality

        stream = ffmpeg.output(stream, output_pattern, **output_args)

        # Run the ffmpeg command
        ffmpeg.run(stream, overwrite_output=True)
        print(f"Successfully converted {input_video} to {format.upper()} sequences")

    except ffmpeg.Error as e:
        print(f"An error occurred: {e.stderr.decode()}")


# Configure logger
logger = logging.getLogger(__name__)

class PoseInterpolator:
    def __init__(self, timestamps, homogeneous_matrix):
        positions = homogeneous_matrix[:, :3, 3]
        rotations = Rotation.from_matrix(homogeneous_matrix[:, :3, :3])

        # Find indices where timestamps are not monotonically increasing
        drop_indices = [
            idx
            for idx in range(1, len(timestamps))
            if timestamps[idx] <= timestamps[idx - 1]
        ]

        # Remove the problematic timestamps and corresponding data
        if drop_indices:
            # Log the warning with specific details
            logger.warning(
                "Found %d non-monotonic timestamp(s) that will be removed. "
                "Timestamps: %s. Indices: %s",
                len(drop_indices),
                timestamps[drop_indices],
                drop_indices,
            )

            # Optional: Log additional debugging information
            for idx in drop_indices:
                logger.debug(
                    "Dropping timestamp pair - Previous: %s, Current: %s at index %d",
                    timestamps[idx - 1],
                    timestamps[idx],
                    idx,
                )

            timestamps = np.delete(timestamps, drop_indices)
            positions = np.delete(positions, drop_indices, axis=0)
            rotations = Rotation.from_matrix(
                np.delete(homogeneous_matrix[:, :3, :3], drop_indices, axis=0)
            )

        self.pos_interp = interpolate.interp1d(
            timestamps, positions, kind="linear", axis=0
        )
        self.rot_interp = st.Slerp(timestamps, rotations)

    def __call__(self, ts):
        positions = self.pos_interp(ts)
        rotations = self.rot_interp(ts)
        rotation_matrix = rotations.as_matrix()
        homogeneous_matrix = np.array(
            [
                convert_homogeneous_matrix(R=rot, p=pos)
                for rot, pos in zip(rotation_matrix, positions)
            ]
        )
        return homogeneous_matrix


def interpolate_episode(from_path:str,to_path:str,camera_latency:float,tracking_latency:float,T_ee_cam:np.ndarray):
    from_root = zarr.open(from_path,mode="r")
    to_root = zarr.open(to_path,mode="a")

    camera_data = from_root["camera_1"]
    camera_capture_time:np.ndarray = camera_data["receive_time"][:] # type: ignore
    if camera_latency:
        camera_capture_time = camera_capture_time - camera_latency

    tracking_data = from_root["camera_0"]
    tracking_capture_time:np.ndarray = tracking_data["receive_time"][:] # type: ignore
    if tracking_latency:
        tracking_capture_time = tracking_capture_time - tracking_latency
    # import ipdb; ipdb.set_trace()
    pose:np.ndarray = tracking_data["pose"][:] # type: ignore

    # Find valid time range
    min_time = max( camera_capture_time.min(),
        tracking_capture_time.min(),)
    max_time = min(camera_capture_time.max(),
        tracking_capture_time.max(),)

    # Filter camera timestamps to be within range
    # Create initial boolean mask for time validity
    valid_indices = (camera_capture_time >= min_time) & (
        camera_capture_time <= max_time
    )

    # Combine both conditions - must be within time range AND be a kept frame
    valid_camera_times = camera_capture_time[valid_indices]

    # Create interpolation function and interpolate only valid times
    f_pose = PoseInterpolator(tracking_capture_time, pose)
    pose_interp = f_pose(valid_camera_times)

    # Convert world->camera poses to world->end-effector poses using camera->EE transform from config
    T_cam_ee = np.linalg.inv(T_ee_cam)
    pose_interp = np.array([W_T_cam @ T_cam_ee for W_T_cam in pose_interp])

    print(
        f"Original shape: {len(camera_capture_time)}, After filtering and downsampling: {len(valid_camera_times)}"
    )

    # Prepare training data
    to_root["valid_indices"] = valid_indices
    to_root["pose_interp"] = align_valid_frames(pose_interp, valid_indices)

    
