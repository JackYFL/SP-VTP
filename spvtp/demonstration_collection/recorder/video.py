import copy
import os
import time
from dataclasses import fields
from queue import Empty, Full, Queue  # Explicitly import exceptions
from threading import Event, Lock, Thread
from typing import List, Optional, Type

import cv2
import numpy as np
import zarr
#from spvtp.diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

from spvtp.common.frame_manager import FrameRateContext
from spvtp.demonstration_collection.camera.camera import (
    Camera,
    FrameData,
    FrameNumericData,
)

# from flyingumi.camera.iphone_camera import IphoneCamera
from spvtp.demonstration_collection.recorder.manager import Recorder

#register_codecs()


class VideoRecorder(Recorder):
    def __init__(
        self,
        record_fps: int,
        stream_fps: int,
        video_record_path: str,
        camera_sources: List[Camera],
        frame_data_class: Type[FrameData],
        verbose: bool = False,
        convert_bgr_to_rgb: bool = False,
        max_workers: int = 32,
        frame_queue_size: int = 20,
    ) -> None:
        self.video_record_path: str = video_record_path
        self.camera_sources: List[Camera] = camera_sources
        camera_names = [camera.camera_name for camera in camera_sources]
        if np.unique(camera_names).size != len(camera_names):
            raise ValueError("Camera names must be unique.")
        self.frame_data_class: Type[FrameNumericData] = frame_data_class
        self.verbose: bool = verbose
        self._streaming_event: Event = Event()
        self._recording_event: Event = Event()
        self.frame_queues: List[Queue] = [
            Queue(maxsize=frame_queue_size) for _ in camera_sources
        ]
        self.record_fps: int = record_fps
        self.stream_fps: int = stream_fps
        self.record_dt: float = 1 / record_fps
        self.convert_bgr_to_rgb: bool = convert_bgr_to_rgb
        # Create directory if it doesn't exist
        os.makedirs(video_record_path, exist_ok=True)
        self.record_roots = zarr.open(video_record_path, mode="a")
        self.max_workers = max_workers

        # Initialize thread lists
        self.camera_threads: List[Thread] = []
        self.recording_threads: List[Thread] = []

        self._episode_id = self.init_episode_id()
        print(f"Starting episode {self.episode_id}")

    @property
    def episode_id(self):
        return self._episode_id

    def init_episode_id(self):
        ids = [int(key.removeprefix('episode_')) for key in self.record_roots.group_keys()
               if key.startswith('episode_') and key.removeprefix('episode_').isdigit()]
        return max(ids, default=-1) + 1

    def set_episode_id(self, episode_id: int = None) -> None:
        self._episode_id = episode_id

    def reset_episode_recording(self):
        if self._recording_event.is_set():
            """
            call this function when recording already start will cause no data to be logged.
            the reference passed to the record thread will be replaced.
            """
            return False
        os.makedirs(
            os.path.join(self.video_record_path, f"episode_{self.episode_id}"),
            exist_ok=True,
        )
        # Delete existing video files if they exist
        for i in range(len(self.camera_sources)):
            video_path = os.path.join(
                self.video_record_path, f"episode_{self.episode_id}", f"camera_{i}.mp4"
            )
            if os.path.exists(video_path):
                os.remove(video_path)

        self.video_writers: List[Optional[cv2.VideoWriter]] = [
            None for _ in self.camera_sources
        ]
        self.episode_frames: List[dict] = [{} for _ in self.camera_sources]
        for i in range(len(self.camera_sources)):
            self.episode_frames[i] = {
                field.name: [] for field in fields(self.frame_data_class)
            }

        return True

    def clear_recording(self):
        """Delete the current episode's folder and reset episode ID."""
        if self._recording_event.is_set():
            print("Cannot delete while recording is in progress.")
            return False

        episode_path = os.path.join(
            self.video_record_path, f"episode_{self.episode_id}"
        )
        try:
            if os.path.exists(episode_path):
                # Delete all files in the episode directory
                for file in os.listdir(episode_path):
                    file_path = os.path.join(episode_path, file)
                    os.remove(file_path)
                # Remove the directory itself
                os.rmdir(episode_path)
                print(f"Deleted episode {self.episode_id}")
                return True
        except Exception as e:
            print(f"Error deleting episode: {e}")
        return False

    def _camera_stream_thread(self, camera_idx):
        """Thread function to continuously get frames from a camera"""
        camera = self.camera_sources[camera_idx]
        queue = self.frame_queues[camera_idx]
        start_time = time.monotonic()
        frame_received = 0
        while self._streaming_event.is_set():
            try:
                with FrameRateContext(self.stream_fps, verbose=self.verbose) as fr:
                    frame_data = camera.get_camera_frame()
                    if frame_data is not None and frame_data.rgb is not None:
                        # Check if queue is full first
                        frame_received += 1
                        if self.verbose:
                            actual_fps = frame_received / (
                                time.monotonic() - start_time
                            )
                            # print(
                            #     f"Actual FPS for camera {camera_idx} streaming: {actual_fps}"
                            # )
                        if queue.full():
                            try:
                                queue.get_nowait()  # Remove oldest frame
                                if self.verbose:
                                    print(
                                        f"Queue full for camera {camera_idx}, removed oldest frame"
                                    )
                            except Empty:
                                pass

                        try:
                            queue.put_nowait(frame_data)
                            if self.verbose:
                                print(f"Put frame queued from camera {camera_idx}")
                                # print("queue size", queue.qsize())
                        except Full:
                            if self.verbose:
                                print(f"Failed to put frame for camera {camera_idx}")
                    else:
                        print(f"No frame data from camera {camera_idx}")
            except Exception as e:
                print(f"Error capturing frame from camera {camera_idx}: {e}")
                time.sleep(0.1)

    def _recording_thread(self, camera_idx, episode_id):
        """Thread function to process and store frames from the queue"""
        stored_frame = self.episode_frames[camera_idx]
        video_path = os.path.join(
            self.video_record_path,
            f"episode_{episode_id}",
            f"camera_{camera_idx}.mp4",
        )
        fps = self.record_fps
        # Get first frame to determine dimensions and initialize video writer
        first_frame = self.peek_latest_frame(camera_idx)
        while first_frame is None:
            print("Waiting for first frame to initialize video writer...")
            first_frame = self.peek_latest_frame(camera_idx)

        frame = first_frame.rgb
        height, width = frame.shape[:2]
        print("video_path", video_path)
        self.video_writers[camera_idx] = cv2.VideoWriter(  # Store in instance variable
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if self.verbose:
            print(f"Initialized video writer for camera {camera_idx}")
        video_writer = self.video_writers[camera_idx]  # Use local reference
        try:
            last_timestamp = first_frame.receive_time
            start_time = time.monotonic()
            frame_received = 0

            while self._recording_event.is_set():
                with FrameRateContext(fps, verbose=self.verbose) as fr:
                    # import ipdb; ipdb.set_trace()
                    frame_data = self.get_next_frame(
                        camera_idx, last_timestamp, timeout=1 / fps
                        # camera_idx, last_timestamp, timeout=100

                    )
                    if frame_data is None:
                        print("No frames in queue!")
                        continue
                    else:
                        frame_received += 1
                        last_timestamp = frame_data.receive_time
                        actual_fps = frame_received / (time.monotonic() - start_time)
                        if self.verbose:
                            print(
                                f"Actual FPS for camera_{camera_idx} recording: {actual_fps}"
                            )

                    for field in fields(self.frame_data_class):
                        value = getattr(frame_data, field.name)
                        if value is not None:
                            if field.name == "rgb":
                                if self.convert_bgr_to_rgb:
                                    value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
                                video_writer.write(value)
                                if self.verbose:
                                    print(
                                        f"Written frame to video for camera {camera_idx}"
                                    )
                            if field.name in self.frame_data_class.numeric_fields():
                                stored_frame[field.name].append(value)
        except Exception as e:
            print(f"Error processing frame from camera {camera_idx}: {e}")

    def peek_latest_frame(self, camera_idx: int) -> Optional[FrameData]:
        """Peek the latest frame from the queue without removing it"""
        if 0 <= camera_idx < len(self.frame_queues):
            queue = self.frame_queues[camera_idx]
            # print("queue size", queue.qsize())
            with queue.mutex:
                if queue.queue:
                    # print(f"Peeked frame from camera {camera_idx}")
                    frame = queue.queue[-1]
                    # if self.verbose:
                    #     print(f"Peeked frame from camera {camera_idx}")
                    return frame
            # print("No frames in queue.")
        return None

    def get_last_k_frames(
        self, camera_idx: int, k: int, remove: bool = False
    ) -> List[FrameData]:
        """
        Get the latest k frames from the queue with option to remove them.

        Args:
            camera_idx: Camera index
            k: Number of frames to get
            remove: Whether to remove the frames from queue
        """
        if 0 <= camera_idx < len(self.frame_queues):
            queue = self.frame_queues[camera_idx]
            with queue.mutex:
                if queue.queue:
                    queue_list = list(queue.queue)
                    frames = queue_list[-k:]
                    if remove:
                        # Clear the retrieved frames from queue
                        for _ in range(min(k, len(queue_list))):
                            queue.queue.pop()
                    return frames
        return None

    def get_last_k_frames_from_all(self, k: int, remove: bool = False) -> dict:
        """Get the latest k frames from all cameras.

        Args:
            k (int): Number of latest frames to retrieve

        Returns:
            dict: Dictionary with camera names as keys and lists of frames as values
        """
        frames_dict = {}
        for i, camera in enumerate(self.camera_sources):
            camera_name = getattr(camera, "camera_name", f"camera_{i}")
            # print(f"Getting last {k} frames from {camera_name}...")
            frames = self.get_last_k_frames(i, k, remove)
            print(f"Retrieved {len(frames) if frames else 0} frames from {camera_name}")
            if frames is not None:
                frames_dict[camera_name] = frames
            else:
                frames_dict[camera_name] = None

        return frames_dict

    def get_next_frame(
        self,
        camera_idx: int,
        last_timestamp: Optional[float] = None,
        timeout: float = 1.0,
    ) -> Optional[FrameData]:
        """Get the newest frame from the queue that has a receive_time larger than the last timestamp,
        waiting if necessary until a newer frame arrives.

        Args:
            camera_idx: Index of the camera to get frame from
            last_timestamp: The timestamp of the last frame that was processed.
                        If None, will return the newest frame in queue.
            timeout: Maximum time to wait for a newer frame in seconds

        Returns:
            Optional[FrameData]: The newest frame with receive_time > last_timestamp, or None if no such frame exists
        """
        if not 0 <= camera_idx < len(self.frame_queues):
            return None

        queue = self.frame_queues[camera_idx]
        start_time = time.monotonic()

        while (time.monotonic() - start_time) < timeout:
            with queue.mutex:
                if not queue.queue:
                    continue

                newest_frame = queue.queue[-1]
                if last_timestamp is None or (
                    newest_frame.receive_time is not None
                    and newest_frame.receive_time > last_timestamp
                ):
                    return newest_frame

            time.sleep(0.001)

        return None

    def start_streaming(self):
        if self._streaming_event.is_set():
            print("Streaming is already in progress.")
            return False

        # Set streaming event
        self._streaming_event.set()
        # Start camera streams
        print("Starting camera streams...")
        started_cameras = []
        for camera in self.camera_sources:
            if hasattr(camera, "start_streaming"):
                success = camera.start_streaming()
                if success is False:
                    print(f"Failed to start camera {camera.camera_name}")
                    self._streaming_event.clear()
                    for started_camera in started_cameras:
                        if hasattr(started_camera, "stop_streaming"):
                            started_camera.stop_streaming()
                    return False
                started_cameras.append(camera)

        # Give cameras time to initialize
        time.sleep(0.5)

        # Then start camera streaming threads
        print("Starting camera threads...")
        self.camera_threads = []
        for i in range(len(self.camera_sources)):
            thread = Thread(target=self._camera_stream_thread, args=(i,))
            thread.daemon = True
            thread.start()
            self.camera_threads.append(thread)

        return True

    def start_recording(self, episode_id=None):
        """Start recording from all cameras"""
        if episode_id is not None:
            self.set_episode_id(episode_id)
        if self._recording_event.is_set():
            print("Recording is already in progress.")
            return False
        self._recording_event.set()
        time.sleep(0.5)
        try:
            print("Starting recording threads...")
            self.recording_threads = []
            for i in range(len(self.camera_sources)):
                thread = Thread(
                    target=self._recording_thread, args=(i, self.episode_id)
                )
                thread.daemon = True
                thread.start()
                self.recording_threads.append(thread)

            print("Recording started successfully.")
            return True

        except Exception as e:
            print(f"Failed to start recording: {e}")
            self.stop_recording()
            return False

    def stop_streaming(self):
        """Stop streaming from all cameras"""
        if not self._streaming_event.is_set():
            print("Streaming is not in progress.")
            return False

        print("Stopping streaming...")
        self._streaming_event.clear()

        # Wait for threads to finish
        print("Waiting for threads to finish...")
        for thread in self.camera_threads:
            thread.join(timeout=2.0)

        # Stop camera streams
        print("Stopping camera streams...")
        for camera in self.camera_sources:
            try:
                if hasattr(camera, "stop_streaming"):
                    camera.stop_streaming()
            except Exception as e:
                print(f"Error stopping camera: {e}")

        print("Streaming stopped successfully.")

        return True

    def stop_recording(self):
        """Stop recording and save all data"""
        if not self._recording_event.is_set():
            print("Recording is not in progress.")
            return False

        print("Stopping recording...")
        self._recording_event.clear()

        # Wait for threads to finish
        print("Waiting for threads to finish...")
        for thread in self.recording_threads:
            thread.join(timeout=2.0)

        return True

    def save_recordings(self):
        """Save all recorded frames to disk"""
        this_episode_id = self._episode_id
        episode_data = self.record_roots.require_group(f"episode_{this_episode_id}")

        # Release all video writers
        print("Releasing video writers...")
        for i, writer in enumerate(self.video_writers):
            if writer is not None:
                writer.release()
                # self.video_writers[i] = None
        for i in range(len(self.camera_sources)):
            if self.verbose:
                print(f"Writing camera_{i} data to disk...")
            episode_camera_data = episode_data.require_group(f"camera_{i}")
            episode_to_save = self.episode_frames[i]

            for key, value_list in episode_to_save.items():
                if not value_list:
                    if self.verbose:
                        print(f"No {key} data to save for camera_{i}")
                    continue
                if self.verbose:
                    print(f"Writing {key} data...")
                episode_camera_data[key] = value_list

        self.set_episode_id(this_episode_id + 1)
        print("Recording saved successfully.")

    @property
    def is_recording(self) -> bool:
        """Check if currently recording video.

        Returns:
            bool: True if recording is in progress, False otherwise.
        """
        return self._recording_event.is_set()


if __name__ == "__main__":
    import time

    # Test code
    #all_cam = get_all_oak_cameras()
    #cam_0 = OakCamera("oak_0", device_id=all_cam[0])
    #cam_1 = OakCamera("oak_1", device_id=all_cam[1])
    video_recorder = VideoRecorder(
        record_fps=30,
        video_record_path=os.path.expanduser("~/Dev/DexUMI/data_local/video"),
        frame_data_class=FrameData,
        #camera_sources=[cam_0, cam_1],
        # camera_sources=[cam_0],
        verbose=False,
        convert_bgr_to_rgb=False,
    )
    fps = 30
    dt = 1 / fps
    video_recorder.start_streaming()
    start_time = time.monotonic()
    iter = 0
    while True:
        end_time = start_time + (1 + iter) * dt
        frames_dict = video_recorder.get_lask_k_frames_from_all_cameras(k=1)

        # Show frames from all cameras
        frames_to_show = []
        for camera_name, frames in frames_dict.items():
            if frames and frames[0] is not None:
                # Get first frame (k=1)
                frame_data = frames[0]
                # Add episode number overlay
                viz_frame = copy.deepcopy(frame_data.rgb)
                cv2.putText(
                    viz_frame,
                    f"Episode: {video_recorder.episode_id} - {camera_name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                frames_to_show.append(viz_frame)

        if frames_to_show:
            # Calculate dimensions
            n_frames = len(frames_to_show)

            # Ensure all frames have same dimensions by resizing
            h, w = frames_to_show[0].shape[:2]
            frames_to_show = [cv2.resize(frame, (w, h)) for frame in frames_to_show]

            # Concatenate horizontally
            combined_frame = np.hstack(frames_to_show)

            # Show combined frame
            cv2.imshow("All Cameras", combined_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            if video_recorder.stop_recording():
                video_recorder.clear_recording()
            video_recorder.stop_streaming()
            break
        elif key == ord("s"):
            if video_recorder.reset_episode_recording():
                print("Starting recording...")
                video_recorder.start_recording()
            else:
                print("Recording already started.")
        elif key == ord("w"):
            print("Saving recording...")
            if video_recorder.stop_recording():
                video_recorder.save_recordings()
        elif key == ord("a"):
            print("Restarting episode...")
            if video_recorder.stop_recording():
                video_recorder.clear_recording()

        if time.monotonic() < end_time:
            time.sleep(end_time - time.monotonic())
        iter += 1
