import time
import traceback
import subprocess
import select
from typing import Optional, Tuple, Union
from collections import deque
import sys
import os

import cv2
import numpy as np

# Assuming these are your original base class references
from spvtp.demonstration_collection.camera.camera import Camera, FrameData

class GoPro(Camera):
    """
    GoPro camera implementation for HDMI Capture Card (UVC).
    Uses ffmpeg to read from /dev/videoX with low latency settings.
    """
    def __init__(
        self,
        camera_name: str,
        camera_resolution: Tuple[int, int] = (1920, 1080),
        latency: Optional[float] = 0.0,
        frame_type: str = "video",
        fps: int = 50,  # Recommendation: Most cheap capture cards only support 30fps at 1080p. Setting 60 might cause errors.
        device_id: Optional[Union[str, int]] = None,
        video_path: Optional[str] = None,
    ):
        super().__init__(
            camera_name=camera_name,
            camera_resolution=camera_resolution,
            latency=latency,
        )
        self.fps = fps
        self.device = device_id if device_id is not None else 0
        self.video_path = os.path.expanduser(video_path) if video_path else None
        self.width, self.height = camera_resolution
        self.frame_size = self.width * self.height * 3  # BGR24 size

        self.proc: Optional[subprocess.Popen] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self._frame_times = deque(maxlen=30)
        self._use_ffmpeg_pipe = False

    def _normalize_linux_device(self) -> str:
        if isinstance(self.device, int):
            return f"/dev/video{self.device}"
        if isinstance(self.device, str) and self.device.isdigit():
            return f"/dev/video{self.device}"
        if self.device is None:
            return "/dev/video0"
        return str(self.device)

    def _start_video_file_stream(self) -> bool:
        if self.video_path is None:
            return False
        if not os.path.exists(self.video_path):
            raise RuntimeError(f"Video file {self.video_path} not found.")

        self.cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open GoPro video file: {self.video_path}")

        self._use_ffmpeg_pipe = False
        print(f"Camera-{self.camera_name} reading from video file {self.video_path}")
        return True

    def _start_macos_stream(self) -> bool:
        device_index = int(self.device)
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open AVFoundation device index {device_index}. "
                "Try a different --gopro-device index in macOS."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._use_ffmpeg_pipe = False
        print(f"Camera-{self.camera_name} started on AVFoundation device {device_index}")
        return True

    def _start_linux_stream(self) -> bool:
        device = self._normalize_linux_device()

        if not os.path.exists(device):
            raise RuntimeError(f"Device {device} not found! Check connection.")

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "v4l2",
            "-input_format",
            "mjpeg",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(self.fps),
            "-i",
            device,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-vsync",
            "drop",
            "pipe:1",
        ]

        print(f"Executing command: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=10**7,
        )

        time.sleep(0.5)
        if self.proc.poll() is not None:
            raise RuntimeError(
                "FFmpeg process exited immediately with code "
                f"{self.proc.returncode}. Check device path, resolution, and FPS."
            )

        self._use_ffmpeg_pipe = True
        print(f"Camera-{self.camera_name} started on {device}")
        return True

    def start_streaming(self):
        try:
            if self.video_path is not None:
                self._start_video_file_stream()
            elif sys.platform == "darwin":
                self._start_macos_stream()
            elif sys.platform.startswith("linux"):
                self._start_linux_stream()
            else:
                self.cap = cv2.VideoCapture(int(self.device))
                if not self.cap.isOpened():
                    raise RuntimeError(
                        f"Unsupported platform {sys.platform} and failed to open camera {self.device}."
                    )
                self._use_ffmpeg_pipe = False

            self.is_running = True
            self._frame_times.clear()
            return True

        except Exception as e:
            traceback.print_exc()
            self.is_running = False
            return False

    def _get_ffmpeg_frame(self):
        if self.proc is None:
            return None
        if self.proc.poll() is not None:
            print("Error: FFmpeg process has died unexpectedly.")
            self.is_running = False
            return None

        r, _, _ = select.select([self.proc.stdout], [], [], 0.01)
        if not r:
            return None

        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) == 0:
            print("Error: FFmpeg stream ended (EOF).")
            self.is_running = False
            return None
        if len(raw) < self.frame_size:
            return None

        return np.frombuffer(raw, np.uint8).reshape(self.height, self.width, 3).copy()

    def _get_capture_frame(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            if self.video_path is not None:
                print(f"Camera-{self.camera_name} reached end of video file.")
                self.is_running = False
            return None
        if frame.shape[:2][::-1] != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height))
        return frame

    def get_camera_frame(self):
        """Reads a single frame from the current GoPro source."""
        if not self.is_running:
            return None

        try:
            if self._use_ffmpeg_pipe:
                frame = self._get_ffmpeg_frame()
            else:
                frame = self._get_capture_frame()

            if frame is None:
                return None

            receive_time = time.monotonic()
            self._frame_times.append(receive_time)
            capture_time = receive_time - (self.latency if self.latency else 0)
            return FrameData(
                rgb=frame,
                capture_time=capture_time,
                receive_time=receive_time,
            )
        except Exception:
            traceback.print_exc()
            return None

    def live_streaming(self):
        """
        Simple loop to display the camera feed for testing.
        """
        print("Starting live preview window...")
        empty_frames_count = 0
        
        while True:
            frame_data = self.get_camera_frame()
            
            # If we keep getting None, the camera might not have started correctly
            if frame_data is None:
                empty_frames_count += 1
                if empty_frames_count % 100 == 0:
                    print("Waiting for frames... (Check terminal for FFmpeg errors)")
                time.sleep(0.01)
                continue
            
            empty_frames_count = 0
            
            if frame_data.rgb is not None:
                # Calculate and display FPS
                fps = self._estimate_fps()
                cv2.putText(frame_data.rgb, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow(f"GoPro - {self.device}", frame_data.rgb)
                
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.stop_streaming()
        cv2.destroyAllWindows()

    def stop_streaming(self):
        self.is_running = False
        if self.proc is not None:
            print("Terminating FFmpeg process...")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print(f"Camera-{self.camera_name} stopped.")
        cv2.destroyAllWindows()

    def _estimate_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        return (len(self._frame_times) - 1) / (self._frame_times[-1] - self._frame_times[0])

if __name__ == "__main__":
    device = 0 if sys.platform == "darwin" else "/dev/video0"

    print(f"Initializing GoPro on {device}...")

    cam = GoPro(camera_name="GoPro", device_id=device, fps=30)

    cam.start_streaming()
    cam.live_streaming()
