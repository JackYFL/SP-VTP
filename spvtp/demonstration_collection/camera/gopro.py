import time
import traceback
from typing import Optional, Tuple

import av
import cv2
import numpy as np

from spvtp.demonstration_collection.camera.camera import Camera, FrameData


class GoPro(Camera):
    """
    Camera class for GoPro 10 (or other action cameras that record video files).
    Reads video frames from a GoPro video file or USB stream.
    """

    def __init__(
        self,
        camera_name: str = "GoProCamera",
        camera_resolution: Tuple[int, int] = (1920, 1440),
        latency: Optional[float] = 0.0,
        fps: int = 60,
        video_path: Optional[str] = None,
        device_index: int = 0,
    ):
        """
        Initialize GoPro camera.
        
        Args:
            camera_name: Name of the camera
            camera_resolution: Target resolution (width, height)
            latency: Camera latency in seconds
            fps: Recording frame rate
            video_path: Path to GoPro video file (if None, will try to stream from device)
            device_index: Device index for USB camera (if video_path is None)
        """
        super().__init__(
            camera_name=camera_name,
            camera_resolution=camera_resolution,
            latency=latency,
        )
        self.fps = fps
        self.video_path = video_path
        self.device_index = device_index
        
        # Video capture related
        self.cap = None
        self.container = None
        self.stream = None
        self.frame_index = 0
        
        # Load intrinsics from GoPro calibration if available
        self.load_gopro_intrinsics()

    def load_gopro_intrinsics(self):
        """Load GoPro camera intrinsics from calibration file."""
        try:
            import json
            import os
            
            # Try to load from calibration directory
            calibration_paths = [
                'calibration/gopro_intrinsics_2_7k.json',
                './calibration/gopro_intrinsics_2_7k.json',
                '../calibration/gopro_intrinsics_2_7k.json',
            ]
            
            for cal_path in calibration_paths:
                if os.path.exists(cal_path):
                    with open(cal_path, 'r') as f:
                        calib_data = json.load(f)
                    
                    # Extract intrinsic matrix from calibration
                    if 'camera_matrix' in calib_data:
                        self.intrinsic_matrix = np.array(calib_data['camera_matrix'])
                    elif 'intrinsics' in calib_data:
                        self.intrinsic_matrix = np.array(calib_data['intrinsics'])
                    
                    print(f"{self.camera_name}: Loaded intrinsics from {cal_path}")
                    return
        except Exception as e:
            print(f"{self.camera_name}: Warning - Could not load intrinsics: {e}")
        
        # Set default intrinsics for GoPro Hero 10 at various resolutions
        # These are approximate values and should be calibrated for your specific setup
        if self.camera_resolution == (1920, 1440):
            # Standard GoPro 2.7K resolution
            fx = 1358.0
            fy = 1356.0
            cx = 960.0
            cy = 720.0
        elif self.camera_resolution == (1280, 720):
            # Lower resolution
            fx = 905.0
            fy = 904.0
            cx = 640.0
            cy = 360.0
        else:
            # Default fallback
            fx = self.camera_resolution[0]
            fy = self.camera_resolution[1]
            cx = self.camera_resolution[0] / 2
            cy = self.camera_resolution[1] / 2
        
        self.intrinsic_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

    def get_camera_frame(self):
        """Get the next frame from GoPro video."""
        if not self.is_running:
            return FrameData(rgb=None)
        
        try:
            if self.video_path is not None:
                # Reading from video file
                ret, frame = self.cap.read()
                if not ret:
                    print(f"{self.camera_name}: End of video reached or error reading frame")
                    return FrameData(rgb=None)
                
                receive_time = time.monotonic()
            else:
                # Reading from USB camera device
                ret, frame = self.cap.read()
                if not ret:
                    print(f"{self.camera_name}: Error reading from camera device")
                    return FrameData(rgb=None)
                
                receive_time = time.monotonic()
            
            # Resize if necessary
            if frame.shape[:2][::-1] != self.camera_resolution:
                frame = cv2.resize(frame, self.camera_resolution)
            
            # Convert BGR to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.latency is not None:
                capture_time = receive_time - self.latency
            else:
                capture_time = receive_time
            
            self.frame_index += 1
            
            return FrameData(
                rgb=frame_rgb,
                capture_time=capture_time,
                receive_time=receive_time,
                intrinsics=self.intrinsic_matrix,
            )
        except Exception as e:
            traceback.print_exc()
            print(f"{self.camera_name} Error: {e}")
            return FrameData(rgb=None)

    def start_streaming(self):
        """Start streaming from GoPro video file or USB camera."""
        try:
            if self.video_path is not None:
                # Open video file
                if not self._open_video_file():
                    return False
            else:
                # Open USB camera device
                if not self._open_usb_device():
                    return False
            
            self.is_running = True
            print(f"Camera-{self.camera_name} started.")
            return True
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to start streaming: {e}")
            return False

    def _open_video_file(self) -> bool:
        """Open GoPro video file using OpenCV or PyAV."""
        try:
            # Try OpenCV first (faster)
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video file: {self.video_path}")
            
            # Get video properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"{self.camera_name}: Video file opened")
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps}")
            
            return True
        except Exception as e:
            print(f"Failed to open video file with OpenCV: {e}")
            try:
                # Fallback to PyAV
                self.container = av.open(self.video_path)
                self.stream = self.container.streams.video[0]
                print(f"{self.camera_name}: Video file opened with PyAV")
                return True
            except Exception as e2:
                print(f"Failed to open video file with PyAV: {e2}")
                return False

    def _open_usb_device(self) -> bool:
        """Open GoPro camera as USB device."""
        try:
            # Open camera device (usually /dev/video0 or /dev/video1 on Linux)
            self.cap = cv2.VideoCapture(self.device_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera device: {self.device_index}")
            
            # Set resolution and FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Get actual properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"{self.camera_name}: USB camera opened (device {self.device_index})")
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps}")
            
            return True
        except Exception as e:
            print(f"Failed to open USB device: {e}")
            return False

    def stop_streaming(self):
        """Stop streaming and clean up resources."""
        self.is_running = False
        
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            if self.container is not None:
                self.container.close()
                self.container = None
            
            print(f"Camera-{self.camera_name} stopped.")
        except Exception as e:
            print(f"Error stopping camera: {e}")
        
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test GoPro camera with USB device
    cam = GoPro(camera_name="GoPro", device_index=0)
    cam.start_streaming()
    cam.live_streaming()
    cam.stop_streaming()
