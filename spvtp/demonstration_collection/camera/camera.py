import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class FrameNumericData:
    pose: Optional[np.ndarray] = None  # The pose matrix (4x4 NumPy array)
    capture_time: Optional[float] = None
    receive_time: Optional[float] = None
    intrinsics: Optional[np.ndarray] = None


@dataclass
class FrameData:
    rgb: np.ndarray
    depth: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None  # The pose matrix (4x4 NumPy array)
    capture_time: Optional[float] = None
    receive_time: Optional[float] = None
    intrinsics: Optional[np.ndarray] = None

    @staticmethod
    def numeric_fields():
        return ["pose", "capture_time", "receive_time", "intrinsics"]


class Camera(ABC):
    def __init__(
        self,
        camera_name: str,
        camera_resolution: Tuple[int, int],
        latency: float = None,
    ) -> None:
        self.camera_name = camera_name
        self.camera_resolution = camera_resolution
        self.intrinsic_matrix = None
        self.latency = latency

        self.is_running = False

    def set_intrinsic_matrix(self, matrix):
        """
        Utility method to set the camera's intrinsic matrix.
        """
        self.intrinsic_matrix = matrix
        print(f"{self.camera_name}: Intrinsic matrix set.")

    def set_latency(self, latency):
        """
        Utility method to set the camera's latency.
        """
        self.latency = latency
        print(f"{self.camera_name}: Latency set.")

    @abstractmethod
    def start_streaming(self):
        pass

    @abstractmethod
    def get_camera_frame(self):
        pass

    @abstractmethod
    def stop_streaming(self):
        pass

    def live_streaming(self):
        start_time = time.monotonic()
        frame_count = 0
        while True:
            frame_data = self.get_camera_frame()
            if frame_data.rgb is not None:
                # Calculate FPS
                current_time = time.monotonic()
                frame_count += 1
                fps = frame_count / (current_time - start_time)

                # Put FPS text on frame
                cv2.putText(
                    frame_data.rgb,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("RGB", frame_data.rgb)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cv2.destroyAllWindows()
