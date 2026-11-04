import os
import sys
import time
import traceback
from pathlib import Path

# from threading import Event, Thread
from multiprocessing import Event
from typing import Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from spvtp.demonstration_collection.camera.camera import Camera, FrameData

try:
    from record3d import Record3DStream
except ImportError:
    Record3DStream = None


def _require_record3d():
    if Record3DStream is None:
        raise ImportError(
            "record3d is required for iPhone capture. "
            "On macOS, install native OpenCV first with `brew install opencv`, "
            "then set `OpenCV_DIR=\"$(brew --prefix opencv)/lib/cmake/opencv4\"` "
            "and run `uv pip install setuptools wheel` followed by "
            "`uv pip install --no-build-isolation record3d`."
        )


class IPhone(Camera):
    def __init__(
        self,
        camera_resolution: Tuple[int, int] = None,
        camera_name: str = "IphoneCamera",
        latency=0.143,
        # latency: float = None,
    ) -> None:
        super().__init__(camera_name, camera_resolution, latency)
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1

    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print("Stream stopped")

    def connect_to_device(self, dev_idx):
        _require_record3d()
        print("Searching for Record3D devices")
        devs = Record3DStream.get_connected_devices()
        print("{} Record3D device(s) found".format(len(devs)))
        for dev in devs:
            print(
                "\tRecord3D Device ID: {}\n\tUDID: {}\n".format(
                    dev.product_id, dev.udid
                )
            )

        if len(devs) <= dev_idx:
            raise RuntimeError(
                "Cannot connect to Record3D device #{}, try different index.".format(
                    dev_idx
                )
            )

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array(
            [[coeffs.fx, 0, coeffs.tx], [0, coeffs.fy, coeffs.ty], [0, 0, 1]]
        )

    def get_camera_frame(self):
        if self.is_running:
            try:
                self.event.wait()  # Wait for new frame to arrive
                # Copy the newly arrived RGBD frame
                depth = self.session.get_depth_frame()
                rgb = self.session.get_rgb_frame()
                if rgb is None:
                    return FrameData(
                        rgb=None,
                        depth=None,
                        pose=None,
                        capture_time=None,
                        receive_time=None,
                    )
                receive_time = time.monotonic()
                if self.latency is not None:
                    capture_time = receive_time - self.latency
                else:
                    capture_time = receive_time
                self.intrinsic_matrix = self.get_intrinsic_mat_from_coeffs(
                    self.session.get_intrinsic_mat()
                )
                camera_pose = (
                    self.session.get_camera_pose()
                )  # Quaternion + world position (accessible via camera_pose.[qx|qy|qz|qw|tx|ty|tz])

                extrinsic_matrix = np.eye(4)
                qx, qy, qz, qw, px, py, pz = (
                    camera_pose.qx,
                    camera_pose.qy,
                    camera_pose.qz,
                    camera_pose.qw,
                    camera_pose.tx,
                    camera_pose.ty,
                    camera_pose.tz,
                )
                # scipy convention is (x, y, z, w)
                extrinsic_matrix[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
                extrinsic_matrix[:3, -1] = [px, py, pz]
                extrinsic_matrix[3, 3] = 1
                # Postprocess it
                if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                    depth = cv2.flip(depth, 1)
                    rgb = cv2.flip(rgb, 1)

                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))

                self.event.clear()  # Clear the event after processing the frame

            except Exception as e:
                traceback.print_exc()
                print("Error: ", e)

            return FrameData(
                rgb=rgb,
                depth=depth,
                pose=extrinsic_matrix,
                capture_time=capture_time,
                receive_time=receive_time,
                intrinsics=self.intrinsic_matrix,
            )

    def start_streaming(self):
        # TODO: assume that the device index is 0
        self.connect_to_device(dev_idx=0)
        time.sleep(1)
        self.is_running = True  # Flag to control the processing loop
        print(f"Camera-{self.camera_name} started.")

    def stop_streaming(self):
        self.is_running = False
        print(f"Camera-{self.camera_name} stopped.")


if __name__ == "__main__":
    cam = IPhone()
    cam.start_streaming()
    cam.live_streaming()
    cam.stop_streaming()
