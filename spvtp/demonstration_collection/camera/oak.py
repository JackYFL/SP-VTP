import time
import traceback
from typing import Optional, Tuple

import cv2
import depthai as dai

from spvtp.demonstration_collection.camera.camera import Camera, FrameData


def get_all_oak_cameras():
    """
    Get all connected OAK cameras.
    """
    devices_id = []
    for device in dai.Device.getAllAvailableDevices():
        print(f"{device.getMxId()} {device.state}")
        devices_id.append(device.getMxId())
    # sort devices_id
    devices_id.sort()
    return devices_id


class Oak(Camera):
    def __init__(
        self,
        camera_name: str,
        camera_resolution: Tuple[int, int] = (1280, 800),
        latency: Optional[float] = 0.119,
        frame_type: str = "isp",
        fps: int = 60,
        device_id: str = None,
    ):
        super().__init__(
            camera_name=camera_name,
            camera_resolution=camera_resolution,
            latency=latency,
        )
        assert frame_type in ["isp", "video", "preview"]
        # Create pipeline
        self.device_id = device_id
        self.pipeline = dai.Pipeline()

        # Define source
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        # Camera properties
        self.fps = fps
        # self.camRgb.setPreviewSize(*self.camera_resolution)
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
        self.camRgb.setFps(fps)

        # stillMjpegOut = self.pipeline.create(dai.node.XLinkOut)
        # stillEncoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)

        # output
        self.ispOut = self.pipeline.create(dai.node.XLinkOut)
        self.videoOut = self.pipeline.create(dai.node.XLinkOut)
        # self.previewOut = self.pipeline.create(dai.node.XLinkOut)

        # Set stream name
        self.ispOut.setStreamName("isp")
        self.videoOut.setStreamName("video")
        # self.previewOut.setStreamName("preview")

        # Linking
        self.camRgb.isp.link(self.ispOut.input)
        self.camRgb.video.link(self.videoOut.input)
        # self.camRgb.preview.link(self.previewOut.input)

        self.is_running = False
        self.frame_type = frame_type

    def get_camera_frame(self):
        if self.is_running:
            try:
                if self.frame_type == "isp":
                    inRgb = self.ispQueue.get()
                elif self.frame_type == "video":
                    inRgb = self.videoQueue.get()
                # elif self.frame_type=="preview":
                #     inRgb = self.previewQueue.get()
                # inRgb = self.qRgb.get()  # Blocking call, waits for a new frame
                receive_time = time.monotonic()

                # Convert to OpenCV format (BGR)
                rgb = inRgb.getCvFrame()

                if self.latency is not None:
                    capture_time = receive_time - self.latency
                else:
                    capture_time = receive_time

                return FrameData(
                    rgb=rgb,
                    capture_time=capture_time,
                    # capture_time= inRgb.getTimestamp(),
                    receive_time=receive_time,
                )
            except Exception as e:
                traceback.print_exc()
                print("Error: ", e)

    def start_streaming(self):
        try:
            if self.device_id is not None:
                device_info = dai.DeviceInfo(self.device_id)  # MXID
                self.device = dai.Device(
                    self.pipeline, device_info, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS
                )
            else:
                self.device = dai.Device(
                    self.pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS
                )
            self.ispQueue = self.device.getOutputQueue(
                name="isp", maxSize=1, blocking=False
            )
            self.videoQueue = self.device.getOutputQueue(
                "video", maxSize=1, blocking=False
            )
            # self.previeQueue = self.device.getOutputQueue('preview',maxSize=1, blocking=False)
            self.is_running = True
            print(f"Camera-{self.camera_name} started.")
            print("Connected cameras:", self.device.getConnectedCameraFeatures())
            print("USB speed:", self.device.getUsbSpeed().name)
            if self.device.getBootloaderVersion() is not None:
                print("Bootloader version:", self.device.getBootloaderVersion())
            print(
                f"Device name: {self.device.getDeviceName()}, Product name: {self.device.getProductName()}"
            )
        except Exception as e:
            traceback.print_exc()
            print("Failed to start streaming:", e)

    def stop_streaming(self):
        self.is_running = False
        print(f"Camera-{self.camera_name} stopped.")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    all_cam = get_all_oak_cameras()
    cam = Oak(camera_name="OAK Camera", device_id=all_cam[0])
    cam.start_streaming()
    cam.live_streaming()
    cam.stop_streaming()
