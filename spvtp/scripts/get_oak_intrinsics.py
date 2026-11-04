"""
Script to get the factory-calibrated intrinsic and distortion parameters of the specific Oak camera device, connected to the laptop.
Ref: https://docs.luxonis.com/software/depthai/examples/calibration_reader/

If post-factory manual calibration is needed, the intrinsic parameters can be obtained by following the calibration procedure here: https://docs.luxonis.com/hardware/platform/depth/calibration/
"""
# TODO @huy-ha: Test this script with physical Oak device
import depthai as dai
import json
import numpy as np

# Create pipeline and device
pipeline = dai.Pipeline()
with dai.Device(pipeline) as device:
    calib = device.readCalibration()
    # Get intrinsics for the RGB camera (OV9782)
    width, height = 1280, 800  # OV9782 resolution
    intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, width, height)
    distortion = calib.getDistortionCoefficients(dai.CameraBoardSocket.RGB)
    # # tmp values for testing
    # intrinsics = np.array([[480, 0, 640],
    #                         [0, 480, 400],
    #                         [0, 0, 1]], dtype=np.float32)
    # distortion = np.array([-0.35, 0.15, 0, 0], dtype=np.float32)

    print("Intrinsic Matrix:", intrinsics)
    print("Distortion Coefficients:", distortion)

    def to_list(x):
        return x.tolist() if hasattr(x, 'tolist') else x

    intrinsics_dict = {
        'intrinsics': to_list(intrinsics),
        'distortion': to_list(distortion),
        'image_width': width,
        'image_height': height
    }

    file_path = 'calibration/oak_intrinsics.json'
    with open(file_path, 'w') as f:
        json.dump(intrinsics_dict, f)

    print(f"Intrinsic parameters saved to: {file_path}")