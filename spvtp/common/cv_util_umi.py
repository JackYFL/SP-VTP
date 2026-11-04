"""
Adapted from spvtp.demonstration_processing.umi.common.cv_util from UMI
"""
from typing import Dict, Tuple, Optional

import copy
import numpy as np
import cv2
import os
import scipy.interpolate as si

# =================== intrinsics ===================

def parse_fisheye_intrinsics_opencv(json_data: dict) -> Dict[str, np.ndarray]:
    """
    Reads camera intrinsics from OpenCameraImuCalibration to opencv format.
    Example:
    {
        "final_reproj_error": 0.17053819312281043,
        "fps": 60.0,
        "image_height": 1080,
        "image_width": 1920,
        "intrinsic_type": "FISHEYE",
        "intrinsics": {
            "aspect_ratio": 1.0026582765352035,
            "focal_length": 420.56809123853304,
            "principal_pt_x": 959.857586309181,
            "principal_pt_y": 542.8155851051391,
            "radial_distortion_1": -0.011968137016185161,
            "radial_distortion_2": -0.03929790706019372,
            "radial_distortion_3": 0.018577224235396064,
            "radial_distortion_4": -0.005075629959840777,
            "skew": 0.0
        },
        "nr_calib_images": 129,
        "stabelized": false
    }
    """
    assert json_data['intrinsic_type'] == 'FISHEYE'
    intr_data = json_data['intrinsics']
    
    # img size
    h = json_data['image_height']
    w = json_data['image_width']

    # pinhole parameters
    f = intr_data['focal_length']
    px = intr_data['principal_pt_x']
    py = intr_data['principal_pt_y']
    
    # Kannala-Brandt non-linear parameters for distortion
    kb8 = [
        intr_data['radial_distortion_1'],
        intr_data['radial_distortion_2'],
        intr_data['radial_distortion_3'],
        intr_data['radial_distortion_4']
    ]

    opencv_intr_dict = {
        'DIM': np.array([w, h], dtype=np.int64),
        'K': np.array([
            [f, 0, px],
            [0, f, py],
            [0, 0, 1]
        ], dtype=np.float64),
        'D': np.array([kb8]).T
    }
    return opencv_intr_dict

def parse_fisheye_intrinsics_oak(json_data: dict) -> Dict[str, np.ndarray]:
    """
    Reads camera intrinsics from OpenCameraImuCalibration to opencv format.
    Example:
    {
        "image_height": 800,
        "image_width": 1280,
        "intrinsics": np.ndarray, # 3x3
        "distortion": np.ndarray, # 1x4 
    }
    """
    # img size
    h = json_data['image_height']
    w = json_data['image_width']

    intr_data = json_data['intrinsics']
    distortion = json_data['distortion']

    opencv_intr_dict = {
        'DIM': np.array([w, h], dtype=np.int64),
        'K': np.array(intr_data, dtype=np.float64),
        'D': np.array(distortion, dtype=np.float64).reshape(-1, 1)
    }
    return opencv_intr_dict


def convert_fisheye_intrinsics_resolution(
        opencv_intr_dict: Dict[str, np.ndarray], 
        target_resolution: Tuple[int, int]
        ) -> Dict[str, np.ndarray]:
    """
    Convert fisheye intrinsics parameter to a different resolution,
    assuming that images are not cropped in the vertical dimension,
    and only symmetrically cropped/padded in horizontal dimension.
    """
    iw, ih = opencv_intr_dict['DIM']
    iK = opencv_intr_dict['K']
    ifx = iK[0,0]
    ify = iK[1,1]
    ipx = iK[0,2]
    ipy = iK[1,2]

    ow, oh = target_resolution
    ofx = ifx / ih * oh
    ofy = ify / ih * oh
    opx = (ipx - (iw / 2)) / ih * oh + (ow / 2)
    opy = ipy / ih * oh
    oK = np.array([
        [ofx, 0, opx],
        [0, ofy, opy],
        [0, 0, 1]
    ], dtype=np.float64)

    out_intr_dict = copy.deepcopy(opencv_intr_dict)
    out_intr_dict['DIM'] = np.array([ow, oh], dtype=np.int64)
    out_intr_dict['K'] = oK
    return out_intr_dict


class FisheyeRectConverter:
    def __init__(self, K, D, DIM, out_size, out_fov):
        out_size = np.array(out_size)
        # vertical fov
        out_f = (out_size[1] / 2) / np.tan(out_fov/180*np.pi/2)
        out_K = np.array([
            [out_f, 0, out_size[0]/2],
            [0, out_f, out_size[1]/2],
            [0, 0, 1]
        ], dtype=np.float32)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), out_K, out_size, cv2.CV_16SC2)

        self.map1 = map1
        self.map2 = map2
    
    def forward(self, img):
        rect_img = cv2.remap(img, 
            self.map1, self.map2,
            interpolation=cv2.INTER_AREA, 
            borderMode=cv2.BORDER_CONSTANT)
        return rect_img


# ================= ArUcO tag =====================
def parse_aruco_config(aruco_config_dict: dict):
    """
    example:
    aruco_dict:
        predefined: DICT_4X4_50
    marker_size_map: # all unit in meters
        default: 0.15
        12: 0.2
    """
    aruco_dict = get_aruco_dict(**aruco_config_dict['aruco_dict'])

    n_markers = len(aruco_dict.bytesList)
    marker_size_map = aruco_config_dict['marker_size_map']
    default_size = marker_size_map.get('default', None)
    
    out_marker_size_map = dict()
    for marker_id in range(n_markers):
        size = default_size
        if marker_id in marker_size_map:
            size = marker_size_map[marker_id]
        out_marker_size_map[marker_id] = size
    
    result = {
        'aruco_dict': aruco_dict,
        'marker_size_map': out_marker_size_map
    }
    return result


def get_aruco_dict(predefined:str
                   ) -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, predefined))


 

# def detect_localize_aruco_tags(
#         img: np.ndarray, 
#         aruco_dict: cv2.aruco.Dictionary, 
#         marker_size_map: Dict[int, float], 
#         fisheye_intr_dict: Dict[str, np.ndarray], 
#         refine_subpix: bool=True,
#         visualize: bool=False,
#         draw_axes: bool=False,
#         corner_radius: int=3,
#         save_path: Optional[str]=None
#     ) -> Tuple[Dict[int, dict], np.ndarray]:
#     """
#     Detects and localizes ArUco tags. Optionally visualizes detected corners/axes and saves image.

#     Args:
#         img: Input image.
#         aruco_dict: cv2.aruco.Dictionary object.
#         marker_size_map: {marker_id: size_in_meters}.
#         fisheye_intr_dict: {'K': intrinsics_matrix, 'D': distortion_coeffs}.
#         refine_subpix: If True, applies corner subpixel refinement.
#         visualize: If True, draw detected markers, corners, and optional pose axes.
#         draw_axes: If True (and visualize=True), draw pose axes.
#         corner_radius: Pixel radius for drawn corner circles.
#         save_path: If given and visualize=True, save visualization image here.

#     Returns:
#         tag_dict: {marker_id: {'rvec', 'tvec', 'corners'}}.
#         img_vis: Visualization image (same as input if visualize=False).
#     """
#     K = fisheye_intr_dict['K']
#     D = fisheye_intr_dict['D']

#     param = cv2.aruco.DetectorParameters()
#     if refine_subpix:
#         param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

#     corners, ids, rejected = cv2.aruco.detectMarkers(
#         image=img, dictionary=aruco_dict, parameters=param
#     )

#     img_vis = img.copy() if visualize else img

#     if corners is None or len(corners) == 0:
#         if visualize and save_path is not None:
#             cv2.imwrite(save_path, img_vis)
#         return dict()

#     if visualize:
#         cv2.aruco.drawDetectedMarkers(img_vis, corners, ids)

#     tag_dict = {}
#     for this_id, this_corners in zip(ids, corners):
#         this_id = int(this_id[0])

#         if visualize:
#             for (x, y) in this_corners.reshape(-1, 2):
#                 cv2.circle(img_vis, (int(x), int(y)), corner_radius, (0, 255, 255), -1)

#         if this_id not in marker_size_map:
#             continue

#         marker_size_m = marker_size_map[this_id]
#         undistorted = cv2.fisheye.undistortPoints(this_corners, K, D, P=K)

#         rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
#             undistorted, marker_size_m, K, np.zeros((1, 5))
#         )

#         rvec = rvec.squeeze()
#         tvec = tvec.squeeze()

#         tag_dict[this_id] = {
#             'rvec': rvec,
#             'tvec': tvec,
#             'corners': this_corners.squeeze()
#         }

#         if visualize and draw_axes:
#             axis_len = float(marker_size_m) * 0.5
#             cv2.aruco.drawAxis(img_vis, K, np.zeros((1, 5)), rvec, tvec, axis_len)

#     # Save visualization if requested
#     if visualize and save_path is not None:
#         cv2.imwrite(save_path, img_vis)

#     return tag_dict

import cv2
from pathlib import Path
from typing import Union, Iterable

def images_to_video(
    img_dir: Union[str, Path],
    out_path: Union[str, Path],
    fps: int = 30,
    ext: str = ".png",
    codec: str = "mp4v",   # good default; try "avc1" if you have H.264 installed
    resize_to_first: bool = True,
) -> str:
    """
    Make a video from images named like '0.png', '1.png', ... in numeric order.

    Args:
        img_dir: Directory containing images.
        out_path: Path to output video file (e.g., 'out.mp4').
        fps: Frames per second for the output.
        ext: Image extension to read (default '.png').
        codec: FourCC codec code for OpenCV VideoWriter.
        resize_to_first: If True, resize subsequent frames to the first frame's size.

    Returns:
        The output video path (string).
    """
    img_dir = Path(img_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect files with integer stems only, e.g., "0.png", "12.png"
    frames: Iterable[Path] = []
    for p in img_dir.glob(f"*{ext}"):
        try:
            int(p.stem)  # ensure numeric stem
            frames.append(p)
        except ValueError:
            continue

    frames = sorted(frames, key=lambda p: int(p.stem))
    if not frames:
        raise ValueError(f"No images like '<number>{ext}' found in {img_dir}")

    # Read first frame to determine size
    first = cv2.imread(str(frames[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise ValueError(f"Failed to read {frames[0]}")
    h, w = first.shape[:2]

    # Initialize writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), True)
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter. Try a different codec (e.g., 'avc1' or 'XVID').")

    # Write frames
    for fp in frames:
        img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[warn] Skipping unreadable image: {fp}")
            continue

        # Ensure 3-channel BGR for video (convert gray or RGBA as needed)
        if img.ndim == 2:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if resize_to_first and (img.shape[1] != w or img.shape[0] != h):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        writer.write(img)

    writer.release()
    return str(out_path)


def detect_localize_aruco_tags(
        img: np.ndarray, 
        aruco_dict: cv2.aruco.Dictionary, 
        marker_size_map: Dict[int, float], 
        fisheye_intr_dict: Dict[str, np.ndarray], 
        refine_subpix: bool=True,
        visualize: bool=False,
        draw_axes: bool=False,
        corner_radius: int=3,
        frame_idx: int=0,
        save_dir: Optional[str]=None
    ) -> Dict[int, dict]:
    """
    Detects and localizes ArUco tags. Optionally visualizes detected corners/axes and saves image.

    Args:
        img: Input image.
        aruco_dict: cv2.aruco.Dictionary object.
        marker_size_map: {marker_id: size_in_meters}.
        fisheye_intr_dict: {'K': intrinsics_matrix, 'D': distortion_coeffs}.
        refine_subpix: If True, applies corner subpixel refinement.
        visualize: If True, draw detected markers, corners, and optional pose axes.
        draw_axes: If True (and visualize=True), draw pose axes.
        corner_radius: Pixel radius for drawn corner circles.
        save_path: If given and visualize=True, save visualization image here.

    Returns:
        tag_dict: {marker_id: {'rvec', 'tvec', 'corners'}}.
        img_vis: Visualization image (same as input if visualize=False).
    """
    if visualize and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{frame_idx}.png')
    
    K = fisheye_intr_dict['K']
    D = fisheye_intr_dict['D']

    param = cv2.aruco.DetectorParameters()
    if refine_subpix:
        param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    detector = cv2.aruco.ArucoDetector(aruco_dict, param)
    corners, ids, rejected = detector.detectMarkers(img)

    img_vis = img.copy() if visualize else img

    if corners is None or len(corners) == 0:
        if visualize and save_path is not None:
            img_vis_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_path, img_vis_rgb)
        print(f"No markers detected in frame {frame_idx}")
        return dict()

    if visualize:
        cv2.aruco.drawDetectedMarkers(img_vis, corners, ids)

    tag_dict = {}
    for this_id, this_corners in zip(ids, corners):
        this_id = int(this_id[0])

        if visualize:
            for (x, y) in this_corners.reshape(-1, 2):
                cv2.circle(img_vis, (int(x), int(y)), corner_radius, (0, 255, 255), -1)

        if this_id not in marker_size_map:
            continue

        marker_size_m = marker_size_map[this_id]
        # estimatePoseSingleMarkers was removed in OpenCV 4.8+; use solvePnP instead
        half = marker_size_m / 2.0
        obj_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float32)
        _, rvec, tvec = cv2.solvePnP(
            obj_points, this_corners.reshape(-1, 2), K, D
        )

        rvec = rvec.squeeze()
        tvec = tvec.squeeze()

        # store multiple detections of the same aruco tags, if they exist (such as acidentally included the streaming camera video on a nearby screen)
        if this_id in tag_dict: 
            # only overwrite the current rvec, tvec, corners if the z-value of the detected instance of the aruco code is smaller than the previous one
            # assumption: the closer aruco code is more accurate, as any streaming camera video will appear smaller on image, therefore the z-value will be smaller
            if tvec[-1] < tag_dict[this_id]['tvec'][-1]:
                tag_dict[this_id] = {
                    'rvec': rvec,
                    'tvec': tvec,
                    'corners': this_corners.squeeze()
                }
        else:
            # first detection of this aruco code    
            tag_dict[this_id] = {
                'rvec': rvec,
                'tvec': tvec,
                'corners': this_corners.squeeze()
            }

        if visualize and draw_axes:
            axis_len = float(marker_size_m) * 0.5
            cv2.aruco.drawAxis(img_vis, K, D, rvec, tvec, axis_len)

    # Save visualization if requested
    if visualize and save_path is not None:
        img_vis_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, img_vis_rgb)
 
        print(f"Saved visualization to {save_path}")

    return tag_dict

def get_charuco_board(
        aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100), 
        tag_id_offset=50,
        grid_size=(8, 5), square_length_mm=50, tag_length_mm=30):
    
    aruco_dict = cv2.aruco.Dictionary(
        aruco_dict.bytesList[tag_id_offset:], 
        aruco_dict.markerSize)
    board = cv2.aruco.CharucoBoard(
        size=grid_size,
        squareLength=square_length_mm/1000,
        markerLength=tag_length_mm/1000,
        dictionary=aruco_dict)
    return board

def draw_charuco_board(board, dpi=300, padding_mm=15):
    grid_size = np.array(board.getChessboardSize())
    square_length_mm = board.getSquareLength() * 1000

    mm_per_inch = 25.4
    board_size_pixel = (grid_size * square_length_mm + padding_mm * 2) / mm_per_inch * dpi
    board_size_pixel = board_size_pixel.round().astype(np.int64)
    padding_pixel = int(padding_mm / mm_per_inch * dpi)
    board_img = board.generateImage(outSize=board_size_pixel, marginSize=padding_pixel)
    return board_img

def get_gripper_width(tag_dict, left_id, right_id, nominal_z=0.072, z_tolerance=0.008):
    zmax = nominal_z + z_tolerance
    zmin = nominal_z - z_tolerance

    left_x = None
    if left_id in tag_dict:
        tvec = tag_dict[left_id]['tvec']
        # check if depth is reasonable (to filter outliers)
        if zmin < tvec[-1] < zmax:
            left_x = tvec[0]

    right_x = None
    if right_id in tag_dict:
        tvec = tag_dict[right_id]['tvec']
        if zmin < tvec[-1] < zmax:
            right_x = tvec[0]

    width = None
    if (left_x is not None) and (right_x is not None):
        width = right_x - left_x
    elif left_x is not None:
        width = abs(left_x) * 2
    elif right_x is not None:
        width = abs(right_x) * 2
    return width


# =========== image mask ====================
def canonical_to_pixel_coords(coords, img_shape=(2028, 2704)):
    pts = np.asarray(coords) * img_shape[0] + np.array(img_shape[::-1]) * 0.5
    return pts

def pixel_coords_to_canonical(pts, img_shape=(2028, 2704)):
    coords = (np.asarray(pts) - np.array(img_shape[::-1]) * 0.5) / img_shape[0]
    return coords

def draw_canonical_polygon(img: np.ndarray, coords: np.ndarray, color: tuple):
    pts = canonical_to_pixel_coords(coords, img.shape[:2])
    pts = np.round(pts).astype(np.int32)
    cv2.fillPoly(img, pts, color=color)
    return img

def get_mirror_canonical_polygon():
    left_pts = [
        [540, 1700],
        [680, 1450],
        [590, 1070],
        [290, 1130],
        [290, 1770],
        [550, 1770]
    ]
    resolution = [2028, 2704]
    left_coords = pixel_coords_to_canonical(left_pts, resolution)
    right_coords = left_coords.copy()
    right_coords[:,0] *= -1
    coords = np.stack([left_coords, right_coords])
    return coords


def get_mirror_crop_slices(img_shape=(1080,1920), left=True):
    left_pts = [
        [290, 1120],
        [650, 1480]
    ]
    resolution = [2028, 2704]
    left_coords = pixel_coords_to_canonical(left_pts, resolution)
    if not left:
        left_coords[:,0] *= -1
    left_pts = canonical_to_pixel_coords(left_coords, img_shape=img_shape)
    left_pts = np.round(left_pts).astype(np.int32)
    slices = (
        slice(np.min(left_pts[:,1]), np.max(left_pts[:,1])), 
        slice(np.min(left_pts[:,0]), np.max(left_pts[:,0]))
    )
    return slices


def get_gripper_canonical_polygon():
    left_pts = [
        [1352, 1730],
        [1100, 1700],
        [650, 1500],
        [0, 1350],
        [0, 2028],
        [1352, 2704]
    ]
    resolution = [2028, 2704]
    left_coords = pixel_coords_to_canonical(left_pts, resolution)
    right_coords = left_coords.copy()
    right_coords[:,0] *= -1
    coords = np.stack([left_coords, right_coords])
    return coords

def get_finger_canonical_polygon(height=0.37, top_width=0.25, bottom_width=1.4):
    # image size
    resolution = [2028, 2704]
    img_h, img_w = resolution

    # calculate coordinates
    top_y = 1. - height
    bottom_y = 1.
    width = img_w / img_h
    middle_x = width / 2.
    top_left_x = middle_x - top_width / 2.
    top_right_x = middle_x + top_width / 2.
    bottom_left_x = middle_x - bottom_width / 2.
    bottom_right_x = middle_x + bottom_width / 2.

    top_y *= img_h
    bottom_y *= img_h
    top_left_x *= img_h
    top_right_x *= img_h
    bottom_left_x *= img_h
    bottom_right_x *= img_h

    # create polygon points for opencv API
    points = [[
        [bottom_left_x, bottom_y],
        [top_left_x, top_y],
        [top_right_x, top_y],
        [bottom_right_x, bottom_y]
    ]]
    coords = pixel_coords_to_canonical(points, img_shape=resolution)
    return coords

def draw_predefined_mask(img, color=(0,0,0), mirror=True, gripper=True, finger=True, use_aa=False):
    all_coords = list()
    if mirror:
        all_coords.extend(get_mirror_canonical_polygon())
    if gripper:
        all_coords.extend(get_gripper_canonical_polygon())
    if finger:
        all_coords.extend(get_finger_canonical_polygon())
        
    for coords in all_coords:
        pts = canonical_to_pixel_coords(coords, img.shape[:2])
        pts = np.round(pts).astype(np.int32)
        flag = cv2.LINE_AA if use_aa else cv2.LINE_8
        cv2.fillPoly(img,[pts], color=color, lineType=flag)
    return img

def get_gripper_with_finger_mask(img, height=0.37, top_width=0.25, bottom_width=1.4, color=(0,0,0)):
    # image size
    img_h = img.shape[0]
    img_w = img.shape[1]

    # calculate coordinates
    top_y = 1. - height
    bottom_y = 1.
    width = img_w / img_h
    middle_x = width / 2.
    top_left_x = middle_x - top_width / 2.
    top_right_x = middle_x + top_width / 2.
    bottom_left_x = middle_x - bottom_width / 2.
    bottom_right_x = middle_x + bottom_width / 2.

    top_y *= img_h
    bottom_y *= img_h
    top_left_x *= img_h
    top_right_x *= img_h
    bottom_left_x *= img_h
    bottom_right_x *= img_h

    # create polygon points for opencv API
    points = np.array([[
        [bottom_left_x, bottom_y],
        [top_left_x, top_y],
        [top_right_x, top_y],
        [bottom_right_x, bottom_y]
    ]], dtype=np.int32)

    img = cv2.fillPoly(img, points, color=color, lineType=cv2.LINE_AA)
    return img

def inpaint_tag(img, corners, tag_scale=1.4, n_samples=16):
    # scale corners with respect to geometric center
    center = np.mean(corners, axis=0)
    scaled_corners = tag_scale * (corners - center) + center
    
    # sample pixels on the boundary to obtain median color
    sample_points = si.interp1d(
        [0,1,2,3,4], list(scaled_corners) + [scaled_corners[0]], 
        axis=0)(np.linspace(0,4,n_samples)).astype(np.int32)
    sample_colors = img[
        np.clip(sample_points[:,1], 0, img.shape[0]-1), 
        np.clip(sample_points[:,0], 0, img.shape[1]-1)
    ]
    median_color = np.median(sample_colors, axis=0).astype(img.dtype)
    
    # draw tag with median color
    img = cv2.fillPoly(
        img, scaled_corners[None,...].astype(np.int32), 
        color=median_color.tolist())
    return img

# =========== other utils ====================
def get_image_transform(in_res, out_res, crop_ratio:float = 1.0, bgr_to_rgb: bool=False):
    iw, ih = in_res
    ow, oh = out_res
    ch = round(ih * crop_ratio)
    cw = round(ih * crop_ratio / oh * ow)
    interp_method = cv2.INTER_AREA

    w_slice_start = (iw - cw) // 2
    w_slice = slice(w_slice_start, w_slice_start + cw)
    h_slice_start = (ih - ch) // 2
    h_slice = slice(h_slice_start, h_slice_start + ch)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray):
        assert img.shape == ((ih,iw,3))
        # crop
        img = img[h_slice, w_slice, c_slice]
        # resize
        img = cv2.resize(img, out_res, interpolation=interp_method)
        return img
    
    return transform
