# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
from tqdm import tqdm
import yaml
import json
import av
import numpy as np
import cv2
import pickle

from spvtp.common.cv_util_umi import (
    parse_aruco_config, 
    parse_fisheye_intrinsics_opencv,
    parse_fisheye_intrinsics_oak,
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    images_to_video
)

# %%
@click.command()
@click.option('-i', '--input', required=True)
@click.option('-o', '--output', required=True)
@click.option('-ij', '--intrinsics_json', required=True)
@click.option('-ay', '--aruco_yaml', required=True)
@click.option('-n', '--num_workers', type=int, default=4)
@click.option('-v', '--ar_detection_vis_dir', type=str, default='aruco_detection_vis')
@click.option('--visualize', required=True, default=False, help='Whether to visualize the aruco detection results')
def main(input, output, intrinsics_json, aruco_yaml, num_workers, ar_detection_vis_dir, visualize):
    cv2.setNumThreads(num_workers)
    demonstration_dir = output.split('/')[-2]
    zarr_session_dir = output.split('/')[-3].split('.')[0]

    # load aruco config
    aruco_config = parse_aruco_config(yaml.safe_load(open(aruco_yaml, 'r')))
    aruco_dict = aruco_config['aruco_dict']
    marker_size_map = aruco_config['marker_size_map']

    # load intrinsics
    intr_json = json.load(open(intrinsics_json, 'r'))
    if "intrinsic_type" in intr_json:
        print('Loading Oak intrinsics in opencv format...')
        raw_fisheye_intr = parse_fisheye_intrinsics_opencv(intr_json)
    elif "distortion" in intr_json:
        print('Loading Oak intrinsics in oak factory format...')
        raw_fisheye_intr = parse_fisheye_intrinsics_oak(intr_json)

    print("Raw fisheye intrinsics:", raw_fisheye_intr)
 
    results = list()
    with av.open(os.path.expanduser(input)) as in_container:
        in_stream = in_container.streams.video[0]
        in_stream.thread_type = "AUTO"
        in_stream.thread_count = num_workers

        in_res = np.array([in_stream.height, in_stream.width])[::-1]
        fisheye_intr = convert_fisheye_intrinsics_resolution(
            opencv_intr_dict=raw_fisheye_intr, target_resolution=in_res)

        for i, frame in tqdm(enumerate(in_container.decode(in_stream)), total=in_stream.frames):
            img = frame.to_ndarray(format='rgb24')
            frame_cts_sec = frame.pts * in_stream.time_base
            tag_dict = detect_localize_aruco_tags(
                img=img,
                aruco_dict=aruco_dict,
                marker_size_map=marker_size_map,
                fisheye_intr_dict=fisheye_intr,
                refine_subpix=True,
                frame_idx=i,
                visualize= visualize,
                save_dir = os.path.join(ar_detection_vis_dir,zarr_session_dir, demonstration_dir)
            )
            
            result = {
                'frame_idx': i,
                'time': float(frame_cts_sec),
                'tag_dict': tag_dict
            }
            results.append(result)
    
    # dump
    pickle.dump(results, open(os.path.expanduser(output), 'wb'))
    print(f"Saved results to {output}")

    # visualize
    episode = input.split('/')[-2]
    input_zarr_sess = input.split('/')[-3].split('.')[0]
   
    if visualize:
        images_to_video(img_dir=os.path.join(ar_detection_vis_dir, input_zarr_sess, episode),
                        out_path=os.path.join(ar_detection_vis_dir, f'{input_zarr_sess}_{episode}_aruco_detection.mp4'),
                        fps=60)

# %%
if __name__ == "__main__":
    main()
