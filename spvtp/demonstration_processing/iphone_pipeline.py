import os, collections, json, pickle, zarr, pathlib, multiprocessing, subprocess, shutil
import sys
from pathlib import Path
from typing import Callable, Iterator, List, Optional
import numpy as np


from omegaconf import DictConfig
from spvtp.demonstration_processing.utils.data_util import interpolate_episode
from spvtp.demonstration_processing.utils.gripper_util import get_gripper_width_offset
from spvtp.common.episode_arrays import align_valid_frames
from spvtp.demonstration_processing.utils.color_util import blue
from glob import glob
from omegaconf import DictConfig
from tqdm import tqdm
import concurrent.futures
from typing import List
from spvtp.common.flying_umi_constants import FLYINGUMI_OAK_OFFSET_FROM_FINGER_AR_TAG_Z  
from spvtp.demonstration_processing.umi.common.interpolation_util import (
    get_gripper_calibration_interpolator, 
    get_interp1d
)
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)

def calibrate_gripper_range(demonstration_iterator, cfg: DictConfig, output_dir:str):
    skipped_demonstrations = set()
    processed_demonstrations = set()
    
    # calibrate gripper range for gripper calibration demonstration, assuming its epsode_0
    for demonstration_dir in demonstration_iterator(None):
        
        # assuming the first episode is for gripper calibration of each collected session of demos
        # TODO @huy-ha: incorporate special flag to avoid the assumption that the gripper_calibration episode to always be episode_0
        if 'episode_0' in demonstration_dir:
            demo_basename = os.path.basename(demonstration_dir)
            tag_path = Path(output_dir).joinpath(demo_basename, 'tag_detection.pkl').absolute()

            input_path = str(tag_path)
            output_path = str(Path(output_dir).joinpath(
                demo_basename
            ).joinpath(f'gripper_range.json').absolute())
 
            tag_det_threshold = cfg.tag_det_threshold

            tag_detection_results = pickle.load(open(input_path, 'rb'))
            
            # identify gripper hardware id
            n_frames = len(tag_detection_results)
            tag_counts = collections.defaultdict(lambda: 0)
            for frame in tag_detection_results:
                for key in frame['tag_dict'].keys():
                    tag_counts[key] += 1
            tag_stats = collections.defaultdict(lambda: 0.0)
            for k, v in tag_counts.items():
                tag_stats[k] = v / n_frames

            max_tag_id = np.max(list(tag_stats.keys()))
            tag_per_gripper = 6
            max_gripper_id = max_tag_id // tag_per_gripper

            gripper_prob_map = dict()
            for gripper_id in range(max_gripper_id+1):
                left_id = gripper_id * tag_per_gripper
                right_id = left_id + 1
                left_prob = tag_stats[left_id]
                right_prob = tag_stats[right_id]
                gripper_prob = min(left_prob, right_prob)
                if gripper_prob <= 0:
                    continue
                gripper_prob_map[gripper_id] = gripper_prob
            
            if len(gripper_prob_map) == 0:
                print("No grippers detected!")
                assert False

            gripper_probs = sorted(gripper_prob_map.items(), key=lambda x:x[1])
            gripper_id = gripper_probs[-1][0]
            gripper_prob = gripper_probs[-1][1]

            print(f"Detected gripper id: {gripper_id} with probability {gripper_prob}")
            if gripper_prob < tag_det_threshold:
                print(f"Detection rate {gripper_prob} < {tag_det_threshold} threshold.")
                assert False
                
            # run calibration
            left_id = gripper_id * tag_per_gripper
            right_id = left_id + 1

            gripper_widths = list()
            for i, dt in enumerate(tag_detection_results):
                tag_dict = dt['tag_dict']
                width = get_gripper_width_offset(tag_dict, left_id, right_id, nominal_z=FLYINGUMI_OAK_OFFSET_FROM_FINGER_AR_TAG_Z)
                print('left_id:', left_id, 'right_id:', right_id, 'width:', width)
                if width is None:
                    width = float('Nan')
                gripper_widths.append(width)
            gripper_widths = np.array(gripper_widths)
            max_width = np.nanmax(gripper_widths)
            min_width = np.nanmin(gripper_widths)

            result = {
                'gripper_id': gripper_id,
                'left_finger_tag_id': left_id,
                'right_finger_tag_id': right_id,
                'max_width': max_width,
                'min_width': min_width
            }
            print(f'Saving gripper calibration to {output_path}')
            json.dump(result, open(output_path, 'w'), indent=2)

        # generate calibrated, interpolated gripper widths
        gripper_widths, _ = get_demo_gripper_width_oak(demonstration_dir, output_dir)

        # save the gripper widths to the output zarr
        demo_basename = os.path.basename(demonstration_dir)
        to_root = zarr.open(os.path.join(output_dir, demo_basename),mode="a")

        # load the valid indices from the interpolated episode
        valid_indices = to_root["valid_indices"][:] 
        to_root["gripper_widths"] = align_valid_frames(gripper_widths[valid_indices], valid_indices)

        processed_demonstrations.add(demonstration_dir)
        if demonstration_dir in skipped_demonstrations:
            skipped_demonstrations.remove(demonstration_dir)

    print(f"Processed {len(processed_demonstrations)} demonstrations, skipped {len(skipped_demonstrations)} demonstrations.")


def get_demo_gripper_width_oak(demonstration_dir, output_dir):
    """Returns a list of gripper widths for each frame in the demonstration. Adapted from 06_generate_dataset_plan.py from UMI.
    Note that since the iphone records ultrawide at 10Hz (and ultrawide is used to detect AR tags), but since we want to have gripper width for every frame of the main camera video (which is at 60Hz)

    Returns:
    - gripper_widths: list of gripper widths for each frame in the demonstration
    - gripper_detection_types: list of integers indicating the presence of the gripper in each frame. 0: not present, 1: both fingers present, 2: left finger only, 3: right finger only
    """

    # find the associated gripper calibration
    gripper_cal_run_dir = Path(output_dir).joinpath('episode_0')
 
    # get the gripper calibration interpolator
    with open(Path(gripper_cal_run_dir).joinpath(f'gripper_range.json'), 'r') as f:
        gripper_range_data = json.load(f)
    max_width = gripper_range_data['max_width']
    min_width = gripper_range_data['min_width']
    gripper_cal_data = {
        'aruco_measured_width': [min_width, max_width],
        'aruco_actual_width': [min_width, max_width]
    }
    gripper_cal_interp = get_gripper_calibration_interpolator(**gripper_cal_data)

    # load the tag detection results from the output directory
    demo_basename = os.path.basename(demonstration_dir)
    pkl_path = Path(output_dir).joinpath(demo_basename, 'tag_detection.pkl')
    if not pkl_path.is_file():
        print(f"Ignored {pkl_path}, no tag_detection.pkl")

    with open(pkl_path, 'rb') as f:
        tag_detection_results = pickle.load(f)
    video_timestamps = np.array([x['time'] for x in tag_detection_results])

    # identify the gripper id
    left_id = gripper_range_data['left_finger_tag_id']
    right_id = gripper_range_data['right_finger_tag_id']

    # extract the gripper width from the tag detection results
    detected_gripper_timestamps = list()
    detected_gripper_widths = list()
 
    for td in tag_detection_results:
        width = get_gripper_width_offset(td['tag_dict'], left_id, right_id, nominal_z=FLYINGUMI_OAK_OFFSET_FROM_FINGER_AR_TAG_Z)
        if width is not None:
            detected_gripper_timestamps.append(td['time'])
            detected_gripper_widths.append(gripper_cal_interp(width))
    

    # from spvtp.demonstration_processing.process_stages.visualize_flyingumi import plot_gripper_width
    # plot_gripper_width(
    #     detected_gripper_widths,
    #     detected_gripper_timestamps,
    #     os.path.join(output_dir, f'{os.path.basename(demonstration_dir)}_gripper_width_plot_intermediate.png')
    # )

    # some frames may not have had detections
    if len(detected_gripper_timestamps) == 0:
        # return NaNs for all frames (no detections available)
        this_gripper_widths = np.full(video_timestamps.shape, np.nan, dtype=float)
        return this_gripper_widths, video_timestamps

    # interpolate detected widths to fill gaps
    gripper_interp = get_interp1d(detected_gripper_timestamps, detected_gripper_widths)

    # use the main RGB to get the timestamps. This is because the ultrawide records the tag detections at 10Hz, but we want to have gripper width for every frame of the main camera video (which is at 60Hz)
    this_gripper_widths = gripper_interp(video_timestamps)
    return this_gripper_widths, video_timestamps

 
def iterate_demonstrations(demonstrations_dir:str, demo_type:Optional[str]=None):
    if demo_type is not None:
        assert demo_type in {'demonstration', 'grippercalibration'}
    for demonstration_dir in sorted(list(glob(demonstrations_dir + '/*'))):
        if not os.path.isdir(demonstration_dir):
            continue
        base_name = os.path.basename(demonstration_dir)
        if not (base_name.startswith('episode_') or base_name.startswith('gripper_calibration')):
            continue
            
        if demo_type is not None:
            base_dir_name = os.path.basename(demonstration_dir)
            if demo_type == 'grippercalibration':
                if not base_dir_name.startswith('gripper_calibration'):
                    continue
            elif demo_type == 'demonstration':
                if not os.path.isdir(os.path.join(demonstration_dir, 'episode')):
                    continue

        yield demonstration_dir

def detect_ar_tag_oak(demonstration_iterator, cfg, output_dir):
    from .pipeline import detect_tags
    return detect_tags(demonstration_iterator, cfg, output_dir)
                        

def sync_cameras(demonstration_iterator:Callable[[Optional[List[str]]], Iterator[str]], cfg: DictConfig, output_dir:str):
    for demonstration_dir in demonstration_iterator(None):
        demo_basename = os.path.basename(demonstration_dir)
        to_dir = os.path.join(output_dir, demo_basename)
        interpolate_episode(
            from_path=demonstration_dir,
            to_path=to_dir,
            camera_latency=cfg.camera_latency,
            tracking_latency=cfg.tracking_latency,
            T_ee_cam=np.array(cfg.T_ee_cam, dtype=np.float64),
        )
        # Copy camera_1.mp4 to the output directory
        src_video = os.path.join(demonstration_dir, 'camera_1.mp4')
        dst_video = os.path.join(to_dir, 'camera_1.mp4')
        if os.path.isfile(src_video) and not os.path.isfile(dst_video):
            shutil.copy2(src_video, dst_video)

def visualize_processed_data(demonstration_iterator:Callable[[Optional[List[str]]], Iterator[str]], cfg: DictConfig, output_dir:str):
    from spvtp.demonstration_processing.process_stages.visualize_flyingumi import visualize_flying_umi_processed_data_episode
    for demonstration_dir in demonstration_iterator(None):
        # visualize the processed data
        visualize_flying_umi_processed_data_episode(demonstration_dir=demonstration_dir, cfg =cfg, output_dir = output_dir)

def generate_replay_buffer_flyingumi(cfg: DictConfig):
    """Generate a replay buffer from the processed data."""   

    script_path = pathlib.Path(__file__).parent.joinpath('scripts', 'generate_replay_buffer_flyingumi.py')
    assert script_path.is_file()
    cmd = [
        'python', str(script_path),
        '-o', os.path.join(cfg.output_policy_replay_buffer_zarr),
        '--original_input', str(cfg.input_dir),
        str(cfg.output_dir) # this is the output directory where the processed data is stored
    ]
    print("cmd:", cmd)
    result = subprocess.run(cmd)
    assert result.returncode == 0

def main(cfg: DictConfig):

    # Determine which stages to run
    all_stages = ['sync_cameras','detect', 'calibrate', 'visualize', 'gen_replay_buffer']

    if cfg.stages is None:
        cfg.stages = all_stages

    stages = [stage for stage in cfg.stages if stage not in cfg.skip_stages]
    assert all([stage in all_stages for stage in stages])

    # Define demonstration iterator
    get_demonstration_iterator = lambda demo_type=None: iterate_demonstrations(cfg.input_dir, demo_type)

    # Ensure the processed output root is a Zarr group (creates .zgroup at root if missing)
    zarr.open(cfg.output_dir, mode="a")

    if 'sync_cameras' in stages:
        print(blue("\n--- SYNC CAMERAS STAGE ---"))
        sync_cameras(get_demonstration_iterator, cfg=cfg.sync_cameras,output_dir=cfg.output_dir)

    if 'detect' in stages:
        print(blue("\n--- DETECT AR TAG STAGE ---"))
        detect_ar_tag_oak(get_demonstration_iterator, cfg=cfg.detect,output_dir=cfg.output_dir)

    if 'calibrate' in stages:
        print(blue("\n--- CALIBRATE GRIPPER STAGE ---"))
        calibrate_gripper_range(get_demonstration_iterator, cfg=cfg.calibrate, output_dir=cfg.output_dir)

    if 'visualize' in stages:
        print(blue("\n--- VISUALIZE STAGE ---"))
        visualize_processed_data(get_demonstration_iterator, cfg = cfg.visualize, output_dir=cfg.output_dir) # add show in gui flag (real time render, not save to file), need to output file path to zarr for policy inference
    
    if 'gen_replay_buffer' in stages:
        print(blue("\n--- GENERATE REPLAY BUFFER STAGE ---"))
        generate_replay_buffer_flyingumi(cfg=cfg)
        
if __name__ == '__main__':
    main()
