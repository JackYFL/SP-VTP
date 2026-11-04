import json, os
import pathlib
import click
import zarr
import pickle
import numpy as np
import cv2
import av
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
import sys
from scipy.spatial.transform import Rotation as R
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
UMI_ROOT = REPO_ROOT / 'demonstration_processing'
for _p in (str(REPO_ROOT), str(UMI_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from spvtp.demonstration_processing.umi.common.cv_util import (
    get_image_transform, 
    draw_predefined_mask,
    inpaint_tag,
)
from pathlib import Path

from spvtp.diffusion_policy.common.replay_buffer import ReplayBuffer
from spvtp.diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()


def true_segments_np(valid_indices, closed=False):
    vi = np.asarray(valid_indices, dtype=bool)
    if vi.size == 0 or not vi.any():
        return []
    d = np.diff(np.r_[False, vi, False].astype(int))
    starts = np.where(d == 1)[0]
    ends   = np.where(d == -1)[0]
    return list(zip(starts, ends - 1)) if closed else list(zip(starts, ends))


@click.command()
@click.argument('input', nargs=-1)
@click.option('-i', '--original_input', required=True, help='Input original zarr path')
@click.option('-o', '--output', required=True, help='Zarr path')
@click.option('-or', '--out_res', type=str, default='224,224')
@click.option('-cl', '--compression_level', type=int, default=99)
@click.option('-n', '--num_workers', type=int, default=None)

def main(input, original_input, output, out_res, compression_level, num_workers):
    if os.path.isfile(output):
        if click.confirm(f'Output file {output} exists! Overwrite?', abort=True):
            pass
    

    out_res = tuple(int(x) for x in out_res.split(','))
    
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()  # Limit to max 4 workers by default
    print(f"Using {num_workers} workers for video processing")
    cv2.setNumThreads(1)
            
 
    # Use temporary directory for intermediate storage to avoid OOM
    import tempfile
    temp_dir = tempfile.mkdtemp()
    temp_store = zarr.DirectoryStore(temp_dir)
    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=temp_store)
    
    # Use output_zarr data (gripper_widths, poses, etc.)
    input_intermediate_zarr_path = pathlib.Path(input[0])
    input_intermediate_zarr = zarr.open(input_intermediate_zarr_path, mode='r')
    # Dump lowdim data to replay buffer
    # Generate argument for videos
    n_cameras = 1
    buffer_start = 0
    all_videos = set()
    vid_args = list()

    orignal_zarr_input = pathlib.Path(original_input)

    for ep_name, ep_grp in input_intermediate_zarr.groups():
        # Skip non-episode groups (e.g. nested output directories)
        if not ep_name.startswith('episode_'):
            continue
        # Skip gripper calibration episodes (episode_0)
        if 'episode_0' in ep_name:
            print(f"Skipping gripper calibration episode: {ep_name}")
            continue
            
        cameras = []
        # import ipdb; ipdb.set_trace()
        gripper_width = ep_grp['gripper_widths'][:] # valid gripper widths
        poses = ep_grp['pose_interp'][:] # valid poses
        valid_indices = ep_grp['valid_indices'][:] # list of bools
        segments = true_segments_np(valid_indices)
 

        # find the first index where the valid_indices is True
        for segment in segments:
            if segment[0] >= 0:
                segment_start, segment_end = segment[0], segment[1]
                segment_length = segment_end - segment_start 
                cameras.append({
                    "video_path": str(input_intermediate_zarr_path.joinpath(ep_name, 'camera_1.mp4')),
                    "video_start_end": (segment_start, segment_end)
                })
                
        videos_dict = defaultdict(list)
        episode_data = dict()
        gripper_id = 0
        
        # poses shape: (N, 4, 4) - 4x4 transformation matrices
        # Extract position from translation vector (last column, first 3 elements)
        eef_pos = poses[:, :3, 3]  # Shape: (N, 3)
        
        # Extract rotation matrix and convert to axis-angle representation
        rotation_matrices = poses[:, :3, :3]  # Shape: (N, 3, 3)
        # Convert rotation matrices to axis-angle
        eef_rot = np.array([R.from_matrix(rot_mat).as_rotvec() for rot_mat in rotation_matrices])  # Shape: (N, 3)
        
        # Create demo poses in 6D format [pos(3) + axis_angle(3)]
        demo_start_pose_6d = np.concatenate([eef_pos[0], eef_rot[0]])  # Shape: (6,)
        demo_end_pose_6d = np.concatenate([eef_pos[-1], eef_rot[-1]])  # Shape: (6,)
        
        # Repeat for all timesteps to match expected format
        demo_start_pose = np.tile(demo_start_pose_6d, (len(eef_pos), 1))  # Shape: (N, 6)
        demo_end_pose = np.tile(demo_end_pose_6d, (len(eef_pos), 1))  # Shape: (N, 6)
        
        robot_name = f'robot{gripper_id}'
        episode_data[robot_name + '_eef_pos'] = eef_pos.astype(np.float32)
        episode_data[robot_name + '_eef_rot_axis_angle'] = eef_rot.astype(np.float32)
        episode_data[robot_name + '_gripper_width'] = np.expand_dims(gripper_width, axis=-1).astype(np.float32)
        episode_data[robot_name + '_demo_start_pose'] = demo_start_pose
        episode_data[robot_name + '_demo_end_pose'] = demo_end_pose
        
        out_replay_buffer.add_episode(data=episode_data, compressors=None)
        # aggregate video gen aguments
        n_frames = None
        for cam_id, camera in enumerate(cameras):
            video_path = Path(camera['video_path'])
            assert video_path.is_file()
            
            video_start, video_end = camera['video_start_end']
            if n_frames is None:
                n_frames = video_end - video_start
            else:
                assert n_frames == (video_end - video_start)
            
            videos_dict[str(video_path)].append({
                'camera_idx': cam_id,
                'frame_start': video_start,
                'frame_end': video_end,
                'buffer_start': buffer_start
            })
 
        buffer_start += n_frames
        vid_args.extend(videos_dict.items())
        all_videos.update(videos_dict.keys())
    
    print(f"{len(all_videos)} videos used in total!")
    
    # Get image size
    with av.open(vid_args[0][0]) as container:
        in_stream = container.streams.video[0]
        ih, iw = in_stream.height, in_stream.width
    
    # Dump images
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    for cam_id in range(n_cameras):
        name = f'camera{cam_id}_rgb'
        _ = out_replay_buffer.data.require_dataset(
            name=name,
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )

    def video_to_zarr(replay_buffer, mp4_path, tasks, out_res):
        pkl_path = os.path.join(os.path.dirname(mp4_path), 'tag_detection.pkl')
        tag_detection_results = pickle.load(open(pkl_path, 'rb'))

        tasks = sorted(tasks, key=lambda x: x['frame_start'])
        camera_idx = None
        for task in tasks:
            if camera_idx is None:
                camera_idx = task['camera_idx']
            else:
                assert camera_idx == task['camera_idx']
        name = f'camera{camera_idx}_rgb'
        img_array = replay_buffer.data[name]
        
        curr_task_idx = 0
        
 
        with av.open(mp4_path) as container:
            in_stream = container.streams.video[0]
            in_stream.thread_count = 1
            buffer_idx = 0
            for frame_idx, frame in tqdm(enumerate(container.decode(in_stream)), total=in_stream.frames, leave=False):
                if curr_task_idx >= len(tasks):
                    # all tasks done
                    break
                
                if frame_idx < tasks[curr_task_idx]['frame_start']:
                    # current task not started
                    continue
                elif frame_idx < tasks[curr_task_idx]['frame_end']:
                    if frame_idx == tasks[curr_task_idx]['frame_start']:
                        buffer_idx = tasks[curr_task_idx]['buffer_start']
                    
                    # Do current task
                    img = frame.to_ndarray(format='rgb24')

                    # Inpaint tags
                    # this_det = tag_detection_results[frame_idx]
                    # all_corners = [x['corners'] for x in this_det['tag_dict'].values()]
                    # for corners in all_corners:
                    #     img = inpaint_tag(img, corners)
                        
                    # # Mask out gripper
                    # # TODO @huy-ha: define the new gripper mask based on flying-umi's CAD + camera mounting
                    # img = draw_predefined_mask(img, color=(0,0,0), 
                    #     mirror=False, gripper=True, finger=False)
                    
                    # Center-square crop then resize to match output resolution
                    ih, iw, _ = img.shape
                    if ih > iw:
                        top = (ih - iw) // 2
                        left = 0
                        crop_size = iw
                    else:
                        top = 0
                        left = (iw - ih) // 2
                        crop_size = ih
                    img = img[top:top+crop_size, left:left+crop_size]
                    target_height, target_width = out_res
                    img = cv2.resize(img, (target_width, target_height))
                        
                    # Compress image
                    img_array[buffer_idx] = img
                    buffer_idx += 1
                    
                    if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
                        # current task done, advance
                        curr_task_idx += 1
                else:
                    assert False
                    
    with tqdm(total=len(vid_args)) as pbar:
        # Process videos with controlled concurrency to avoid overwhelming the system
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mp4_path, tasks in vid_args:
                # Wait for a worker to become available before submitting new tasks
                while len(futures) >= num_workers:
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(video_to_zarr, 
                    out_replay_buffer, mp4_path, tasks, out_res))

            # Wait for remaining tasks to complete
            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print([x.result() for x in completed])

    # Dump to disk
    print(f"Saving ReplayBuffer to {output}")
    with zarr.ZipStore(output, mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir)
    print(f"Done! {len(all_videos)} videos used in total!")

if __name__ == '__main__':
    main()
