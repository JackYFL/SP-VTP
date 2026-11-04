"""Utility to print out the contents of a replay buffer for debugging."""

from spvtp.common import import_umi_source
import zarr
from spvtp.common.replay_buffer import ReplayBuffer
from spvtp.diffusion_policy.common.replay_buffer import ReplayBuffer as DPReplayBuffer
from spvtp.diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
import cv2
import os
import numpy as np
from tqdm import tqdm

def print_replay_buffer_libero(path, vis_frame: bool = False, vis_video: bool = False):
    def fix_rgb_image(rgb_image):
        rgb_image = np.rot90(rgb_image, k=2, axes=(0, 1)).copy()
        rgb_image = np.flip(rgb_image, axis=1).copy()
        return rgb_image

    register_codecs(verbose=False)
    with zarr.ZipStore(path, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, 
            store=zarr.MemoryStore()
        )

    print(replay_buffer)
    print(f'Task names: {replay_buffer.task_names[:]}')

    out_dir = os.path.dirname(path)
    dataset_name = os.path.basename(path).split('.')[0]
    if vis_frame:
        # Get a random sample index
        sample_index = np.random.randint(0, replay_buffer['agentview_rgb'].shape[0])
        
        # Get and fix both camera views
        agentview_rgb = fix_rgb_image(replay_buffer['agentview_rgb'][sample_index][...,::-1])
        eye_in_hand_rgb = fix_rgb_image(replay_buffer['eye_in_hand_rgb'][sample_index][...,::-1])
        
        # Hstack the images
        combined_image = np.hstack((agentview_rgb, eye_in_hand_rgb))
        
        # Save the combined image
        out_path = os.path.join(out_dir, f'tmp_{dataset_name}_combined_camera_views.png')
        cv2.imwrite(out_path, combined_image)
        print(f'Saved combined camera views to {out_path}')

    if vis_video:
        # video of combined camera views
        video_path = os.path.join(out_dir, f'tmp_{dataset_name}_combined_camera_views.mp4')
        frame_width = replay_buffer['agentview_rgb'].shape[2] * 2  # Double width for hstacked frames
        frame_height = replay_buffer['agentview_rgb'].shape[1]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('a','v','c','1'), 20, (frame_width, frame_height))

        for i in tqdm(range(replay_buffer['agentview_rgb'].shape[0]), desc='Writing combined camera views video'):
            # Get and fix both camera views
            agentview_frame = fix_rgb_image(replay_buffer['agentview_rgb'][i][...,::-1])
            eye_in_hand_frame = fix_rgb_image(replay_buffer['eye_in_hand_rgb'][i][...,::-1])
            
            # Combine frames horizontally
            combined_frame = np.hstack((agentview_frame, eye_in_hand_frame))
            out.write(combined_frame)

        out.release()
        print(f'Saved combined camera views video to {video_path}')

def print_replay_buffer_umi(path, vis_frame: bool = False, vis_video: bool = False):
    register_codecs(verbose=False)
    with zarr.ZipStore(path, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, 
            store=zarr.MemoryStore()
        )

    print(replay_buffer)
    print(f'Task names: {replay_buffer.task_names[:]}')

    out_dir = os.path.dirname(path)
    if vis_frame:
        # single frame of main camera
        sample_index = np.random.randint(0, replay_buffer['camera0_main_rgb'].shape[0])
        main_rgb = replay_buffer['camera0_main_rgb'][sample_index][...,::-1]
        out_path = os.path.join(out_dir, 'tmp_main_rgb.png')
        cv2.imwrite(out_path, main_rgb)
        print(f'Saved main RGB image to {out_path}')

        # single frame of ultrawide camera
        ultrawide_sample_index = replay_buffer.map_upsample_index('camera0_ultrawide_rgb', sample_index)
        ultrawide_rgb = replay_buffer['camera0_ultrawide_rgb'][ultrawide_sample_index][...,::-1]
        out_path = os.path.join(out_dir, 'tmp_ultrawide_rgb.png')
        cv2.imwrite(out_path, ultrawide_rgb)
        print(f'Saved ultrawide RGB image to {out_path}')

    if vis_video:
        # video of main camera
        video_path = os.path.join(out_dir, 'tmp_main_video.mp4')
        frame_width = replay_buffer['camera0_main_rgb'].shape[2]
        frame_height = replay_buffer['camera0_main_rgb'].shape[1]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('a','v','c','1'), 60, (frame_width, frame_height))

        for i in tqdm(range(replay_buffer['camera0_main_rgb'].shape[0]), desc='Writing main video'):
            frame = replay_buffer['camera0_main_rgb'][i][...,::-1]
            out.write(frame)

        out.release()
        print(f'Saved main video to {video_path}')

        # video of ultrawide camera
        video_path = os.path.join(out_dir, 'tmp_ultrawide_video.mp4')
        frame_width = replay_buffer['camera0_ultrawide_rgb'].shape[2]
        frame_height = replay_buffer['camera0_ultrawide_rgb'].shape[1]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('a','v','c','1'), 10, (frame_width, frame_height))

        for i in tqdm(range(replay_buffer['camera0_ultrawide_rgb'].shape[0]), desc='Writing ultrawide video'):
            frame = replay_buffer['camera0_ultrawide_rgb'][i][...,::-1]
            out.write(frame)

        out.release()
        print(f'Saved ultrawide video to {video_path}')

def print_replay_buffer_draw(path, vis_video: bool = False):
    replay_buffer = ReplayBuffer.copy_from_path(path, store=zarr.MemoryStore())
    print(replay_buffer)
    print(f'Episode names: {replay_buffer.episode_names[:]}')
    print(f'Task names: {replay_buffer.task_names[:]}')

    if vis_video:
        rgb_data = replay_buffer['image']
        # Get output directory
        out_dir = os.path.dirname(path)

        # Create video writer for rgb data with 224x224 resolution
        video_name = os.path.basename(path).split('.')[0]
        video_path = os.path.join(out_dir, f'tmp_{video_name}_draw_video.mp4')
        frame_width = 224
        frame_height = 224
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('a','v','c','1'), 10, (frame_width, frame_height))

        # Write each frame to video
        for i in tqdm(range(rgb_data.shape[0]), desc='Writing draw video'):
            frame = rgb_data[i][...,::-1] # Convert BGR to RGB
            # Resize frame to 224x224
            frame = cv2.resize(frame, (frame_width, frame_height))
            out.write(frame)

        out.release()
        print(f'Saved draw video to {video_path}')

def print_replay_buffer_push_t(path, vis_video: bool = False):
    replay_buffer = DPReplayBuffer.copy_from_path(
            path, keys=['img', 'state', 'action'])
    print(replay_buffer)

    if vis_video:
        rgb_data = replay_buffer['img']
        # Get output directory
        out_dir = os.path.dirname(path)

        # Create video writer for rgb data with 224x224 resolution
        video_name = os.path.basename(path).split('.')[0]
        video_path = os.path.join(out_dir, f'tmp_{video_name}_pusht_video.mp4')
        frame_width = 224
        frame_height = 224
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('a','v','c','1'), 10, (frame_width, frame_height))

        # Write each frame to video
        for i in tqdm(range(rgb_data.shape[0]), desc='Writing pusht video'):
            frame = rgb_data[i][...,::-1] # Convert BGR to RGB
            # Resize frame to 224x224
            frame = cv2.resize(frame, (frame_width, frame_height))
            frame = frame.astype(np.uint8)
            out.write(frame)

        out.release()
        print(f'Saved pusht video to {video_path}')
