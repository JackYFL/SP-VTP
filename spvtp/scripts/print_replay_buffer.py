import argparse
from spvtp.common.replay_buffer_util import print_replay_buffer_libero, print_replay_buffer_umi, print_replay_buffer_draw, print_replay_buffer_push_t

parser = argparse.ArgumentParser(description='View the contents of a replay buffer.')
parser.add_argument('--dataset_path', type=str, help='Path to the replay buffer to view.')
parser.add_argument('--vis_frame', action='store_true', help='Save images of the RGB data.')
parser.add_argument('--vis_video', action='store_true', help='Save video of the RGB data.')
parser.add_argument('--task_name', type=str, help='The name of the task.', choices=['libero', 'umi', 'draw', 'pusht'], required=True)
args = parser.parse_args()

if args.task_name == 'libero':
    print_replay_buffer_libero(args.dataset_path, vis_frame=args.vis_frame, vis_video=args.vis_video)
elif args.task_name == 'umi':
    print_replay_buffer_umi(args.dataset_path, vis_frame=args.vis_frame, vis_video=args.vis_video)
elif args.task_name == 'draw':
    print_replay_buffer_draw(args.dataset_path, vis_video=args.vis_video)
elif args.task_name == 'pusht':
    print_replay_buffer_push_t(args.dataset_path, vis_video=args.vis_video)
else:
    raise ValueError(f'Invalid task name: {args.task_name}')
