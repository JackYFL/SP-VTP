"""Lazy project entry point: hardware and VLA extras load only when selected."""
import argparse
from importlib import import_module
import sys

COMMANDS = {
    'collect': ('spvtp.demonstration_collection.session', True, 'collect iPhone + GoPro/OAK episodes'),
    'process': ('spvtp.demonstration_processing.pipeline', True, 'synchronize and calibrate demonstrations'),
    'annotate': ('spvtp.annotation_tool.video_bbox_annotator', False, 'annotate first-frame object/target boxes'),
    'merge-annotations': ('spvtp.annotation_tool.extract_annotations', False, 'merge per-video annotation files'),
    'train': ('spvtp.vision_traj.scripts.train', True, 'train the SPOT policy'),
    'eval': ('spvtp.vision_traj.scripts.infer', True, 'evaluate all chunks and scenes'),
    'rollout': ('spvtp.vision_traj.scripts.rollout', True, 'evaluate predicted history with replayed RGB'),
    'rollout-suite': ('spvtp.vision_traj.scripts.rollout_suite', True, 'run paired multi-checkpoint rollouts'),
    'visualize': ('spvtp.vision_traj.scripts.vis_stitched_traj', False, 'render stitched episode trajectories'),
    'cache': ('spvtp.vision_traj.data.build_frame_cache', False, 'predecode video frames'),
    'stats': ('spvtp.vision_traj.data.dataset_stats', False, 'summarize dataset statistics'),
    'language-map': ('spvtp.vision_traj.data.build_language_prompt_map', False, 'build task language mappings'),
    'phases': ('spvtp.vision_traj.analysis.phase_eval', False, 'evaluate five execution phases'),
    'boundaries': ('spvtp.vision_traj.analysis.ee_pick_place_boundaries', False, 'detect pick/place contact boundaries'),
    'pick-place': ('spvtp.vision_traj.analysis.pick_place_analysis', False, 'aggregate pick/place errors'),
    'multiseed': ('spvtp.vision_traj.analysis.summarize_multiseed', False, 'aggregate seeds and bootstrap intervals'),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='spvtp', description='Collect, process, train and evaluate SP-VTP',
        epilog='Use spvtp COMMAND --help for command-specific options.')
    parser.add_argument('command', choices=[*COMMANDS, 'vla'], nargs='?')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    if not argv or argv[0] in ('--help', '-h'):
        parser.print_help()
        for name, (_, _, description) in COMMANDS.items():
            print(f'  {name:19s} {description}')
        return
    args = parser.parse_args(argv)
    if args.command == 'vla':
        choices = ['groot', 'pi05', 'smolvla', 'openvla_oft']
        if not args.arguments or args.arguments[0] in ('--help', '-h'):
            print('spvtp vla {' + ','.join(choices) + '} [options]')
            return
        model, *arguments = args.arguments
        if model not in choices:
            parser.error(f'Unknown VLA {model}; choose {choices}')
        module, native = f'spvtp.vision_traj.baselines.{model}_baseline', False
    else:
        module, native, _ = COMMANDS[args.command]
        arguments = args.arguments
    implementation = import_module(module)
    if native:
        implementation.main(arguments)
        return 0
    original = sys.argv
    try:
        sys.argv = [module, *arguments]
        implementation.main()
        return 0
    finally:
        sys.argv = original


if __name__ == '__main__':
    main()
