"""Full offline evaluation using the same data and metrics as training."""
import argparse
import json
from pathlib import Path

import torch

from ..data.prompt_interventions import PROMPT_INTERVENTIONS, apply_prompt_intervention
from ..evaluation.metrics import compute_batch_metrics, display_metrics, rotation_geodesic_degrees
from ..evaluation.runner import evaluate_dataset, render_samples
from ..runtime import load_checkpoint, make_dataset, make_initial_noise, select_episodes


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate offline trajectory predictions')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--annotations')
    parser.add_argument('--frame-cache-root')
    parser.add_argument('--output-dir', '--viz-dir', dest='output_dir')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--split', choices=['train', 'val', 'all', 'tail'], default='val')
    parser.add_argument('--split-manifest')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--val-ratio', type=float)
    parser.add_argument('--scenes')
    parser.add_argument('--n-episodes', type=int, default=0, help='0 evaluates the complete split')
    parser.add_argument('--episode-indices')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--gripper-state-threshold', type=float, default=0.03)
    parser.add_argument('--prompt-intervention', choices=PROMPT_INTERVENTIONS, default='none')
    parser.add_argument('--intervention-seed', type=int, default=0)
    parser.add_argument('--save-step-metrics', action='store_true')
    parser.add_argument('--viz-samples', type=int, default=0)
    parser.add_argument('--viz-sort', choices=['total', 'pos', 'rot', 'grip'], default='total')
    parser.add_argument('--viz-selection', choices=['ranked', 'episode_first'], default='ranked')
    parser.add_argument('--viz-format', choices=['png', 'pdf'], default='png')
    args = parser.parse_args(argv)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    policy, cfg, _ = load_checkpoint(args.ckpt, device)
    policy.eval()
    for key, value in dict(root=args.data_root, annotations_json=args.annotations,
                          frame_cache_root=args.frame_cache_root, split_seed=args.seed,
                          val_ratio=args.val_ratio).items():
        if value is not None:
            cfg.data[key] = value
    if args.scenes is not None:
        cfg.data.scenes = [s.strip() for s in args.scenes.split(',') if s.strip()]
    default_manifest = Path(args.ckpt).parent / 'split_manifest.json'
    manifest = args.split_manifest or (default_manifest if default_manifest.exists()
        and args.seed is None and args.val_ratio is None and args.scenes is None else None)
    episodes = select_episodes(cfg, 'all' if args.split == 'tail' else args.split, manifest)
    episodes, intervention = apply_prompt_intervention(episodes, args.prompt_intervention,
                                                       seed=args.intervention_seed)
    if args.episode_indices:
        episodes = [episodes[int(i)] for i in args.episode_indices.split(',')]
    elif args.n_episodes > 0:
        episodes = episodes[-args.n_episodes:] if args.split == 'tail' else episodes[:args.n_episodes]
    prompt = {'missing_object': 'target_only', 'missing_target': 'object_only'}.get(
        args.prompt_intervention, cfg.data.prompt_form)
    dataset = make_dataset(cfg, episodes, prompt_form=prompt,
        dynamic_prompt_render=bool(cfg.data.get('frame_cache_root')) and args.prompt_intervention
                              not in ('none', 'missing_object', 'missing_target'))
    output = Path(args.output_dir) if args.output_dir else Path(args.ckpt).parent / 'eval_metrics'
    summaries, records = evaluate_dataset(policy, dataset, output,
        batch_size=args.batch_size, num_workers=args.num_workers, sample_seed=args.sample_seed,
        gripper_state_threshold=args.gripper_state_threshold, save_steps=args.save_step_metrics,
        metadata=dict(split=args.split, split_seed=int(cfg.data.get('split_seed', cfg.train.seed)),
            val_ratio=float(cfg.data.val_ratio), prompt_intervention=args.prompt_intervention))
    if args.prompt_intervention != 'none':
        (output / 'prompt_intervention.json').write_text(json.dumps(intervention, indent=2))
    render_samples(policy, dataset, records, output / 'all', count=args.viz_samples,
        sample_seed=args.sample_seed, sort=args.viz_sort, selection=args.viz_selection, format=args.viz_format)
    print(json.dumps({scene: display_metrics(summary) for scene, summary in summaries.items()}, indent=2))
    return summaries


if __name__ == '__main__':
    main()
