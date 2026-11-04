"""Evaluate chronological history/pose feedback while explicitly replaying RGB."""
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..data.prompt_interventions import apply_prompt_intervention
from ..evaluation.metrics import display_metrics
from ..evaluation.rollout import ROLLOUT_MODES, evaluate_rollout
from ..runtime import load_checkpoint, make_dataset, select_episodes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--annotations')
    parser.add_argument('--frame-cache-root')
    parser.add_argument('--split-manifest')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--modes', default=','.join(ROLLOUT_MODES))
    parser.add_argument('--interventions', default='none')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--intervention-seed', type=int, default=0)
    parser.add_argument('--n-episodes', type=int, default=0)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args(argv)
    modes = args.modes.split(',')
    if any(mode not in ROLLOUT_MODES for mode in modes):
        parser.error(f'--modes must be comma-separated values from {ROLLOUT_MODES}')
    if args.split_manifest and not Path(args.split_manifest).is_file():
        parser.error(f'Split manifest not found: {args.split_manifest}')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    policy, cfg, _ = load_checkpoint(args.ckpt, device)
    for key, value in dict(root=args.data_root, annotations_json=args.annotations,
                          frame_cache_root=args.frame_cache_root).items():
        if value is not None:
            cfg.data[key] = value
    manifest = args.split_manifest or Path(args.ckpt).parent / 'split_manifest.json'
    manifest = manifest if Path(manifest).exists() else None
    episodes = select_episodes(cfg, 'val', manifest)
    interventions = args.interventions.split(',')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, Path(args.output_dir) / 'config.yaml')
    results = {}
    for intervention in interventions:
        changed, donors = apply_prompt_intervention(episodes, intervention, seed=args.intervention_seed)
        if args.n_episodes:
            changed = changed[:args.n_episodes]
        prompt = {'missing_object': 'target_only', 'missing_target': 'object_only'}.get(
            intervention, cfg.data.prompt_form)
        dataset = make_dataset(cfg, changed, prompt_form=prompt,
                               dynamic_prompt_render=intervention != 'none')
        condition = Path(args.output_dir) / intervention
        condition.mkdir(parents=True, exist_ok=True)
        (condition / 'prompt_intervention.json').write_text(json.dumps(donors, indent=2))
        for mode in modes:
            summary, _ = evaluate_rollout(policy, dataset, condition / mode,
                mode=mode, batch_size=args.batch_size, sample_seed=args.sample_seed,
                metadata=dict(checkpoint=str(args.ckpt), prompt_intervention=intervention,
                              training_seed=int(cfg.train.seed)))
            results[f'{intervention}/{mode}'] = summary
            print(json.dumps(dict(condition=intervention, mode=mode,
                                  **display_metrics(summary['all']))), flush=True)
    return results


if __name__ == '__main__':
    main()
