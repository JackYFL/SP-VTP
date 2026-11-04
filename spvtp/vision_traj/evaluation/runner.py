"""One inference pass produces overall and per-scene metrics and CSV records."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import torch

from ..runtime import make_initial_noise, make_loader, to_device
from .metrics import compute_batch_metrics, summarize_records, waypoint_metrics


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        if records:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


@torch.no_grad()
def evaluate_dataset(policy, dataset, output_dir=None, *, batch_size=8, num_workers=0,
                     sample_seed=0, gripper_state_threshold=0.03, save_steps=False,
                     max_batches=None, predictor=None, metadata=None):
    if not len(dataset):
        raise ValueError('Evaluation dataset is empty')
    device = next(policy.parameters()).device
    was_training = policy.training
    policy.eval()
    records, groups = [], defaultdict(list)
    output = Path(output_dir) if output_dir else None
    loader = make_loader(dataset, batch_size, num_workers, indexed=True)
    try:
        with ExitStack() as stack:
            step_writer = None
            if save_steps and output:
                (output / 'all').mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context((output / 'all' / 'step_metrics.csv').open('w', newline=''))
            for batch_number, batch in enumerate(loader):
                if max_batches is not None and batch_number >= max_batches:
                    break
                indices = batch.pop('sample_index').tolist()
                batch = to_device(batch, device)
                noise = make_initial_noise(indices, dataset.action_horizon, 10, sample_seed)
                pred = (predictor(batch, noise) if predictor else
                        policy.predict_action(batch, initial_noise=noise)).float()
                gt = batch['future_actions'].float()
                metrics = {k: v.cpu().tolist() for k, v in compute_batch_metrics(
                    pred, gt, gripper_state_threshold=gripper_state_threshold).items()}
                step_values = {k: v.cpu().tolist() for k, v in waypoint_metrics(
                    pred, gt, gripper_state_threshold).items()} if save_steps else None
                for bi, index in enumerate(indices):
                    ei, step = dataset._samples[index]
                    ep = dataset.episodes[ei]
                    raw = dataset._valid_raw_idx[ei]
                    row = dict(sample_index=index, episode_index=ei, episode_key=ep.key,
                               step=step, raw_frame=int(raw[step]), **{k: v[bi] for k, v in metrics.items()})
                    row['total_error'] = row['pos_l2'] + row['rot6d_l2'] + row['grip_l1']
                    records.append(row)
                    groups[ep.scene].append(row)
                    if step_values and output:
                        for h in range(dataset.action_horizon):
                            item = dict(sample_index=index, episode_index=ei, episode_key=ep.key,
                                current_step=step, current_raw_frame=int(raw[step]), horizon_index=h,
                                future_step=step + h + 1, future_raw_frame=int(raw[step+h+1]),
                                **{k: v[bi][h] for k, v in step_values.items()},
                                pred_grip_width=float(pred[bi,h,9]), gt_grip_width=float(gt[bi,h,9]),
                                is_chunk_endpoint=int(h == dataset.action_horizon - 1))
                            if step_writer is None:
                                step_writer = csv.DictWriter(handle, fieldnames=list(item))
                                step_writer.writeheader()
                            step_writer.writerow(item)
        groups['all'] = records
        summaries = {}
        for scene, rows in groups.items():
            summary = dict(summarize_records(rows), sample_seed=sample_seed,
                           gripper_state_threshold=gripper_state_threshold, **(metadata or {}))
            summaries[scene] = summary
            if output:
                folder = output / scene
                write_csv(folder / 'metrics.csv', rows)
                (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
        return summaries, records
    finally:
        policy.train(was_training)


@torch.no_grad()
def render_samples(policy, dataset, records, output, *, count=12, sample_seed=0,
                   sort='total', selection='ranked', format='png'):
    from ..utils.viz import save_trajectory_comparison
    if count == 0:
        return
    key = {'total': 'total_error', 'pos': 'pos_l2', 'rot': 'rot6d_l2', 'grip': 'grip_l1'}[sort]
    selected = sorted(records, key=lambda row: row[key], reverse=True)
    if selection == 'episode_first':
        selected = list({row['episode_key']: row for row in reversed(records)}.values())
    if count > 0:
        selected = selected[:count]
    device = next(policy.parameters()).device
    rows = []
    for row in selected:
        index = row['sample_index']
        sample = dataset[index]
        batch = to_device({k: v.unsqueeze(0) for k, v in sample.items()}, device)
        noise = make_initial_noise([index], dataset.action_horizon, 10, sample_seed)
        pred = policy.predict_action(batch, initial_noise=noise)[0].cpu()
        filename = f'sample_{index:06d}.{format}'
        save_trajectory_comparison(Path(output) / filename,
            first_frame=sample['first_frame'], current_frame=sample['current_frame'],
            prompt_object=sample['prompt_object'], prompt_target=sample['prompt_target'],
            prompt_form=dataset.prompt_form, pred_actions=pred, gt_actions=sample['future_actions'],
            title=f"FDE={100*row['final_displacement_error']:.2f} cm", subtitle=row['episode_key'])
        rows.append(dict(figure=filename, **row))
    write_csv(Path(output) / 'selected_samples.csv', rows)
