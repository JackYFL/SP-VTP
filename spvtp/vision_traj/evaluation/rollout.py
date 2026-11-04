"""Sequential state/history feedback with REPLAYED RGB, not closed-loop control.

Every waypoint is relative to the chunk origin. Predicted world poses are
retained across queries; history is re-expressed at each new origin. Warm-start
states are recorded once. All later pose/width feedback is mode-specific.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..analysis.phase_eval import segment_episode_phases
from ..runtime import make_initial_noise, to_device
from ..utils.rotation_utils import compose_se3, rotmat_to_rot6d, se3_relative
from .metrics import compute_batch_metrics, summarize_records, waypoint_metrics
from .runner import write_csv


ROLLOUT_MODES = ('gt_history', 'history_feedback', 'cumulative')
INPUT_KEYS = ('first_frame', 'current_frame', 'prompt_object', 'prompt_target', 'action_history')
STEP_METRICS = ('pos_l2', 'rotation_geodesic_deg', 'grip_l1', 'grip_state_correct')


def noise_index(episode_key, step):
    token = f'{episode_key.replace(chr(92), "/")}:{step}'.encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], 'little') & ((1 << 60) - 1)


def pose_actions(poses, grippers):
    return torch.cat((poses[..., :3, 3], rotmat_to_rot6d(poses[..., :3, :3]),
                      grippers.unsqueeze(-1)), dim=-1).float()


def predicted_history(poses, grippers, step, horizon, anchor):
    if not horizon:
        return torch.empty(0, 10)
    relative = se3_relative(anchor, poses[step-horizon:step])
    return pose_actions(relative, grippers[step-horizon:step])


def group_step_metrics(records, group_key, metrics):
    """Return pooled and episode-balanced means for arbitrary timeline groups."""
    groups = defaultdict(list)
    for row in records:
        groups[row[group_key]].append(row)
    output = []
    for group, rows in sorted(groups.items()):
        episodes = defaultdict(list)
        for row in rows:
            episodes[row['episode_key']].append(row)
        result = {group_key: group, 'predictions': len(rows), 'episodes': len(episodes)}
        for metric in metrics:
            result[f'mean_{metric}'] = float(np.mean([r[metric] for r in rows]))
            result[f'episode_balanced_{metric}'] = float(np.mean([
                np.mean([r[metric] for r in items]) for items in episodes.values()]))
        output.append(result)
    return output


def write_rollout_outputs(output, dataset, episode_rows, chunks, steps, boundaries, metadata):
    """Save native-unit summaries plus phase/progress rows from the same rollout."""
    summaries = {}
    scopes = ['all', *sorted({ep.scene for ep in dataset.episodes})]
    for scope in scopes:
        select = lambda rows: [r for r in rows if scope == 'all' or r['scene'] == scope]
        ep_rows, ch_rows, st_rows = map(select, (episode_rows, chunks, steps))
        full = [row for row in ch_rows if row['valid_horizon'] == dataset.action_horizon]
        summary = dict(summarize_records(ep_rows), **metadata,
            aggregation='episode_balanced', prediction_steps=len(st_rows),
            rollout_chunks=len(ch_rows), full_horizon_chunks=len(full),
            partial_horizon_chunks=len(ch_rows)-len(full))
        summaries[scope] = summary
        if output is None:
            continue
        folder = Path(output) / scope
        write_csv(folder / 'episodes.csv', ep_rows)
        write_csv(folder / 'chunks.csv', ch_rows)
        write_csv(folder / 'step_metrics.csv', st_rows)
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
        if full:
            (folder / 'full_chunks_summary.json').write_text(json.dumps(
                dict(summarize_records(full), aggregation='full_horizon_chunks_only', **metadata), indent=2))
        phases = group_step_metrics(st_rows, 'phase', STEP_METRICS)
        endpoints = {r['phase']: r for r in group_step_metrics(full, 'phase',
            ('final_displacement_error', 'combined_success_10cm_20deg_grip'))}
        for phase in phases:
            endpoint = endpoints.get(phase['phase'], {})
            phase['full_horizon_endpoints'] = endpoint.get('predictions', 0)
            phase['episode_balanced_full_horizon_fde'] = endpoint.get(
                'episode_balanced_final_displacement_error')
        write_csv(folder / 'phase_metrics.csv', phases)
        write_csv(folder / 'drift_curve.csv', group_step_metrics(st_rows, 'progress_bin', STEP_METRICS))
    if output is not None:
        (Path(output) / 'phase_boundaries.json').write_text(json.dumps(boundaries, indent=2))
    return summaries


@torch.inference_mode()
def evaluate_rollout(policy, dataset, output_dir=None, *, mode='cumulative', batch_size=32,
                     sample_seed=0, gripper_state_threshold=0.03, progress_bins=10,
                     metadata=None):
    """Batch independent episodes; each episode is strictly autoregressive in time.

    gt_history: recorded origin/history at every chunk.
    history_feedback: predicted history, recorded origin at every chunk.
    cumulative: predicted origin/history, without post-initialization GT resets.
    All modes receive recorded RGB at paired query steps, INCLUDING the tail.
    """
    if mode not in ROLLOUT_MODES:
        raise ValueError(f'Unknown rollout mode: {mode}')
    if not dataset.episodes or batch_size < 1 or progress_bins < 1:
        raise ValueError('Nonempty episodes and positive batch/progress sizes are required')
    device = next(policy.parameters()).device
    start, horizon, history = dataset.sample_start_step, dataset.action_horizon, dataset.history_horizon
    states, boundary_rows = [], {}
    for ei, ep in enumerate(dataset.episodes):
        raw = dataset._valid_raw_idx[ei]
        poses = torch.as_tensor(dataset._poses[ei][raw], dtype=torch.float64)
        grips = torch.as_tensor(dataset._grippers[ei][raw], dtype=torch.float64).reshape(-1)
        if not torch.isfinite(poses).all() or not torch.isfinite(grips).all():
            raise ValueError(f'Non-finite recorded states: {ep.key}')
        predicted = torch.full_like(poses, float('nan'))
        predicted_grips = torch.full_like(grips, float('nan'))
        predicted[:start+1], predicted_grips[:start+1] = poses[:start+1], grips[:start+1]
        boundary = segment_episode_phases(grips.numpy(), poses[:, :3, 3].numpy())
        boundary_rows[ep.key] = asdict(boundary)
        states.append(dict(poses=poses, grips=grips, predicted=predicted,
                           predicted_grips=predicted_grips, raw=raw, boundary=boundary))
    was_training = policy.training
    policy.eval()
    chunks, steps = [], []
    try:
        last = max(len(state['raw']) for state in states) - 1
        for current in range(start, last, horizon):
            active = [ei for ei, state in enumerate(states) if current < len(state['raw'])-1]
            for offset in range(0, len(active), batch_size):
                indices = active[offset:offset+batch_size]
                samples, anchors, ids = [], [], []
                for ei in indices:
                    state = states[ei]
                    sample = dataset.sample_at(ei, current, allow_partial=True)
                    anchor = (state['predicted'][current] if mode == 'cumulative'
                              else state['poses'][current])
                    # The first query has exactly the same recorded warm start in all modes.
                    if mode != 'gt_history' and current != start:
                        sample['action_history'] = predicted_history(state['predicted'],
                            state['predicted_grips'], current, history, anchor)
                    samples.append({key: sample[key] for key in INPUT_KEYS})
                    anchors.append(anchor)
                    ids.append(noise_index(dataset.episodes[ei].key, current))
                batch = to_device({key: torch.stack([s[key] for s in samples])
                                   for key in INPUT_KEYS}, device)
                noise = make_initial_noise(ids, horizon, 10, sample_seed)
                predictions = policy.predict_action(batch, initial_noise=noise).float().cpu()
                if predictions.shape != (len(indices), horizon, 10) or not torch.isfinite(predictions).all():
                    raise ValueError('Invalid predictions; no failed episode is silently discarded')
                for bi, ei in enumerate(indices):
                    state, ep, anchor = states[ei], dataset.episodes[ei], anchors[bi]
                    length = min(horizon, len(state['raw'])-1-current)
                    future = slice(current+1, current+1+length)
                    predicted = predictions[bi, :length]
                    # Each waypoint is current-relative, NOT a sequence of incremental transforms.
                    world = anchor @ compose_se3(predicted[:, :3], predicted[:, 3:9]).double()
                    state['predicted'][future] = world
                    state['predicted_grips'][future] = predicted[:, 9].double()
                    pred_actions = pose_actions(world, predicted[:, 9].double()).unsqueeze(0)
                    gt_actions = pose_actions(state['poses'][future], state['grips'][future]).unsqueeze(0)
                    metrics = {k: float(v[0]) for k,v in compute_batch_metrics(pred_actions, gt_actions,
                        gripper_state_threshold=gripper_state_threshold).items()}
                    row = dict(episode_key=ep.key, scene=ep.scene, task=ep.task,
                        sample_index=ids[bi], step=current, raw_frame=int(state['raw'][current]),
                        rollout_chunk=(current-start)//horizon, valid_horizon=length,
                        phase=state['boundary'].phase_at(current+length),
                        anchor_pos_l2=float((anchor[:3,3]-state['poses'][current,:3,3]).norm()), **metrics)
                    chunks.append(row)
                    values = {k:v[0].tolist() for k,v in waypoint_metrics(pred_actions, gt_actions,
                        gripper_state_threshold).items()}
                    for h in range(length):
                        step = current+h+1
                        steps.append(dict(episode_key=ep.key, scene=ep.scene,
                            sample_index=ids[bi], current_step=current, horizon_index=h,
                            future_step=step, future_raw_frame=int(state['raw'][step]),
                            phase=state['boundary'].phase_at(step),
                            progress_bin=min(progress_bins-1, int((step-start)/(len(state['raw'])-1-start)*progress_bins)),
                            is_chunk_endpoint=int(h == length-1), is_episode_endpoint=int(step == len(state['raw'])-1),
                            pred_x=float(world[h,0,3]), pred_y=float(world[h,1,3]), pred_z=float(world[h,2,3]),
                            gt_x=float(state['poses'][step,0,3]), gt_y=float(state['poses'][step,1,3]), gt_z=float(state['poses'][step,2,3]),
                            pred_grip_width=float(predicted[h,9]), gt_grip_width=float(state['grips'][step]),
                            **{key:value[h] for key,value in values.items()}))
        episode_rows = []
        for ei, (ep, state) in enumerate(zip(dataset.episodes, states)):
            pred = pose_actions(state['predicted'][start+1:], state['predicted_grips'][start+1:])
            gt = pose_actions(state['poses'][start+1:], state['grips'][start+1:])
            if not torch.isfinite(pred).all():
                raise ValueError(f'Incomplete rollout coverage: {ep.key}')
            metrics = {k:float(v[0]) for k,v in compute_batch_metrics(pred.unsqueeze(0), gt.unsqueeze(0),
                gripper_state_threshold=gripper_state_threshold).items()}
            episode_rows.append(dict(episode_key=ep.key, scene=ep.scene, task=ep.task,
                                     predicted_steps=len(pred), **metrics))
        info = dict(protocol='replayed_rgb_state_history_rollout', mode=mode,
            action_horizon=horizon, history_horizon=history, initial_recorded_prefix_end=start,
            prompt_form=dataset.prompt_form,
            current_rgb='recorded_replay', predicted_actions_change_rgb=False,
            pose_anchor='predicted' if mode == 'cumulative' else 'recorded_each_query',
            history_source='recorded' if mode == 'gt_history' else 'predicted_after_warm_start',
            query_stride=horizon, tail_policy='score_valid_prefix', sample_seed=sample_seed,
            noise_key='canonical_episode_key_and_current_step', **(metadata or {}))
        summaries = write_rollout_outputs(output_dir, dataset, episode_rows, chunks, steps, boundary_rows, info)
        return summaries, dict(episodes=episode_rows, chunks=chunks, steps=steps)
    finally:
        policy.train(was_training)
