"""Trajectory metrics in native units: meters, degrees, and fractions."""
import numpy as np
import torch

from ..utils.rotation_utils import rot6d_to_rotmat


def rotation_geodesic_degrees(pred_rot6d, gt_rot6d):
    relative = rot6d_to_rotmat(pred_rot6d).transpose(-1, -2) @ rot6d_to_rotmat(gt_rot6d)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cosine))


def waypoint_metrics(pred, gt, gripper_state_threshold=0.03):
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 10:
        raise ValueError(f'Expected matching [B,H,10] tensors, got {pred.shape} and {gt.shape}')
    return dict(pos_l2=(pred[..., :3] - gt[..., :3]).norm(dim=-1),
        rot6d_l2=(pred[..., 3:9] - gt[..., 3:9]).norm(dim=-1),
        rotation_geodesic_deg=rotation_geodesic_degrees(pred[..., 3:9], gt[..., 3:9]),
        grip_l1=(pred[..., 9] - gt[..., 9]).abs(),
        grip_state_correct=((pred[..., 9] > gripper_state_threshold) ==
                            (gt[..., 9] > gripper_state_threshold)).float())


def compute_batch_metrics(pred, gt, *, gripper_state_threshold=0.03):
    steps = waypoint_metrics(pred, gt, gripper_state_threshold)
    values = {key: value.mean(-1) for key, value in steps.items() if key != 'grip_state_correct'}
    values.update(final_displacement_error=steps['pos_l2'][:, -1],
                  final_rotation_geodesic_deg=steps['rotation_geodesic_deg'][:, -1],
                  grip_state_accuracy=steps['grip_state_correct'].mean(-1),
                  final_grip_state_correct=steps['grip_state_correct'][:, -1])
    for cm in (2, 5, 10):
        success = values['final_displacement_error'] <= cm / 100
        values[f'success_at_{cm}cm'] = success.float()
        if cm != 2:
            values[f'combined_success_{cm}cm_20deg_grip'] = (
                success & (values['final_rotation_geodesic_deg'] <= 20)
                & values['final_grip_state_correct'].bool()).float()
    for degrees in (10, 20):
        values[f'rotation_success_{degrees}deg'] = (values['final_rotation_geodesic_deg'] <= degrees).float()
    return values


def summarize_records(records):
    if not records:
        raise ValueError('No usable evaluation samples')
    keys = ('pos_l2', 'final_displacement_error', 'rot6d_l2', 'rotation_geodesic_deg',
            'final_rotation_geodesic_deg', 'grip_l1', 'grip_state_accuracy')
    summary = {f'mean_{key}': float(np.mean([row[key] for row in records])) for key in keys}
    summary['final_grip_state_accuracy'] = float(np.mean([row['final_grip_state_correct'] for row in records]))
    for key in records[0]:
        if key.startswith(('success_at_', 'rotation_success_', 'combined_success_')):
            summary[key] = float(np.mean([row[key] for row in records]))
    return dict(summary, samples=len(records), episodes=len({r['episode_key'] for r in records}),
                units=dict(position='m', gripper='m', rotation_geodesic='deg', success='fraction'))


def display_metrics(summary):
    return dict(pos_cm=100 * summary['mean_pos_l2'],
                fde_cm=100 * summary['mean_final_displacement_error'],
                rotation_deg=summary['mean_rotation_geodesic_deg'],
                grip_cm=100 * summary['mean_grip_l1'],
                success_10cm_percent=100 * summary['success_at_10cm'])
