"""Audit paired episode rollouts and summarize seeds, scenes, phases and drift."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from ..evaluation.runner import write_csv
from .phase_eval import PHASES


CONDITIONS = ('bbox_visual', 'no_prompt', 'shuffled_object', 'shuffled_target')
MODES = ('gt_history', 'history_feedback', 'cumulative')
SCALES = dict(ade_cm=('mean_pos_l2',100), episode_fde_cm=('mean_final_displacement_error',100),
    rotation_deg=('mean_rotation_geodesic_deg',1), grip_cm=('mean_grip_l1',100),
    endpoint_s10_percent=('success_at_10cm',100),
    endpoint_combined10_percent=('combined_success_10cm_20deg_grip',100))


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def condition_name(summary):
    intervention = summary['prompt_intervention']
    if intervention != 'none':
        return intervention
    return 'no_prompt' if summary['prompt_form'] == 'none' else 'bbox_visual'


def aggregate(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for identity, group in sorted(groups.items()):
        output = dict(zip(keys, identity), n_seeds=len({row['seed'] for row in group}))
        for metric in metrics:
            values = [float(row[metric]) for row in group if row.get(metric) not in (None, '')]
            output[metric+'_mean'] = float(np.mean(values)) if values else None
            output[metric+'_std'] = float(np.std(values,ddof=1)) if len(values)>1 else 0.0 if values else None
        result.append(output)
    return result


def audit_cases(cases):
    reference = None
    first = {}
    max_first_difference = 0.0
    for (seed, condition, mode), folder in sorted(cases.items()):
        chunks = read_csv(folder/'chunks.csv')
        identity = sorted((r['episode_key'].replace('\\','/'), int(r['step']),
                           int(r['valid_horizon']), int(r['sample_index'])) for r in chunks)
        if reference is None:
            reference = identity
        assert reference == identity, ('Unpaired query schedule/noise', seed, condition, mode)
        summary = json.loads((folder/'summary.json').read_text())
        if mode != 'cumulative':
            assert all(float(row['anchor_pos_l2']) == 0 for row in chunks)
        else:
            assert any(float(row['anchor_pos_l2']) > 0 for row in chunks if int(row['rollout_chunk']) > 0)
        endpoints = {row['episode_key']: row for row in chunks if int(row['valid_horizon']) < summary['action_horizon']}
        boundaries = json.loads((folder.parent/'phase_boundaries.json').read_text())
        episodes = read_csv(folder/'episodes.csv')
        for row in episodes:
            expected = boundaries[row['episode_key']]['n_steps']-1-summary['initial_recorded_prefix_end']
            assert int(row['predicted_steps']) == expected
            own_chunks = [r for r in chunks if r['episode_key'] == row['episode_key']]
            assert sum(int(r['valid_horizon']) for r in own_chunks) == expected
            last = max(own_chunks,key=lambda r:int(r['step']))
            assert int(last['step'])+int(last['valid_horizon']) == boundaries[row['episode_key']]['n_steps']-1
            if row['episode_key'] in endpoints:
                tail = endpoints[row['episode_key']]
                assert int(tail['step'])+int(tail['valid_horizon']) == boundaries[row['episode_key']]['n_steps']-1
        initial = {row['episode_key']: row for row in chunks if int(row['rollout_chunk']) == 0}
        key = (seed, condition)
        if key not in first:
            first[key] = initial
        else:
            for episode, row in initial.items():
                for metric in ('pos_l2','final_displacement_error','grip_l1'):
                    difference = abs(float(row[metric])-float(first[key][episode][metric]))
                    max_first_difference = max(max_first_difference, difference)
                    assert difference < 1e-6, ('Different warm starts', seed,condition,mode,episode,metric)
    return dict(paired_query_noise_identity_passed=True, tail_coverage_passed=True,
                no_gt_origin_reset_in_cumulative_passed=True,
                max_first_chunk_metric_difference=max_first_difference,
                evaluated_cases=len(cases), paired_chunks_per_case=len(reference or []))


def summarize(root, *, allow_partial=False):
    root = Path(root)
    seed_rows, phase_rows, curves, cases = [], [], [], {}
    for path in sorted(root.glob('seed_*/*/*/*/*/summary.json')):
        if not (path.parent.parent/'phase_boundaries.json').is_file():
            continue  # A mode is complete only after all scopes and boundaries are written.
        summary = json.loads(path.read_text())
        if summary.get('protocol') != 'replayed_rgb_state_history_rollout':
            continue
        seed, condition, mode, scope = (int(summary['training_seed']), condition_name(summary),
                                       summary['mode'], path.parent.name)
        identity = dict(seed=seed, condition=condition, mode=mode, scope=scope)
        full = json.loads((path.parent/'full_chunks_summary.json').read_text())
        seed_rows.append(dict(identity, episodes=summary['episodes'],
            prediction_steps=summary['prediction_steps'], rollout_chunks=summary['rollout_chunks'],
            partial_chunks=summary['partial_horizon_chunks'],
            full_chunk_fde_cm=100*full['mean_final_displacement_error'],
            **{name:summary[key]*scale for name,(key,scale) in SCALES.items()}))
        if scope == 'all':
            cases[(seed,condition,mode)] = path.parent
        for row in read_csv(path.parent/'phase_metrics.csv'):
            value = row['episode_balanced_full_horizon_fde']
            phase_rows.append(dict(identity, phase=row['phase'],
                ade_cm=100*float(row['episode_balanced_pos_l2']),
                grip_cm=100*float(row['episode_balanced_grip_l1']),
                full_horizon_fde_cm=100*float(value) if value else None,
                full_horizon_endpoints=int(row['full_horizon_endpoints'])))
        for row in read_csv(path.parent/'drift_curve.csv'):
            curves.append(dict(identity, progress_bin=int(row['progress_bin']),
                ade_cm=100*float(row['episode_balanced_pos_l2'])))
    expected = {(seed,condition,mode) for seed in (42,0,1) for condition in CONDITIONS for mode in MODES}
    missing = sorted(expected - cases.keys())
    if not allow_partial:
        assert not missing, ('Incomplete evaluations', missing)
        assert len(seed_rows) == 144, f'Expected 36 cases x 4 scopes, got {len(seed_rows)}'
    assert seed_rows, 'No rollout summaries found'
    audit = audit_cases(cases)
    values = aggregate(seed_rows, ('scope','condition','mode'), (*SCALES,'full_chunk_fde_cm'))
    phases = aggregate(phase_rows, ('scope','condition','mode','phase'),
                       ('ade_cm','grip_cm','full_horizon_fde_cm','full_horizon_endpoints'))
    drift = aggregate(curves, ('scope','condition','mode','progress_bin'), ('ade_cm',))
    output = root/'summary'
    write_csv(output/'per_seed.csv', seed_rows)
    write_csv(output/'aggregate.csv', values)
    write_csv(output/'phase_aggregate.csv', phases)
    write_csv(output/'drift_aggregate.csv', drift)
    result = dict(complete=not missing, missing_cases=missing, seeds=sorted({r['seed'] for r in seed_rows}),
        scopes=['all','scene1','scene2','scene3'], conditions=CONDITIONS, modes=MODES,
        audit=audit, aggregate=values, phase_aggregate=phases,
        caveat='Recorded RGB is replayed independently of predicted poses/actions; not closed-loop robot success.')
    (output/'summary.json').write_text(json.dumps(result, indent=2))
    def fmt(row, name):
        return f"{row[name+'_mean']:.2f} ± {row[name+'_std']:.2f}"
    lines = ['# Autoregressive state/history replay evaluation', '',
        f"Complete cases: {len(cases)}/36; seeds: {result['seeds']}.", '',
        'Primary metrics are episode-balanced, include valid partial tails, and use cm/degrees/percent.',
        'RGB remains a recorded replay; endpoint criteria are NOT physical task success.', '',
        '| Mode | Condition | ADE (cm) | Episode-end FDE (cm) | Rot (deg) | Grip (cm) | End S@10 (%) | End Combined@10 (%) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    indexed = {(r['mode'],r['condition']):r for r in values if r['scope']=='all'}
    for mode in MODES:
        for condition in CONDITIONS:
            row = indexed.get((mode,condition))
            if row:
                lines.append('| '+' | '.join([mode,condition,*[fmt(row,key) for key in SCALES]])+' |')
    lines += ['', '## Cumulative rollout by GT-defined execution phase', '',
        'Episode-balanced waypoint ADE (cm); phase labels are post-hoc only. Full-H chunk endpoint FDE is stored separately in phase_aggregate.csv.', '',
        '| Condition | Approach | Pre-grasp | Grasp/occlusion | Transport | Placement |',
        '|---|---:|---:|---:|---:|---:|']
    indexed_phase = {(r['condition'],r['phase']):r for r in phases if r['scope']=='all' and r['mode']=='cumulative'}
    for condition in CONDITIONS:
        cells = [fmt(indexed_phase[(condition,phase)],'ade_cm')
                 if (condition,phase) in indexed_phase else '—' for phase in PHASES]
        lines.append('| '+' | '.join([condition,*cells])+' |')
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(13,3.8))
    for axis, mode in zip(axes, MODES):
        for condition in CONDITIONS:
            points = sorted([r for r in drift if r['scope']=='all' and r['mode']==mode
                             and r['condition']==condition], key=lambda r:r['progress_bin'])
            if not points:
                continue
            x = np.array([(r['progress_bin']+.5)/10 for r in points])
            y = np.array([r['ade_cm_mean'] for r in points])
            sd = np.array([r['ade_cm_std'] for r in points])
            axis.plot(x,y,label=condition)
            axis.fill_between(x,y-sd,y+sd,alpha=.12)
        axis.set(title=mode, xlabel='Normalized post-initialization progress', ylabel='Position error (cm)')
        axis.grid(alpha=.2)
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output/'drift_curves.png',dpi=180)
    fig.savefig(output/'drift_curves.pdf')
    plt.close(fig)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(argv)
    result = summarize(args.root, allow_partial=args.allow_partial)
    print(json.dumps(dict(complete=result['complete'], audit=result['audit']),indent=2))


if __name__ == '__main__':
    main()
