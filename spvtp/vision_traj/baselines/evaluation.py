"""Shared distributed metric collection for external VLA adapters."""
from collections import defaultdict
import json
from pathlib import Path

import torch

from ..evaluation.metrics import compute_batch_metrics, display_metrics, summarize_records
from ..evaluation.runner import write_csv


@torch.no_grad()
def evaluate_vla(accelerator, policy, loader, output_dir, step, max_batches, predict):
    was_training = policy.training
    policy.eval()
    records = {}
    devices = [accelerator.device.index or 0] if accelerator.device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(0)
            for bi, batch in enumerate(loader):
                if max_batches is not None and bi >= max_batches:
                    break
                target = batch['action'].to(accelerator.device).float()
                prediction = predict(batch)[..., :target.shape[1], :10].float()
                metrics = compute_batch_metrics(prediction, target)
                gathered = accelerator.gather_for_metrics(dict(
                    sample_index=batch['index'].to(accelerator.device), **metrics))
                gathered = {key: value.cpu().tolist() for key, value in gathered.items()}
                for i, index in enumerate(gathered.pop('sample_index')):
                    records.setdefault(index, dict(sample_index=index, episode_key=str(index),
                                                  **{k: v[i] for k, v in gathered.items()}))
        rows = [records[i] for i in sorted(records)]
        base = getattr(getattr(loader, 'dataset', None), 'base', None)
        groups = defaultdict(list)
        if base is not None:
            for row in rows:
                ei, frame = base._samples[row['sample_index']]
                episode = base.episodes[ei]
                row.update(episode_key=episode.key, step=frame,
                           raw_frame=int(base._valid_raw_idx[ei][frame]))
                groups[episode.scene].append(row)
        summary = dict(summarize_records(rows), step=int(step))
        if base is None:
            summary.pop('episodes')
        if accelerator.is_main_process:
            folder = Path(output_dir) / 'eval_metrics' / f'step_{step:06d}'
            write_csv(folder / 'metrics.csv', rows)
            (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
            for scene, scene_rows in groups.items():
                write_csv(folder / scene / 'metrics.csv', scene_rows)
                (folder / scene / 'summary.json').write_text(json.dumps(summarize_records(scene_rows), indent=2))
            print(json.dumps(display_metrics(summary)), flush=True)
        return summary
    finally:
        policy.train(was_training)
