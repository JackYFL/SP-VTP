# Code reuse and tests

The root `pyproject.toml` owns packaging/dependencies. `spvtp.cli` dispatches
lazily, so camera SDKs and VLA packages are not required for standard SPOT runs.
The public package and module entry point are `spvtp` / `python -m spvtp`.
Refresh an existing installation with `uv pip install -e .` after renaming.
Historical `runs/` records and external data paths are preserved verbatim;
their old path strings are not active package imports.
Flat module aliases remain within `spvtp`; no legacy root-package shim is
provided. State-dict checkpoints still load; override `--data-root` or
`data.root` if a saved data path points at a relocated directory.

| Module | Responsibility / reusable interface |
| --- | --- |
| `demonstration_collection/session.py` | `RecordingSession`, shared capture lifecycle |
| `demonstration_processing/pipeline.py` | `process(config, backend=..., overrides=...)` |
| `common/episode_arrays.py` | `align_valid_frames(values, valid)` |
| `vision_traj/data/` | episode indexing, splits, prompts, frames/cache |
| `vision_traj/models/` | task/observation encoders and action heads |
| `vision_traj/runtime.py` | `load_config`, `select_episodes`, `make_dataset`, `make_loader`, `load_checkpoint` |
| `vision_traj/engine/trainer.py` | Accelerate training and validation orchestration |
| `vision_traj/evaluation/` | shared `compute_batch_metrics` and `evaluate_dataset` |
| `vision_traj/analysis/` | phases, pick/place boundaries, seed summaries |
| `vision_traj/baselines/` | external VLA adapters and their shared evaluator |
| `vision_traj/scripts/` | thin command entry points |

Import from these canonical modules in new code. Flat old paths remain aliases
for compatibility, not alternate implementations. `VisionTrajDataset.action_arrays(i)`
returns recorded history/targets without video decoding. New policies should
reuse the same splits, sample noise and metric functions instead of duplicating
the evaluation loop. Never add a process-global `sys.path` rewrite on import.

## Checks

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s spvtp/vision_traj/tests -q
OMP_NUM_THREADS=2 python -m unittest discover -s tests -q
```

Tests cover shapes/prompts, fixed splits, known metric values, noise invariance,
cache/direct input equality, recording lifecycle, alignment/processing, and
train/checkpoint/eval on generated videos. They need no pretrained download.
See [GPU validation](gpu_validation.md) for real-data checks; CPU/synthetic tests
do not replace camera hardware or full-training validation.

Namespace migration passed 50 CPU tests, offline wheel/console checks and a
three-scene, 128-window GPU evaluation of a pre-rename checkpoint (maximum
metric difference 0). See the [migration evidence](../../../runs/namespace_migration_20261104/summary.json).
