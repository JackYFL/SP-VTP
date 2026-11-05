# Code reuse and tests

The root `pyproject.toml` owns packaging/dependencies. `spvtp.cli` dispatches
lazily, so camera SDKs and VLA packages are not required for standard SPOT runs.
The public package and module entry point are `spvtp` / `python -m spvtp`.
Refresh an existing installation with `uv pip install -e .` after renaming.
Historical `runs/` records and external data paths are preserved verbatim;
their old path strings are not active package imports.
The 28 flat compatibility modules in `vision_traj/` have been removed. Only
`__init__.py` (public exports) and `runtime.py` (shared runtime helpers) remain
as top-level Python files. Use the canonical modules below; no legacy module
aliases or root-package shim are provided. State-dict checkpoints remain
supported; override `--data-root` or `data.root` if a saved data path has moved.

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

Import implementations directly from these canonical modules, for example:

```python
from spvtp.vision_traj.data.dataset import VisionTrajDataset
from spvtp.vision_traj.models.policy import VisionTrajPolicy, VisionTrajPolicyConfig
from spvtp.vision_traj.evaluation.metrics import compute_batch_metrics
```

Use `spvtp train`, `spvtp eval`, and `spvtp rollout` for commands. Distributed
training can launch `python -m spvtp.vision_traj.scripts.train` through Accelerate.
Old flat-module import paths and module commands are intentionally unsupported.

`VisionTrajDataset.action_arrays(i)`
returns recorded history/targets without video decoding. New policies should
reuse the same splits, sample noise and metric functions instead of duplicating
the evaluation loop. Never add a process-global `sys.path` rewrite on import.

## Checks

```bash
uv pip install -e '.[dev]'
OMP_NUM_THREADS=2 python -m pytest -q
```

All automated tests live in the repository-root `tests/`, outside the runtime
package. `pyproject.toml` limits pytest discovery to this directory, including
both unittest cases and the visualization test. Run a subset with, for example,
`python -m pytest tests/test_rollout.py -q`.

Tests also guard the two-file top-level layout, canonical CLI targets, and
removal of all 28 compatibility modules and in-package tests. They cover
shapes/prompts, fixed splits, known metric values, noise invariance,
cache/direct input equality, recording lifecycle, alignment/processing, and
train/checkpoint/eval on generated videos. They need no pretrained download.
The separate manual smoke script, `python -m tests.test_shapes`, uses a
pretrained DINOv2 backbone and may download its weights; it is not collected
by the automated suite.
See [GPU validation](gpu_validation.md) for real-data checks; CPU/synthetic tests
do not replace camera hardware or full-training validation.

Namespace migration passed 50 CPU tests, offline wheel/console checks and a
three-scene, 128-window GPU evaluation of a pre-rename checkpoint (maximum
metric difference 0). See the [migration evidence](../../../runs/namespace_migration_20261104/summary.json).
