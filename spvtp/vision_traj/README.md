# SPOT: spatially prompted trajectory prediction

This package contains the model, data loading, training, and offline evaluation
code for **Spatially Prompted Visual Trajectory Prediction for Egocentric
Manipulation**. SPOT uses a spatial prompt on the first frame to specify an
object and destination, then combines the current RGB observation and recent
EE motion to predict a chunk of future end-effector poses and gripper widths.

See the [project README](../../README.md) for the paper, dataset, installation,
and data-collection overview. This guide focuses on using and extending the
trajectory-prediction package.

[Layout](#package-layout) · [Data](#data-contract) · [Model](#model-and-prompts) ·
[Training](#training) · [Evaluation](#evaluation) ·
[VLA adapters](#vla-adapters) · [Python API](#reusable-python-api) ·
[Checks](#tests-and-troubleshooting)

## Package layout

```text
vision_traj/
├── configs/config.yaml   # Reference SPOT configuration
├── data/                 # Episode indexing, splits, prompts, video/frame cache
├── models/               # Task/observation encoders and trajectory heads
├── engine/               # Training, validation, and checkpoint orchestration
├── evaluation/           # Shared metrics, window evaluation, feedback rollout
├── baselines/            # External VLA adapters and their evaluation helpers
├── analysis/             # Phases, pick/place boundaries, multi-seed summaries
├── scripts/              # Thin train/eval/rollout/visualization entry points
├── utils/                # Rotation conversions and visualization helpers
├── shs/                  # Shell recipes, including historical experiments
├── docs/                 # Detailed guides and validation notes
├── runtime.py            # Shared config, dataset, loader, and checkpoint helpers
└── __init__.py           # Lazy public exports
```

Automated tests live in the repository-root [`tests/`](../../tests/), not in
this package. Legacy flat modules such as `vision_traj/train.py` and
`vision_traj/policy.py` have been removed. Use the unified `spvtp` CLI or import
the implementations from their canonical subpackages.

All commands below assume an activated Python environment, an editable install
(`uv pip install -e .`), and the **repository root** as the working directory.
Do not run them from this directory. `python -m spvtp` can replace `spvtp`.

## Data contract

### Processed episodes and annotations

Training consumes processed EgoSPT episodes, not raw capture sessions:

```text
<data-root>/<scene>/<task>/recording_output_processed/<episode>/
├── camera_1.mp4       # RGB video
├── pose_interp/      # Zarr array: [T, 4, 4] EE poses
├── gripper_widths/   # Zarr array: [T], meters
└── valid_indices/    # Zarr array: [T], boolean mask over raw video frames
```

The episode directory is a Zarr group. The merged annotation JSON identifies
each episode/video and provides first-frame boxes in **source-image pixels**:

```json
{
  "scene1/put_fork1_to_plate1/episode_1/camera_1.mp4": {
    "0": {
      "object": [100, 120, 180, 240],
      "target": [300, 180, 500, 360]
    }
  }
}
```

Boxes use `[x1, y1, x2, y2]`; the loader normalizes them for the model.
For capture, synchronization, calibration, and annotation, follow the
[data guide](docs/dataset_analyze.md). Existing compact-valid legacy arrays are
expanded onto the raw frame axis; incompatible array lengths raise an error.

### Sampling and splits

The default loader selects valid frames and then subsamples them with
`data.stride=3`, nominally 30 fps to 10 fps. Missing/invalid frames can make
actual time intervals nonuniform. Each ordinary training/evaluation window
needs `K=4` preceding and `H=16` future selected frames.

Validation holds out **tasks within each scene**, keeping every episode of a
held-out task together. Defaults are `data.val_ratio=0.1` and
`data.split_seed=42`, independent of `train.seed`. Training and validation share
scene identities; this is **not an unseen-scene generalization protocol**.
Keep `split_manifest.json` to reproduce the exact episode membership.

### Optional frame cache

```bash
spvtp cache --data-root /path/to/EgoSPT \
  --annotations /path/to/annotations.json \
  --output-root /path/to/local_frame_cache \
  --stride 3 --image-size 224 --workers 4
```

Set `data.frame_cache_root=/path/to/local_frame_cache` during training, or
`--frame-cache-root` during eval/rollout. The cache stores resized RGB arrays,
including first-frame prompt variants, and avoids repeated random MP4 seeks.
Use node-local storage when available. Rebuild after changing annotations,
sampling, image size, or processed frame alignment. See the
[cache contract](docs/dataset_analyze.md#frame-cache).

## Model and prompts

The reference SPOT model uses a frozen **DINOv2-Base** visual backbone,
a task encoder, an observation encoder, and a **flow-matching** trajectory head.
The task path combines first-frame image features with role-tagged object and
target coordinates. The observation path combines current-image features with
recent actions. By default the two paths share the visual backbone.

### Inputs and outputs

The following shapes include batch dimension `B`; defaults are image size
`S=224`, history `K=4`, and future horizon `H=16`:

| Batch field | Shape | Meaning |
| --- | --- | --- |
| `first_frame` | `[B, 3, S, S]` | Fixed first RGB frame, optionally with rendered prompts |
| `current_frame` | `[B, 3, S, S]` | RGB at the current query timestep |
| `prompt_object` | `[B, 4/2/0]` | Normalized box, point, or absent object coordinates |
| `prompt_target` | `[B, 4/2/0]` | Normalized box, point, or absent target coordinates |
| `action_history` | `[B, K, 10]` | Past EE poses relative to the current EE, plus widths |
| `future_actions` | `[B, H, 10]` | Training targets and evaluation references, not inference conditioning |

Images are RGB floats in `[0, 1]`; the loader does not apply ImageNet mean/std
normalization. Prediction returns `[B, H, 10]` in denormalized physical units.
Each waypoint contains translation (3), the first two rotation-matrix columns
as a 6D representation (6), and **absolute gripper width** (1).

All past and future poses use the current EE as their common origin:

```text
T_relative(s) = inverse(T_EE(t)) @ T_EE(s)
T_future(t+h) = T_EE(t) @ T_relative(t+h)
```

These are **current-relative waypoints**, not incremental actions between
successive predictions. Translation and width use meters internally; rotation
6D values are dimensionless. Action mean/std come from training targets only.

### Prompt modes and controlled variants

| `data.prompt_form` | First-frame overlay | Coordinate tokens |
| --- | --- | --- |
| `vision_bbox_and_bbox` (default) | Object and target | Both boxes |
| `bbox` | None | Both boxes |
| `point` | None | Both box centers |
| `vision_bbox` | Object and target | None |
| `object_only` | Object | Object box only |
| `target_only` | Target | Target box only |
| `none` | None | None |

`none` removes the cues but **retains the unmarked first frame and history**.
To train Current Frame Only, disable both the task input and history:

```bash
spvtp train --config /path/to/run.yaml \
  policy.use_task_input=false policy.history_horizon=0 \
  data.prompt_form=none data.sample_start_step=4 \
  train.output_dir=runs/current_frame_only
```

Explicit `sample_start_step` keeps eligible windows paired when varying history
length. Other supported controls include `policy.head_type=diffusion`,
`policy.task_fusion_type=self_attention`, backbone freezing, and a separate
`policy.observation_encoder`. See [model details](docs/model_arch.md) and
[controlled experiments](docs/experiment.md) before changing multiple factors.

## Training

### Reference run

```bash
spvtp train \
  data.root=/path/to/EgoSPT \
  data.annotations_json=/path/to/annotations.json \
  train.output_dir=runs/spot
```

[`configs/config.yaml`](configs/config.yaml) is the source of defaults:
`D=768`, `H=16`, `K=4`, 4 task-fusion layers, 6 trajectory-head layers,
10 inference steps, AdamW lr `2e-4`, 30 epochs, and bf16. WandB is off.
Dotted overrides apply after the YAML selected by `--config FILE`.

The default effective batch is **128**: batch 16 × accumulation 8 × one
process. For two GPUs, preserve that batch with accumulation 4:

```bash
accelerate launch --num_processes 2 --multi_gpu \
  -m spvtp.vision_traj.scripts.train \
  data.root=/path/to/EgoSPT \
  data.annotations_json=/path/to/annotations.json \
  train.grad_accum_steps=4 train.output_dir=runs/spot
```

Equal effective batch sizes do not guarantee bit-identical training across GPU
counts. Historical `shs/` recipes can assume old paths or GPU counts; prefer
the current CLI and inspect a shell script before reusing it.

### Short pipeline check and resume

```bash
spvtp train data.root=/path/to/EgoSPT \
  data.annotations_json=/path/to/annotations.json \
  train.output_dir=runs/smoke train.max_steps=3 train.log_every=1 \
  train.val_every_epochs=1 train.val_max_batches=1 train.run_final_eval=false

spvtp train --config runs/spot/config.yaml \
  --resume runs/spot/ckpt_ep0009.pt train.output_dir=runs/spot
```

`max_steps` counts optimizer updates, not microbatches. A smoke run is a
pipeline check, not a convergence result. For CPU debugging, additionally set
`train.device=cpu train.mixed_precision=no`.

Runs save `config.yaml`, `split_manifest.json`, periodic `ckpt_epXXXX.pt`, and
`ckpt_final.pt`. Resume restores the available model/optimizer/scheduler state,
but **does not select the checkpoint configuration automatically**: pass the
saved `--config`. Dataloader position/RNG state are not restored for exact
mid-epoch replay. Periodic validation is capped by `train.val_max_batches`;
final evaluation uses the full validation set unless disabled. See
[training details](docs/training.md).

## Evaluation

### Independent windows with recorded history

```bash
spvtp eval --ckpt runs/spot/ckpt_final.pt \
  --output-dir runs/spot/eval --batch-size 32 \
  --save-step-metrics --viz-samples 12
```

The evaluator loads configuration from the checkpoint and uses the adjacent
split manifest when applicable. Use `--split-manifest` explicitly to fix split
membership. For relocated data, pass `--data-root`, `--annotations`, and
`--frame-cache-root`. The default evaluates all eligible validation windows;
`--n-episodes` intentionally limits coverage.

Each window uses recorded current RGB and recorded action history. Predictions
are **not** fed into the next window. Outputs include `all/` and per-scene
`metrics.csv` / `summary.json`; optional step records and plots support further
analysis. `--sample-seed` makes window noise independent of batch grouping,
but changing subset indexing can change the assigned noise.

Position L2 and gripper L1 average errors over a chunk; ordinary **FDE is the
last-waypoint error of each H-step chunk**, not the final error of an episode.
Rotation is also reported as an SO(3) geodesic angle. **G-end** measures the
endpoint gripper-state match; **Combined@10** requires endpoint position error
≤10 cm, rotation error ≤20°, and matching gripper state. These threshold rates
are offline trajectory criteria, not physical task-success rates.

Console distances are **cm** and rates are percentages. Saved CSV/JSON distances
remain **m**, angles are degrees, and rates are fractions; convert exactly once
when preparing tables. See [metric definitions and outputs](docs/evaluation.md).

### Sequential predicted-history feedback

```bash
spvtp rollout --ckpt runs/spot/ckpt_final.pt \
  --split-manifest runs/spot/split_manifest.json \
  --output-dir runs/spot/rollout
```

Three paired modes are evaluated: `gt_history` uses recorded history and pose
anchors; `history_feedback` uses predicted history with recorded pose anchors;
`cumulative` uses predicted history and predicted pose anchors, without later
GT pose resets. The first-frame prompt stays fixed. Queries advance by H steps,
and available partial tails are scored rather than discarded.

**RGB is always recorded replay.** Predictions do not affect later images,
contacts, object motion, or episode termination. This is an error-accumulation
diagnostic, not a closed-loop robot evaluation. Rollout summaries use
episode-balanced ADE and **episode-end FDE**; do not directly equate them with
the overlapping-window metrics above. See [rollout protocol](docs/rollout.md).

For phase/pick-place analyses and seed aggregation, start with
`spvtp phases --help`, `spvtp boundaries --help`, `spvtp pick-place --help`, and
`spvtp multiseed --help`. Prompt interventions are available through
`spvtp eval --prompt-intervention MODE`; retraining No Prompt and removing cues
only at evaluation are different experiments.

## VLA adapters

```bash
spvtp vla --help
spvtp vla groot --help
spvtp vla pi05 --help
spvtp vla smolvla --help
spvtp vla openvla_oft --help
```

These are **external-policy adapters**, separate from the SPOT model. They
reuse the processed episodes and trajectory targets, but have backend-specific
pretrained weights, processors, fine-tuning settings, and dependencies. The
generic `.[vla]` extra alone does not install every upstream runtime.

Input protocols are not interchangeable; the current implementations are:

- **GR00T N1.6:** supports `visual_prompt` and `language_only`; the CLI defaults
  to the former. For language-only comparisons explicitly pass
  `--prompt-protocol language_only --language-map /path/to/language_map.json`.
  Requires the Isaac-GR00T runtime and model weights.
- **SmolVLA:** language-only task conditioning, current RGB, and four-step
  action history; no first-frame image or boxes. Requires a language map,
  model weights, and a compatible LeRobot runtime.
- **OpenVLA-OFT:** language-only task conditioning and current RGB; supports
  history horizons 4 and 0. Use `--history-horizon 0 --sample-start-step 4` for
  the history-free paired setting. Requires model weights, a language map, and
  `--oft-source` pointing to the compatible source directory.
- **π0.5:** the adapter currently retains the prompted first frame, current RGB,
  action history, and a generic instruction. It is **not a language-only
  implementation**. Requires its compatible LeRobot port, weights, and
  `--tokenizer-model`.

For language-only adapters, the instruction template is
`pick {object} and place to {target}`. Inspect the map to distinguish category
names from instance IDs. Scene-3 boxes can be used offline to construct the
language annotations; they are not then passed as visual model inputs.
Always record the exact input protocol, history, language map, split, seed,
and evaluation coverage alongside results. See [experiment notes](docs/experiment.md).

## Reusable Python API

Prefer the shared runtime over rebuilding configuration, split, normalization,
or checkpoint logic in a new script. For example, evaluate a saved SPOT model
on one CPU batch without downloading a separate backbone checkpoint:

```python
from spvtp.vision_traj.runtime import (
    load_checkpoint, make_dataset, make_initial_noise, make_loader, select_episodes,
)

policy, cfg, _ = load_checkpoint("runs/spot/ckpt_final.pt", device="cpu")
episodes = select_episodes(cfg, "val", manifest="runs/spot/split_manifest.json")
dataset = make_dataset(cfg, episodes)
batch = next(iter(make_loader(dataset, batch_size=1)))
inputs = {key: value for key, value in batch.items()
          if key not in {"future_actions", "is_final_chunk"}}
noise = make_initial_noise([0], cfg.policy.action_horizon, cfg.policy.action_dim)
policy.eval()
prediction = policy.predict_action(inputs, initial_noise=noise)  # [1, H, 10]
```

The checkpoint contains the action normalizer and full model state. If data
have moved, override `cfg.data.root`, `cfg.data.annotations_json`, and
`cfg.data.frame_cache_root` before constructing the dataset. Only load trusted
checkpoints. For a model built from scratch, initialize its action normalizer
from training data, as the trainer does.

Canonical implementation imports include:

```python
from spvtp.vision_traj.data.dataset import VisionTrajDataset, build_episode_index
from spvtp.vision_traj.models.policy import VisionTrajPolicy, VisionTrajPolicyConfig
from spvtp.vision_traj.evaluation.metrics import compute_batch_metrics
```

`VisionTrajDataset.action_arrays(index)` returns history/targets without video
decoding. Reuse `evaluation.runner` for window evaluation and
`evaluation.rollout` for sequential feedback instead of duplicating metric or
coordinate logic. See [development and module ownership](docs/development.md).

## Tests and troubleshooting

Run the unified automated suite from the repository root:

```bash
uv pip install -e '.[dev]'
OMP_NUM_THREADS=2 python -m pytest -q
```

Tests use generated data and do not require pretrained downloads. The raw-data
synchronization test skips when optional `hydra-core`, `scipy`, `ffmpeg-python`,
`av`, or `colorama` are unavailable. The separate manual
`python -m tests.test_shapes` smoke script can download DINOv2 weights; it is
not collected by pytest. Neither CPU tests nor a short GPU smoke run replace
hardware validation or a complete training experiment.

Before a long run, verify video decoding, mask/pose/width alignment, annotation
coverage, a nonempty dataset, split disjointness, and one batch from each scene.
For cache mismatches, rebuild with the current stride/image size and alignment.
For import errors, refresh the editable install and use canonical subpackages.
For missing data after resume or relocation, check saved paths and pass the
appropriate configuration/CLI overrides rather than silently changing splits.

Detailed references: [documentation index](docs/README.md) ·
[GPU validation evidence](docs/gpu_validation.md) ·
[default configuration](configs/config.yaml).
