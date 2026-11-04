# SP-VTP

iPhone-based UMI data collection, EgoSPT preparation, SPOT training and offline
evaluation. Default inputs: first-frame visual + box prompts, current RGB and
4 recorded history actions; output: a 16-step, 10D relative EE trajectory.

## Install

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
uv pip install -e '.[data,annotation]'  # optional processing / annotation tools
```

Run from the project root. The Python package is `spvtp`; after updating an
existing checkout, rerun `uv pip install -e .` to refresh the console entry point.
You can also run `python -m spvtp --help`. Install `.[iphone]` on the Record3D capture
machine; OAK needs DepthAI 2.x. Match PyTorch to CUDA. Camera SDKs and external
VLA runtimes are optional and are not imported by ordinary training/evaluation.

## Collect and prepare

```bash
spvtp collect --backend iphone-gopro --data-dir recordings/session.zarr --fps 30
spvtp process --backend iphone-gopro --config /path/to/session.yaml
spvtp annotate --video /path/to/camera_1.mp4
spvtp merge-annotations --root /path/to/EgoSPT --output annotations_merged.json
```

Capture backends: `iphone-gopro`, `iphone-oak`, `iphone`. Keys: `s` start,
`w` save, `a` discard/restart, `q` exit. Processing needs the selected rig's
calibration. See [data and alignment](spvtp/vision_traj/docs/dataset_analyze.md).

## Train

```bash
spvtp train data.root=/path/to/EgoSPT data.annotations_json=/path/to/annotations.json \
  train.output_dir=runs/spot

# Two GPUs: 16 x 4 accumulation x 2 processes = global batch 128.
accelerate launch --num_processes 2 --multi_gpu -m spvtp.vision_traj.scripts.train \
  data.root=/path/to/EgoSPT data.annotations_json=/path/to/annotations.json \
  train.grad_accum_steps=4 train.output_dir=runs/spot
```

Defaults: frozen DINOv2-Base, flow matching, D=768, H=16, K=4, lr=2e-4,
30 epochs. One GPU uses batch 16 x accumulation 8. `data.split_seed=42`
is independent of `train.seed`. Use `--config FILE` and/or `key=value` overrides;
`train.max_steps` limits a smoke run. See [training](spvtp/vision_traj/docs/training.md)
and [model / prompt definitions](spvtp/vision_traj/docs/model_arch.md).

## Evaluate

```bash
spvtp eval --ckpt runs/spot/ckpt_final.pt --output-dir runs/spot/eval \
  --batch-size 32 --save-step-metrics --viz-samples 12
```

Loads checkpoint config and the adjacent split manifest; evaluates the full
validation split by default, writing overall and per-scene results in one pass.
Relocate data with `--data-root`, `--annotations`, `--frame-cache-root`.
History is recorded ground truth, not previous predictions. Console distances
are cm; stored distances remain meters with explicit unit metadata. Endpoint
accuracy is an offline proxy, not robot task success. See [metrics and outputs](spvtp/vision_traj/docs/evaluation.md).

For sequential feedback, use `spvtp rollout --help`; it supports recorded
history, predicted history with GT anchors, and cumulative predicted poses.
RGB is still replayed: see the [rollout protocol](spvtp/vision_traj/docs/rollout.md).

## Development

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s spvtp/vision_traj/tests -q
OMP_NUM_THREADS=2 python -m unittest discover -s tests -q
```

`spvtp --help` lists commands; `spvtp COMMAND --help` shows their options.
Optional adapters: `spvtp vla {groot,pi05,smolvla,openvla_oft} --help`.
See the [documentation index](spvtp/vision_traj/docs/README.md) for code reuse,
controlled experiments and GPU validation notes. The root `pyproject.toml` is
the only dependency/package specification; flat aliases such as
`spvtp.vision_traj.train` remain available within the new namespace.
