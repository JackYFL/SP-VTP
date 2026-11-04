# Training

Use the [quickstart](../../../README.md#train) for single/two-GPU commands.
`spvtp train --config FILE key=value ...` loads YAML then applies dotted
overrides. Without `--config`, it loads [`configs/config.yaml`](../configs/config.yaml).
Use absolute data/output paths when launching outside the project root.

## Runtime controls

The reference architecture uses combined visual/coordinate prompts, frozen
DINOv2-Base and flow matching. Defaults: AdamW lr=2e-4, weight decay=1e-4,
cosine schedule with 5% warmup, gradient clipping=1, bf16, 30 epochs.
WandB is off; enable with `.[tracking]` and `train.wandb.enabled=true`.

Effective batch = `train.batch_size * train.grad_accum_steps * GPU processes`.
The default is 16 x 8 x 1 = 128; use accumulation 4 on two GPUs for the same
effective batch. Holding batch size fixed alone does not make experiments
equivalent when the process/accumulation count changes.

`data.num_workers` controls workers per process. Use a matching local frame
cache to avoid repeated random MP4 seeks. CPU debugging requires
`train.device=cpu train.mixed_precision=no`.

## Short GPU check

```bash
CUDA_VISIBLE_DEVICES=0 spvtp train data.root=/path/to/EgoSPT \
  data.annotations_json=/path/to/annotations.json train.output_dir=runs/smoke \
  train.max_steps=3 train.log_every=1 train.val_every_epochs=1 \
  train.val_max_batches=1 train.run_final_eval=false

spvtp eval --ckpt runs/smoke/ckpt_final.pt --output-dir runs/smoke/eval \
  --n-episodes 3 --viz-samples 1
```

`max_steps` counts optimizer updates, not microbatches. The capped epoch still
performs configured validation/checkpointing. The example limits validation
for speed and disables the default full final eval; this is not a quality
measurement. The first three episodes need not cover all scenes: select an
explicit subset/manifest for three-scene checks.

## Checkpoints and evaluation

The output contains `config.yaml`, `split_manifest.json`, periodic
`ckpt_epXXXX.pt`, and `ckpt_final.pt`. Checkpoints store model/config plus
optimizer, scheduler, step and next-epoch state. Full model checkpoints already
contain image weights, so loading them does not download pretrained weights.

```bash
spvtp train --config runs/spot/config.yaml --resume runs/spot/ckpt_ep0009.pt \
  train.output_dir=runs/spot
```

`--resume` restores weights/optimizer/scheduler, not the configuration selected
by the CLI. Pass the saved `--config` to preserve data paths and architecture;
otherwise the CLI starts from its packaged defaults plus explicit overrides.
Resume is not a bit-exact mid-epoch replay:
the dataloader cursor/RNG state are not saved. Smoke checkpoints stopped inside
an epoch advance to the next epoch on resume. Old research checkpoints may
contain only model/config, so they do not resume optimizer/scheduler progress.

The scheduler is constructed before optimizer-state restoration, preserving the
checkpoint's active learning rate on the first resumed update. See
[fix verification](gpu_validation.md#fix-verification) for the regression checks.

Periodic validation uses `train.val_max_batches` (64 by default). Final
evaluation is full validation, on the main process, unless
`train.run_final_eval=false`. The standalone eval and training final eval reuse
the same metrics/runner. See [evaluation](evaluation.md) for formulas and files.
