# Model and prompts

The default SPOT policy combines a task encoder, an observation encoder and a
flow-matching trajectory head. [`configs/config.yaml`](../configs/config.yaml)
is the source of default hyperparameters.

## Inputs and actions

Each sample at time t contains first-frame RGB, current RGB, object/target
prompts, K=4 recorded actions and H=16 future targets. RGB tensors are
`[3,224,224]`; action history is `[K,10]`, targets `[H,10]`.
Images are RGB in [0,1]; preprocessing does not apply ImageNet mean/std.

All past/future poses are expressed relative to the current EE pose:
`T_rel(s) = inverse(T_EE(t)) @ T_EE(s)`. Each action concatenates translation
(3), the first two rotation-matrix columns (6) and absolute gripper width (1).
Translation/width use meters internally. Future waypoints are t+1 through t+16;
these are current-relative targets, not incremental deltas between waypoints.

## Encoders and head

- Task: shared frozen DINOv2-Base image tokens + role-tagged Fourier-encoded
  object/target coordinates, fused by 4 cross-attention layers (D=768, 12 heads).
  The default returns an image summary plus prompt tokens.
- Observation: the current RGB uses the same image backbone/projection;
  an MLP embeds the recorded history, with position and type embeddings.
  `policy.observation_encoder` can select a separate visual tower.
- Head: a 6-layer, 12-head transformer conditioned on both token sequences.
  Default flow matching samples t from Beta(1.5,1), learns the velocity from
  Gaussian noise to normalized actions, and predicts with 10 Euler steps.
  `policy.head_type=diffusion` selects the DDPM/DDIM alternative.

The training split supplies per-dimension target mean/std (256 sampled windows
by default). Prediction is denormalized to physical units. Backbone weights
are frozen by default; prompt/fusion/history/head modules are trainable.

## Prompt variants

Boxes are `[x1,y1,x2,y2]` normalized by source image dimensions; points use box
centers. Visual boxes are drawn before resizing, on the first frame only.

| `data.prompt_form` | First-frame overlays | Coordinate inputs |
| --- | --- | --- |
| `vision_bbox_and_bbox` (default) | object + target | both boxes |
| `bbox` | none | both boxes |
| `point` | none | both centers |
| `vision_bbox` | object + target | none |
| `object_only` | object | object box |
| `target_only` | target | target box |
| `none` | none | none |

`none` still provides the unmarked first frame and history. It is not the
current-frame-only baseline. To remove the task path and history together:

```bash
spvtp train --config /path/to/run.yaml policy.use_task_input=false \
  policy.history_horizon=0 data.prompt_form=none data.sample_start_step=4
```

`sample_start_step=4` preserves the reference model's eligible windows.
Offline train/eval use recorded history; no predictions are fed into subsequent
samples. This protocol is not a closed-loop controller rollout.
