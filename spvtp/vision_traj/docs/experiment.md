# Controlled experiments

This is a compact recipe/reference, not a record of completed runs. Original
research results remain in UMI-Iphone's `runs/vision_traj_experiments_final/`
and submission experiment report; migration does not retrain or replace them.

## Keep comparisons paired

Use the same dataset, merged annotations, split manifest, eligible windows,
action horizon, global batch, schedule and evaluation protocol. Change only the
factor under test. Train on the three scenes together; report `all` and each
scene. Use the same training seeds across methods, keeping `data.split_seed=42`.
This setting does not establish unseen-scene generalization.

| Comparison | Configuration change |
| --- | --- |
| Prompt form | `data.prompt_form`: see [definitions](model_arch.md#prompt-variants) |
| History | `policy.history_horizon=0/4/8/12`; fix `data.sample_start_step` for paired windows |
| Current frame only | `policy.use_task_input=false`, K=0, prompt `none`, sample start 4 |
| Action head | `policy.head_type=flow_matching/diffusion` |
| Task fusion | `policy.task_fusion_type=cross_attention/self_attention` |
| Backbone tuning | `policy.vision_encoder.freeze=false`; record changed learning rate |
| Backbone family | DINOv2, SAM, PE, SigLIP or EVA2 via `policy.vision_encoder` |
| Separate VLA observation tower | `policy.observation_encoder`; keep task encoder/head fixed |

For DINOv2 small/base/large, original `(embed_dim, heads)` settings were
`(384,6)`, `(768,12)`, `(1024,16)` respectively (both fusion/head attention).
Original frozen-base lr sweep: 1e-4/2e-4/5e-4; unfrozen-base lr: 1e-5.
These are historical ablation choices, not universal tuned defaults.

## Language-conditioned VLA baselines

`spvtp vla {groot,pi05,smolvla,openvla_oft} --help` exposes optional adapters.
They require upstream runtimes and local pretrained weights; pi0.5 also requires
a tokenizer and OpenVLA-OFT its source path. Select language-only task
conditioning without a first-frame visual prompt when comparing that setting.
Current RGB is still an observation. Check each adapter's history setting;
do not assume all VLA architectures consume the same temporal inputs.
Record the exact language string and whether object/target IDs are included.

## Record results

Keep config, manifest, seed, checkpoint path, train log, validation coverage and
all/per-scene JSON/CSV outputs for every run. Report position/FDE/gripper in cm,
rotation in degrees (6D L2 separately), endpoint thresholds and Combined@10.
Across seeds, report mean/std and the number of completed seeds. Partial smoke
runs are implementation checks, not completed experimental results.

Historical shell launchers remain under `shs/experiments/`. Some expect the old
Git repository, data location or eight-GPU setup. Prefer the current CLI for
new runs; inspect scripts and explicitly set paths, process count and
accumulation before reusing them. Do not treat an old Excel workbook or launcher
default as the current source of configuration.
