# Migration GPU validation — 2026-11-04

Host: action10; GPU 0: RTX 6000 Ada, 48 GB. PyTorch 2.5.1+cu121.
These checks use real EgoSPT videos/arrays and the original SPOT architecture;
they are implementation checks, not new completed training experiments.
Normal GPU training/checkpoint/evaluation passed. Initial checks exposed two
issues: the first resumed optimizer update and compact-array frame caching.
Both code fixes are now applied; see [fix verification](#fix-verification).
The original failure evidence below is retained as historical context.

## Same-input comparison

The old and refactored packages were imported in separate processes. Both
loaded the original `baseline_vision_bbox_and_bbox_all_scenes/ckpt_final.pt`
strictly, using the same six real samples (first/last window of one validation
episode per scene) and the same per-sample Gaussian noise.

- Trainable parameters: 97,568,266, matching the old log's 97.57M.
- FP32 prediction maximum absolute difference: 0.
- bf16 training loss: 0.4224250019 in both implementations; difference: 0.
- Three checked output/decoder parameter gradients: maximum difference 0.
- Eight shared position/rotation/gripper metric tensors: difference 0.
- The 42 existing CPU regression tests also passed (25 model/analysis + 17 workflow/processing).

Raw results: [parity.json](../../../runs/gpu_validation_20261104/parity.json).

## Dataset differences are intentional, not numerical parity

Legacy processing sometimes compacted valid poses/widths but retained a
raw-frame validity mask. The old loader clipped arrays to a common length;
the new loader expands compact arrays onto the raw frame axis.

| Selected episode | Old windows | Corrected windows |
| --- | ---: | ---: |
| scene1 / put_fork1_to_bowl2 / episode_0 | 65 | 66 |
| scene2 / put_fork4_to_bowl1 / episode_0 | 38 | 38 |
| scene3 / subscene6 / episode_1 | 23 | 24 |

At matched episode/time steps, RGB and prompt tensors are bitwise equal, but
some history/target actions differ because of corrected alignment. Comparing
the same flat index after a window-count change would compare different
samples, and the old loader could silently substitute another sample on error.
Therefore model parity above uses identical tensors explicitly. Historical
aggregate scores cannot be treated as a strict end-to-end reproduction after
this data correction.

## Original checkpoint through the new eval CLI

The three episodes contribute 128 windows. Distances below are cm.

| Scope | Windows | Position L2 | FDE | Gripper L1 |
| --- | ---: | ---: | ---: | ---: |
| scene1 | 66 | 8.579 | 13.081 | 0.618 |
| scene2 | 38 | 5.189 | 10.114 | 0.535 |
| scene3 | 24 | 6.107 | 11.870 | 0.326 |
| all | 128 | 7.109 | 11.973 | 0.539 |

The historical full-val log covered 10,024 windows: position 6.994 cm,
FDE 11.466 cm, gripper 1.133 cm. Coverage, alignment and inference noise indexing
differ; these numbers only establish historical scale, not an accuracy delta.
JSON/CSV, waypoint records and plots are under
[reference_eval/](../../../runs/gpu_validation_20261104/reference_eval).

## Short training pipeline

The original full-store startup was stopped after over eight minutes waiting
on shared-filesystem RPC, before any optimizer update. Its log is retained as
`train_full_index_attempt.log`. For a bounded smoke check, 12 training episodes
(four per scene) and the three validation episodes above were selected from
the original split, without changing architecture or assigning validation
episodes to training.

The subset has 487 training and 128 validation windows. Three optimizer
updates completed with bf16, batch 16 and accumulation 8 (global batch 128).
Logged losses were 2.22473, 2.23032 and 2.52083, all finite. Periodic validation,
checkpoint saving, standalone checkpoint reload, all/per-scene evaluation,
waypoint CSVs and visualization passed. These nearly untrained losses/scores
are not compared as convergence results against a 30-epoch historical model.

The initial resume command omitted `--config` and failed on the packaged default
annotation path, before training. The current CLI does not load checkpoint
configuration automatically on resume. The [training recipe](training.md#checkpoints-and-evaluation)
now explicitly passes the saved config. With that correction, the run reached
epoch 2 / step 4 with finite loss 2.37466 and saved optimizer/scheduler counters
at 4. However, it did **not** perform an effective first weight update:

- The saved optimizer LR was 1e-4.
- Constructing the warmup scheduler reset the active optimizer LR to 0.
- Loading scheduler state did not restore that optimizer parameter-group LR.
- The three checked decoder/output weights had maximum change 0 after step 4;
  validation metrics were also unchanged.

The cause was the restore/constructor order in
[`engine/trainer.py`](../engine/trainer.py). Resume is therefore not marked
fully passed in the original run. See [resume_update.json](../../../runs/gpu_validation_20261104/resume_update.json)
and [audit.json](../../../runs/gpu_validation_20261104/audit.json).

## Cache compatibility finding

A generated-video regression reproduced `cache frame-count mismatch` with
compact-valid legacy arrays: the cache builder truncated frame selection while
the loader expanded arrays. Direct MP4 loading was unaffected.
This was the pre-fix behavior; current code supports compact arrays and rebuilds
old cache metadata. See the original failure in
[compact_cache.json](../../../runs/gpu_validation_20261104/compact_cache.json).

Not covered: live capture hardware, a complete 30-epoch run, full-validation
quality reproduction, external VLA runtime training or GPU DDP performance.
Old research results were not overwritten. Temporary model/checkpoint artifacts
remain under `/tmp/spvtp-docs-gpu.9irGPz/artifacts`; compact logs/metrics are
preserved in the project's `runs/gpu_validation_20261104/`.
Exact diagnostic scripts are saved in that run's `commands/` directory.

## Fix verification

Both fixes are implemented without changing the model or action setting:

- The cache builder reuses `align_valid_frames`, selects the full valid raw-frame
  sequence and stores its SHA-256 fingerprint. Cache checks rebuild old metadata,
  truncated arrays or changed masks even without `--overwrite`.
- Training constructs/restores the scheduler before loading optimizer state,
  preserving the checkpoint's actual learning rate on the first resumed update.

All **47 CPU tests passed** (25 model/analysis + 22 workflow/processing).
The [five new regression tests](../../../tests/test_regressions.py) all failed
before the fix and passed afterward. They cover compact arrays across three
strides/seven prompts, stale/truncated caches, same-count mask changes, malformed
array lengths and an actual first resumed weight update.

On action10 GPU 0, 15 real episodes were cached. Three-scene validation episodes
passed all 42 first/last-window comparisons across seven prompts: every returned
tensor was bitwise equal between cache and direct decoding.

The bf16 resume check retained 487 train/128 validation windows and global batch
128. Starting from step 3, the actual first update used saved LR **1e-4** and
reached optimizer/scheduler step 4. Checked parameter changes were:

| Parameter | Maximum absolute change |
| --- | ---: |
| `flow_head.action_out.weight` | 1.0051113e-4 |
| `flow_head.action_out.bias` | 1.0129064e-5 |
| `flow_head.decoder.layers.5.norm3.bias` | 3.1836084e-5 |

Raw evidence: [cache](../../../runs/gpu_validation_20261104/fixes/real_cache.json),
[resume update](../../../runs/gpu_validation_20261104/fixes/resume_update.json).
The resumed checkpoint's final eval and standalone reload both covered all 128
windows across three scenes; the five checked aggregate metrics agreed within
relative tolerance 1e-6 / absolute tolerance 1e-7. Waypoint CSVs and a trajectory
plot were also generated. The [combined verification summary](../../../runs/gpu_validation_20261104/fixes/fix_summary.json)
records the passing CPU/GPU checks and source hashes.

Original pre-fix failure JSONs above are retained as historical evidence, not
the current status. Rebuild pre-fix caches once by rerunning `spvtp cache`.
Fixed-run temporary checkpoint/cache: `/tmp/spvtp-fixes.39jKW6/artifacts/`.
