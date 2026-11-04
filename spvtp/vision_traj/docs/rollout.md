# Sequential state/history feedback evaluation

This is a **replayed-RGB diagnostic**, not physical closed-loop robot control.
Normal `spvtp eval` remains the independent-window, recorded-history protocol.

```bash
spvtp rollout --ckpt /path/to/ckpt_final.pt --data-root /path/to/EgoSPT \
  --annotations /path/to/annotations.json --frame-cache-root /path/to/cache \
  --split-manifest /path/to/split_manifest.json --output-dir runs/rollout \
  --interventions none,shuffled_object,shuffled_target --batch-size 32
```

`--modes` defaults to `gt_history,history_feedback,cumulative`.
Load the separately trained No Prompt checkpoint for that baseline; removing
cues from the full-prompt checkpoint is a different intervention.

## State and coordinate protocol

The first-frame image/prompt is fixed throughout an episode. All modes share
the same recorded warm-start prefix (default query start K=4). Query steps are
4,20,36,... for H=16. Tail queries still predict H waypoints, but only the
available prefix is scored/stored, so the placement tail is not discarded.

| Mode | History after initialization | Chunk pose anchor |
| --- | --- | --- |
| `gt_history` | recorded past EE states/widths | recorded current EE |
| `history_feedback` | previous predicted states/widths | recorded current EE |
| `cumulative` | previous predicted states/widths | predicted current EE |

Each waypoint is relative to its chunk origin, not to the previous waypoint:
`T_pred(t+h) = T_anchor(t) @ delta_T_pred(t,h)`. Later history contains the K
states strictly *before* the current timestep, transformed with the inverse
of the new anchor. Width is absolute, in meters, and is not transformed.
The cumulative mode never resets its pose to GT after initialization.

The initial recorded history is identical in every mode. Subsequent predicted
history is constructed from saved predicted poses, not copied as old-frame 10D
vectors. `future_actions` and phase labels are excluded from model inputs.
Noise is seeded by canonical episode key plus query step, independently of
condition, mode, batching and training seed.

## Outputs

Each condition/mode writes `all` and per-scene directories:

- `summary.json` / `episodes.csv`: episode-balanced ADE and **episode-end** FDE,
  rotation, width and endpoint criteria over the complete scored trajectory.
- `chunks.csv`: every query, valid horizon, pose-anchor drift and chunk metrics.
- `full_chunks_summary.json`: only full-H chunks, excluding partial tails.
- `step_metrics.csv`: every valid future timestep exactly once, with replay-frame
  index, GT-defined phase, predicted/GT position and width, and waypoint errors.
- `phase_metrics.csv`: pooled/episode-balanced waypoint errors plus separately
  counted full-H endpoint FDE; empty endpoint groups remain missing, not zero.
- `drift_curve.csv`: episode-balanced errors in normalized-progress bins.

Native CSV/JSON distances are m, rotation is deg, and endpoint criteria are
fractions. The summary tool converts reader-facing lengths to cm and rates to
percent. Episode-end FDE (possibly from a partial tail) must not be confused
with the original fixed-16-step FDE. Progress curves average waypoints within
bins; their last bin need not equal the exact episode-end error.

```bash
spvtp rollout-suite --manifest /path/to/run_manifest.json
python -m spvtp.vision_traj.analysis.summarize_rollout --root runs/rollout_suite
```

The suite manifest supplies checkpoints, seeds, conditions and common paths.
Summarization checks paired query/noise identity, common initial predictions,
tail coverage and anchor reset behavior before computing seed mean/sample std,
per-scene tables, five-phase tables and drift plots. `--allow-partial` explicitly
labels incomplete summaries and must not be used for final three-seed claims.

## Interpretation

Current RGB remains the recorded frame at the scheduled timestep; predictions
do not change that image, the objects, contact dynamics or episode termination.
In the cumulative mode the predicted pose and replayed camera viewpoint can
therefore become inconsistent. These results are neither a deployment-success
measurement nor a guaranteed upper/lower bound on real closed-loop performance.
GT phases are post-hoc analysis groups, not model inputs. Compare modes on
matched episodes/queries; do not compare directly with old overlapping-window
tables or assume cached arrays have identical alignment across data versions.
