# Offline evaluation and metrics

For chronological predicted-history feedback and cumulative pose rollouts, see
[the separate rollout protocol](rollout.md). Ordinary eval below keeps recorded
history and does not feed predictions into later windows.

```bash
spvtp eval --ckpt /path/to/ckpt_final.pt --data-root /path/to/EgoSPT \
  --annotations /path/to/annotations.json --output-dir runs/eval \
  --batch-size 32 --save-step-metrics --viz-samples 12
```

## Protocol

Configuration comes from the checkpoint. The adjacent `split_manifest.json`
is used when available, unless split-defining options are overridden; pass
`--split-manifest FILE` explicitly when needed. By default, all validation
episodes and all eligible windows are evaluated. `--n-episodes` limits episodes;
`--episode-indices` selects positions in the split; `--scenes` filters scenes.
`--sample-seed` sets inference noise per dataset sample index, independent of
batch grouping. Reindexing a subset changes that sample's noise assignment.

Each window uses the recorded current RGB and recorded past EE actions.
The model predicts H=16 future waypoints without seeing future images/actions.
Subsequent windows again use recorded observations/history, not predictions.
Thus neither endpoint accuracy nor Combined success is closed-loop robot success.

## Formulas

For a sample with H waypoints, let e_h be Euclidean translation error, d_h
Euclidean error between the two 6D rotation representations, g_h absolute width
error and theta_h the SO(3) geodesic angle. Convert predicted/GT 6D rotations
to matrices before computing
`theta_h = acos(clamp((trace(R_pred.T @ R_gt)-1)/2, -1, 1))` in degrees.

| Output | Per-sample calculation |
| --- | --- |
| `pos_l2` | mean_h e_h |
| `final_displacement_error` | e_H; endpoint of every predicted chunk |
| `rot6d_l2` | mean_h d_h; dimensionless, not degrees |
| `rotation_geodesic_deg` | mean_h theta_h |
| `final_rotation_geodesic_deg` | theta_H |
| `grip_l1` | mean_h g_h |
| `grip_state_accuracy` | mean_h matching binary gripper states |
| `final_grip_state_correct` (G-end) | matching state at H |
| `success_at_{2,5,10}cm` | e_H <= the indicated threshold |
| `rotation_success_{10,20}deg` | theta_H <= the indicated threshold |
| `combined_success_{5,10}cm_20deg_grip` | endpoint position AND theta_H <= 20 degrees AND G-end |

Gripper state is open iff width > 0.03 m by default; override with
`--gripper-state-threshold` in meters. Combined@10 uses 10 cm. Aggregates are
arithmetic means over windows (not episode-balanced averages); overlapping
windows are correlated. `final_grip_state_accuracy` is the mean of G-end.

## Outputs and units

A single inference pass writes `all/` and one directory per scene, each with
`metrics.csv` and `summary.json`. `--save-step-metrics` adds
`all/step_metrics.csv` with waypoint/frame indices and predicted/GT widths for
phase analyses. `--viz-samples` adds plots and `selected_samples.csv`.

Stored distances are **meters**, angles **degrees**, success/state values
**fractions**, and `rot6d_l2` dimensionless. JSON includes unit metadata. Console
output converts position/FDE/gripper to **cm** and success to **percent**.
Multiply native distances by 100 exactly once when preparing a cm table.
Historical reports may use other display conventions; check their metadata.

Additional analysis commands: `spvtp phases`, `boundaries`, `pick-place`,
`multiseed`. Prompt robustness: `spvtp eval --prompt-intervention MODE ...`;
use `--help` for available interventions and options.
