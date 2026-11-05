# Data collection and preparation

## Collect, process, annotate

The [quickstart](../../../../README.md#quickstart) contains the commands.
`RecordingSession` shares recording lifecycle/cleanup across camera backends.
For iPhone + GoPro/OAK, camera 0 is the iPhone pose source and camera 1 supplies
the RGB observations. Hardware SDKs are required only on the capture machine.

```bash
spvtp process --backend iphone-gopro input_dir=/path/to/raw/session.zarr \
  output_dir=/path/to/processed overwrite=false \
  'stages=[sync_cameras,detect,calibrate]'
```

Alternatively pass `--config FILE`. The packaged processing configs include
rig-specific timing, camera-to-EE transforms, tag/intrinsic files and optional
visualization/replay-buffer stages. Check these against the actual rig before
processing. Keep its gripper calibration episode; do not reuse another rig's
calibration silently. `pipeline.process()` dispatches to shared processing
stages; GoPro and OAK wrappers do not duplicate that implementation.

The classic session/replay-buffer route is a separate compatibility workflow,
not a prerequisite for SPOT. See [processing notes](../../demonstration_processing/README.md).

## Processed layout

```text
<root>/<scene>/<task>/recording_output_processed/<episode>/
    camera_1.mp4
    pose_interp/       # Zarr [T,4,4] EE poses
    gripper_widths/    # Zarr [T] widths, meters
    valid_indices/    # Zarr [T] boolean mask over raw RGB frames
```

The episode is a Zarr group with an adjacent video. Use `spvtp annotate` to
label first-frame object/target boxes, then `spvtp merge-annotations` to produce
the JSON consumed by `data.annotations_json`. Episode/video paths and scene/task
names must agree with that mapping.

## Frame alignment and sampling

New processing writes pose/width arrays on the raw-video frame axis; invalid
entries are masked. `common.episode_arrays.align_valid_frames` also accepts
legacy compact arrays whose length equals `valid_indices.sum()` and expands
them onto that axis. Other lengths raise an error instead of silently clipping
misaligned data.

The loader uses `flatnonzero(valid_indices)[::stride]` (`stride=3`, nominally
30 fps to 10 fps). With invalid frames, elapsed time between valid samples can
vary. Current steps require K preceding and H following valid, strided frames.
`sample_start_step` defaults to K; set it explicitly for paired ablations.
An unreadable sample raises an error rather than substituting a different one.

Splitting holds out tasks *within each scene*, then assigns all their episodes
to validation (`val_ratio=0.1`, `split_seed=42`). Training and validation therefore
share scene identities; this is not an unseen-scene generalization setting.
Keep the generated split manifest for comparisons and relocated evaluations.

## Frame cache

```bash
spvtp cache --help
spvtp stats --help
```

The cache stores resized RGB as mmap arrays, with four first-frame variants
(unmarked, both boxes, object-only, target-only) followed by strided current
frames. Set `data.frame_cache_root` when training. Cache metadata must match
stride, image size and frame count; regenerate after changing these or the
processed alignment. Local scratch is optional and node-local: a path existing
on one machine need not exist on another.

The builder and loader share the same full/compact array alignment contract.
Cache metadata fingerprints the selected raw frame indices, so changed masks
(even with the same frame count) invalidate cached frames. For caches built
before this fix, rerun `spvtp cache` once: missing fingerprints and truncated
caches are rebuilt without needing `--overwrite`. See
[fix verification](gpu_validation.md#fix-verification).

Prompt interventions must change both coordinate inputs and rendered cues.
For changed boxes, the loader redraws the first-frame visual prompt instead of
reusing the original correctly marked cache image.

Before a long run, check video decoding, mask/pose/width alignment, calibration,
merged annotations, split disjointness and at least one batch from every scene.
