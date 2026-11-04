# Demonstration processing

Install processing dependencies from the project root: `uv pip install -e '.[data]'`.
For the SPOT dataset, use the shared entry point:

```bash
spvtp process --backend iphone-gopro --config /path/to/session.yaml
```

`--backend` selects `iphone-gopro`, `iphone-oak`, `iphone` or `gopro`.
Dotted overrides follow the config, e.g. `input_dir=/path/to/raw overwrite=false`.
Check camera timing, EE transforms, intrinsics, ArUco and gripper calibration
against the actual capture rig before processing.

`pipeline.process()` is the reusable API. iPhone-GoPro and iPhone-OAK share
`iphone_pipeline.py`: synchronization, detection, width calibration, optional
visualization/replay-buffer stages. See [data and alignment](../vision_traj/docs/dataset_analyze.md)
for the format consumed directly by SPOT.

The older session-building route remains available via module entry points
`process_demos_iphone_gopro`, `create_session_iphone` and
`build_umi_dataset_iphone` under `spvtp.demonstration_processing`.
It uses its own Hydra configs and is not an additional mandatory stage for
already processed EgoSPT episodes. Inspect module `--help` and `config/` before
using that compatibility workflow.
