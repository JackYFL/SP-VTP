# Documentation

Start with the [project quickstart](../../../README.md). Commands below assume
the project is installed and the working directory is its root.

- [Data](dataset_analyze.md): collection, processing, annotations and frame alignment.
- [Model](model_arch.md): task/observation encoders, prompt variants and actions.
- [Training](training.md): configuration, effective batch size and checkpoints.
- [Evaluation](evaluation.md): offline protocol, formulas, units and outputs.
- [Sequential rollout](rollout.md): predicted history, cumulative poses and replay limitations.
- [Experiments](experiment.md): comparison controls and historical launch scripts.
- [Development](development.md): module ownership, reusable functions and tests.
- [GPU validation](gpu_validation.md): migration checks and comparison with old logs.

Defaults are defined only in [`configs/config.yaml`](../configs/config.yaml).
Use the unified `spvtp` CLI for new runs; old flat imports remain aliases.
Historical experiment scripts describe their original settings, not necessarily
the packaged defaults. No dataset, pretrained weights or experiment results
are bundled with this repository.
