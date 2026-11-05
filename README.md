<h1 align="center">
  Spatially Prompted Visual Trajectory Prediction<br />
  for Egocentric Manipulation
</h1>

<p align="center">
  <strong>WACV 2027</strong><br />
  <em>Specify the object. Specify the destination. Predict the motion.</em>
</p>

<p align="center">
  <a href="https://jackyfl.github.io/">Yifan Li</a><sup>1,*</sup> &nbsp;
  <a href="https://scholar.google.com/citations?user=AcPskncAAAAJ">Xinyu Zhou</a><sup>1,*</sup> &nbsp;
  <a href="https://gyhandy.github.io/">Yunhao Ge</a><sup>2</sup> &nbsp;
  <a href="https://www.egr.msu.edu/~xbtan/">Xiaobo Tan</a><sup>1</sup> &nbsp;
  <a href="https://www.egr.msu.edu/~yukong/">Yu Kong</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> Michigan State University &nbsp;·&nbsp; <sup>2</sup> NVIDIA Research<br />
  <sup>*</sup> Equal contribution
</p>

<p align="center">
  <a href="https://jackyfl.github.io/SP-VTP-project-page/static/spvtp-paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-18453B?style=for-the-badge" alt="Paper PDF" /></a>
  <a href="https://arxiv.org/abs/2605.20085"><img src="https://img.shields.io/badge/arXiv-2605.20085-B31B1B?style=for-the-badge" alt="arXiv 2605.20085" /></a>
  <a href="https://jackyfl.github.io/SP-VTP-project-page/"><img src="https://img.shields.io/badge/Project-Page-176B87?style=for-the-badge" alt="Project page" /></a>
  <a href="https://huggingface.co/datasets/JackYFL233/EgoSPT"><img src="https://img.shields.io/badge/Dataset-EgoSPT-D99A00?style=for-the-badge" alt="EgoSPT dataset" /></a>
  <a href="https://jackyfl.github.io/SP-VTP-project-page/#overview"><img src="https://img.shields.io/badge/Watch-Demo-6554A4?style=for-the-badge" alt="Watch demo" /></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> &nbsp;·&nbsp;
  <a href="#quickstart">Quickstart</a> &nbsp;·&nbsp;
  <a href="#training">Training</a> &nbsp;·&nbsp;
  <a href="#evaluation">Evaluation</a> &nbsp;·&nbsp;
  <a href="#documentation">Documentation</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>

> **TL;DR** — A spatial prompt on the first frame specifies **which object to pick** and **where to place it**. SPOT combines this fixed task specification with the current egocentric image and recent motion to predict future 3D end-effector (EE) trajectories.

<p align="center">
  <a href="https://jackyfl.github.io/SP-VTP-project-page/#overview">
    <img src="assets/overview.png" width="100%" alt="Spatial object–target prompts resolve ambiguous goals in cluttered egocentric scenes." />
  </a>
</p>
<p align="center"><em>From ambiguous language to explicit spatial goals.</em> &nbsp; <a href="https://jackyfl.github.io/SP-VTP-project-page/#overview">▶ Watch the demo</a></p>

## Overview

This repository brings together the task, dataset, and policy introduced in our paper:

| Component | What it provides |
| :--- | :--- |
| **SP-VTP** | Visual trajectory prediction conditioned on first-frame object–target points or boxes. |
| **EgoSPT** | **2,841 episodes · 3 scenes** of egocentric fork pick-and-place, with object/target annotations and recovered EE trajectories. |
| **SPOT** | A spatially prompted policy that predicts **16 future waypoints**, each containing translation, 6D rotation, and gripper width. |

The code supports **capture → processing → annotation → training → offline evaluation**, including adapters for GR00T N1.6, π0.5, SmolVLA, and OpenVLA-OFT. Dataset files, pretrained weights, and experiment artifacts are not bundled with the repository.

### How SPOT works

<p align="center">
  <img src="assets/spot_architecture.png" width="100%" alt="SPOT: task encoder for the prompted first frame, observation encoder for current RGB and action history, and a flow-matching trajectory generator." />
</p>

The **task encoder** combines rendered boxes and coordinate tokens into a fixed task representation. The **observation encoder** adds the current RGB frame and recent motion. A **flow-matching decoder** predicts relative EE waypoints without an explicit object tracker. Every waypoint is relative to the chunk origin—not an increment from the previous waypoint.

## Quickstart

### 1. Install

```bash
git clone git@github.com:JackYFL/SP-VTP.git
cd SP-VTP

uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

Run commands from the repository root and match PyTorch to your CUDA installation. The package and CLI are both named `spvtp`; `python -m spvtp --help` also works. After updating an editable installation, rerun `uv pip install -e .`.

### 2. Prepare data

Download [EgoSPT on Hugging Face](https://huggingface.co/datasets/JackYFL233/EgoSPT). Set `data.root` to your local dataset and `data.annotations_json` to its annotation file, as in the training command below.

<details>
<summary><strong>Collect and annotate your own demonstrations</strong></summary>

Install the optional data tools, then select the backend matching your hardware:

```bash
uv pip install -e '.[data,annotation]'

spvtp collect --backend iphone-gopro --data-dir recordings/session.zarr --fps 30
spvtp process --backend iphone-gopro --config /path/to/session.yaml
spvtp annotate --video /path/to/camera_1.mp4
spvtp merge-annotations --root /path/to/EgoSPT --output annotations_merged.json
```

Backends: `iphone-gopro`, `iphone-oak`, and `iphone`. Capture keys: `s` start, `w` save, `a` discard/restart, `q` exit. Processing requires the selected rig's calibration. Install `.[iphone]` on the Record3D capture machine; OAK requires DepthAI 2.x. Hardware SDKs are not imported by ordinary SPOT training or evaluation.

See [data collection, processing, and alignment](spvtp/vision_traj/docs/dataset_analyze.md).

</details>

<a id="train"></a>

## Training

Start with the reference SPOT configuration:

```bash
spvtp train \
  data.root=/path/to/EgoSPT \
  data.annotations_json=/path/to/annotations.json \
  train.output_dir=runs/spot
```

**Defaults:** frozen DINOv2-Base · combined visual/coordinate box prompts · flow matching · `D=768` · `H=16` · `K=4` · learning rate `2e-4` · 30 epochs.

The default single-GPU effective batch size is **128** (`16 × 8` gradient accumulation). `data.split_seed=42` controls the split independently of `train.seed`. Use `--config FILE` or `key=value` overrides; `train.max_steps` limits a smoke run.

<details>
<summary><strong>Two-GPU training with the same effective batch size</strong></summary>

```bash
accelerate launch --num_processes 2 --multi_gpu -m spvtp.vision_traj.scripts.train \
  data.root=/path/to/EgoSPT \
  data.annotations_json=/path/to/annotations.json \
  train.grad_accum_steps=4 \
  train.output_dir=runs/spot
```

`16 samples × 4 accumulation steps × 2 processes = 128`. Matching the effective batch size does not guarantee bit-identical training across GPU counts.

</details>

See [training and checkpoints](spvtp/vision_traj/docs/training.md), [model and prompt definitions](spvtp/vision_traj/docs/model_arch.md), and the [default configuration](spvtp/vision_traj/configs/config.yaml).

## Evaluation

### Independent-window prediction

```bash
spvtp eval --ckpt runs/spot/ckpt_final.pt \
  --output-dir runs/spot/eval \
  --batch-size 32 --save-step-metrics --viz-samples 12
```

The evaluator loads the checkpoint's configuration and adjacent split manifest, then evaluates the full validation split with overall and per-scene outputs. History comes from the recording, not previous predictions. To relocate data, use `--data-root`, `--annotations`, and `--frame-cache-root`.

Console distances use cm; stored distances remain in meters with unit metadata. See [metrics and output files](spvtp/vision_traj/docs/evaluation.md).

### Sequential predicted-state feedback

```bash
spvtp rollout --ckpt runs/spot/ckpt_final.pt \
  --split-manifest runs/spot/split_manifest.json \
  --output-dir runs/spot/rollout
```

This compares recorded history, predicted history with recorded pose anchors, and cumulative predicted poses. **RGB remains recorded replay:** predictions do not change later images or object states. Episode-end FDE differs from fixed-16-step chunk FDE. Neither evaluation measures closed-loop robot task success. See the [rollout protocol](spvtp/vision_traj/docs/rollout.md).

### VLA adapters

GR00T N1.6, π0.5, SmolVLA, and OpenVLA-OFT adapters use backend-specific dependencies and fine-tuning settings. External VLA runtimes are optional for ordinary SPOT use.

```bash
spvtp vla --help
spvtp vla smolvla --help
```

See the [experiment notes](spvtp/vision_traj/docs/experiment.md) for input modalities and comparison settings.

## Documentation

| I want to… | Start here |
| :--- | :--- |
| Collect, process, or inspect data | [Data guide](spvtp/vision_traj/docs/dataset_analyze.md) |
| Understand SPOT and its prompts | [Model architecture](spvtp/vision_traj/docs/model_arch.md) |
| Train, resume, or tune GPU usage | [Training guide](spvtp/vision_traj/docs/training.md) |
| Understand metrics and prediction feedback | [Evaluation](spvtp/vision_traj/docs/evaluation.md) · [Rollout](spvtp/vision_traj/docs/rollout.md) |
| Reproduce controls or extend the code | [Experiments](spvtp/vision_traj/docs/experiment.md) · [Development](spvtp/vision_traj/docs/development.md) |

<details>
<summary><strong>Development checks</strong></summary>

```bash
uv pip install -e '.[dev]'
OMP_NUM_THREADS=2 python -m pytest -q
```

All automated tests live in the root `tests/` directory. Pytest discovers both
the unittest cases and visualization tests; runtime packages contain no tests.

`spvtp --help` lists commands; `spvtp COMMAND --help` shows their options. The root `pyproject.toml` is the single dependency specification. Import implementations from their canonical subpackages (`data`, `models`, `evaluation`, `analysis`, `baselines`, and `scripts`); legacy flat-module entry points have been removed. See the [documentation index](spvtp/vision_traj/docs/README.md) and [GPU validation notes](spvtp/vision_traj/docs/gpu_validation.md).

</details>

## Citation

If this work is useful for your research, please cite:

```bibtex
@inproceedings{li2027spvtp,
  title={Spatially Prompted Visual Trajectory Prediction for Egocentric Manipulation},
  author={Li, Yifan and Zhou, Xinyu and Ge, Yunhao and Tan, Xiaobo and Kong, Yu},
  booktitle={Winter Conference on Applications of Computer Vision (WACV)},
  year={2027}
}
```
