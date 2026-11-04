#!/usr/bin/env bash
# Single-GPU training — use for quick debugging or when only one GPU is free.
#
# Restrict to one or more scenes via the SCENES env var (comma-separated):
#   SCENES=scene1            bash spvtp/vision_traj/shs/train/single_gpu.sh
#   SCENES=scene1,scene2     bash spvtp/vision_traj/shs/train/single_gpu.sh
# Add an extra run-name suffix via EXTRA_NAME:
#   EXTRA_NAME=cross_attn SCENES=scene1 bash spvtp/vision_traj/shs/train/single_gpu.sh
# Leave SCENES unset to train on all scenes.
set -eu

cd "$(git rev-parse --show-toplevel)"

if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
echo "[launch] VIRTUAL_ENV=${VIRTUAL_ENV:-<none>}  python=$(which python)"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
HEAD_TYPE="${HEAD_TYPE:-diffusion}"
FLOW_TIME_SAMPLING="${FLOW_TIME_SAMPLING:-beta}"
FLOW_BETA_ALPHA="${FLOW_BETA_ALPHA:-1.5}"
FLOW_BETA_BETA="${FLOW_BETA_BETA:-1.0}"
SCENES="${SCENES:-}"
EXTRA_NAME="${EXTRA_NAME:-}"

extra_overrides=()
scene_tag=""
extra_name_tag=""
if [[ -n "${SCENES}" ]]; then
    extra_overrides+=("data.scenes=[${SCENES}]")
    scene_tag="_${SCENES//,/+}"
fi
if [[ -n "${EXTRA_NAME}" ]]; then
    extra_name_tag="_${EXTRA_NAME}"
fi

RUN_NAME="${RUN_NAME:-single_gpu_${HEAD_TYPE}${scene_tag}${extra_name_tag}}"

python -m spvtp.vision_traj.scripts.train \
    train.batch_size=16 \
    train.mixed_precision=bf16 \
    data.num_workers=4 \
    "policy.head_type=${HEAD_TYPE}" \
    "policy.flow_time_sampling=${FLOW_TIME_SAMPLING}" \
    "policy.flow_beta_alpha=${FLOW_BETA_ALPHA}" \
    "policy.flow_beta_beta=${FLOW_BETA_BETA}" \
    train.wandb.enabled=true \
    train.wandb.project=spvtp_vision_traj \
    "train.wandb.run_name=${RUN_NAME}" \
    "train.output_dir=runs/${RUN_NAME}" \
    "${extra_overrides[@]}"
