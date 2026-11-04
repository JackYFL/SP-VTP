#!/usr/bin/env bash
# 1-epoch smoke test on multi-GPU. Finishes in a few minutes; used to verify
# DDP wiring, memory footprint, and data pipeline before a real run.
#
# Restrict to one or more scenes via the SCENES env var:
#   SCENES=scene1            bash spvtp/vision_traj/shs/train/smoke.sh
#   SCENES=scene1,scene2     bash spvtp/vision_traj/shs/train/smoke.sh
# Add an extra run-name suffix via EXTRA_NAME:
#   EXTRA_NAME=cross_attn SCENES=scene1 bash spvtp/vision_traj/shs/train/smoke.sh
set -eu

cd "$(git rev-parse --show-toplevel)"

# Activate the uv venv if present and the user hasn't done so already
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
echo "[launch] VIRTUAL_ENV=${VIRTUAL_ENV:-<none>}  accelerate=$(which accelerate)"
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

RUN_NAME="${RUN_NAME:-smoke_${HEAD_TYPE}${scene_tag}${extra_name_tag}}"

accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 \
    -m spvtp.vision_traj.scripts.train \
    train.batch_size=8 \
    train.mixed_precision=bf16 \
    train.num_epochs=1 \
    train.val_every_epochs=1 \
    train.save_every_epochs=1 \
    train.log_every=20 \
    train.wandb.enabled=false \
    data.num_workers=4 \
    "policy.head_type=${HEAD_TYPE}" \
    "policy.flow_time_sampling=${FLOW_TIME_SAMPLING}" \
    "policy.flow_beta_alpha=${FLOW_BETA_ALPHA}" \
    "policy.flow_beta_beta=${FLOW_BETA_BETA}" \
    "train.output_dir=runs/${RUN_NAME}" \
    "${extra_overrides[@]}"
