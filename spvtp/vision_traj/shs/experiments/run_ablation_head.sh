#!/usr/bin/env bash
# Launch action-head ablations.

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -eu

cd "$(git rev-parse --show-toplevel)"

SCENES="${SCENES:-scene1,scene2,scene3}"
SCRIPT="spvtp/vision_traj/shs/experiments/launch_train_final.sh"
EXPERIMENT_GROUP=ablation_head EXPERIMENT_NAME=diffusion SCENES="${SCENES}" \
    bash "${SCRIPT}" data.prompt_form=vision_bbox_and_bbox policy.head_type=diffusion train.batch_size=32

# EXPERIMENT_GROUP=ablation_head EXPERIMENT_NAME=flow_matching SCENES="${SCENES}" \
#     bash "${SCRIPT}" data.prompt_form=point policy.head_type=flow_matching
