#!/usr/bin/env bash
# Launch supported prompt baseline experiments on scene1+scene2+scene3 together.

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -eu

cd "$(git rev-parse --show-toplevel)"

SCENES="${SCENES:-scene1,scene2,scene3}"
SCRIPT="spvtp/vision_traj/shs/experiments/launch_train_final.sh"

EXPERIMENT_GROUP=baseline EXPERIMENT_NAME=vision_bbox SCENES="${SCENES}" \
    bash "${SCRIPT}" data.prompt_form=vision_bbox

EXPERIMENT_GROUP=baseline EXPERIMENT_NAME=bbox SCENES="${SCENES}" \
    bash "${SCRIPT}" data.prompt_form=bbox

EXPERIMENT_GROUP=baseline EXPERIMENT_NAME=point SCENES="${SCENES}" \
    bash "${SCRIPT}" data.prompt_form=point

EXPERIMENT_GROUP=baseline EXPERIMENT_NAME=no_prompt SCENES="${SCENES}" \
    bash "${SCRIPT}" data.prompt_form=none

# EXPERIMENT_GROUP=baseline EXPERIMENT_NAME=vision_bbox_and_bbox SCENES="${SCENES}" \
#     bash "${SCRIPT}" data.prompt_form=vision_bbox_and_bbox
