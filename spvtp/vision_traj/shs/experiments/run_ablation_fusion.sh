#!/usr/bin/env bash
# Launch task-fusion ablations.

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -eu

cd "$(git rev-parse --show-toplevel)"

SCENES="${SCENES:-scene1,scene2,scene3}"
SCRIPT="spvtp/vision_traj/shs/experiments/launch_train_final.sh"

# EXPERIMENT_GROUP=ablation_fusion EXPERIMENT_NAME=cross_attention SCENES="${SCENES}" \
#     bash "${SCRIPT}" data.prompt_form=point policy.task_fusion_type=cross_attention

EXPERIMENT_GROUP=ablation_fusion EXPERIMENT_NAME=self_attention SCENES="${SCENES}" \
    bash "${SCRIPT}" data.prompt_form=vision_bbox_and_bbox policy.task_fusion_type=self_attention
