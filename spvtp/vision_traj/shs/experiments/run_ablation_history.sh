#!/usr/bin/env bash
# Launch history-horizon ablations.

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -eu

cd "$(git rev-parse --show-toplevel)"

SCENES="${SCENES:-scene1,scene2,scene3}"
SCRIPT="spvtp/vision_traj/shs/experiments/launch_train_final.sh"

for history in 0 8 12; do
    EXPERIMENT_GROUP=ablation_history EXPERIMENT_NAME="history_${history}" SCENES="${SCENES}" \
        bash "${SCRIPT}" data.prompt_form=vision_bbox_and_bbox "policy.history_horizon=${history}"
done
