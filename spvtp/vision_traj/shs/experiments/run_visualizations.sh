#!/usr/bin/env bash
# Produce qualitative visualization artifacts for a trained run.

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -eu

cd "$(git rev-parse --show-toplevel)"

RUN_DIR="${RUN_DIR:?set RUN_DIR}"
CKPT="${CKPT:-${RUN_DIR}/ckpt_final.pt}"
SPLIT="${SPLIT:-val}"
N_EPISODES="${N_EPISODES:-500}"
EPISODE_INDICES="${EPISODE_INDICES:-}"
VIZ_SAMPLES="${VIZ_SAMPLES:-500}"

for scene in scene1 scene2 scene3; do
    SCENES="${scene}" \
    CKPT="${CKPT}" \
    SPLIT="${SPLIT}" \
    N_EPISODES="${N_EPISODES}" \
    EPISODE_INDICES="${EPISODE_INDICES}" \
    FIGURE_FORMAT=pdf \
    OUT_DIR="${RUN_DIR}/visualizations/${scene}/full_trajectory" \
    bash spvtp/vision_traj/shs/eval/stitched_vis.sh

    SCENES="${scene}" \
    CKPT="${CKPT}" \
    SPLIT="${SPLIT}" \
    N_EPISODES="${N_EPISODES}" \
    EPISODE_INDICES="${EPISODE_INDICES}" \
    VIZ_SAMPLES="${VIZ_SAMPLES}" \
    VIZ_SELECTION=episode_first \
    VIZ_FORMAT=pdf \
    VIZ_DIR="${RUN_DIR}/visualizations/${scene}/first_chunk" \
    bash spvtp/vision_traj/shs/eval/vis.sh
done
