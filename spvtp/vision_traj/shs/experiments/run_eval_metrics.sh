#!/usr/bin/env sh
# Compute offline metrics for a trained run without producing eval images.
# RUN_DIR=runs/vision_traj_experiments/baseline_bbox_all_scenes \
#   bash spvtp/vision_traj/shs/experiments/run_eval_metrics.sh

set -eu

cd "$(git rev-parse --show-toplevel)"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    . .venv/bin/activate
fi

RUN_DIR="${RUN_DIR:?set RUN_DIR, e.g. runs/vision_traj_experiments/baseline_bbox_all_scenes}"
CKPT="${CKPT:-${RUN_DIR}/ckpt_final.pt}"
DATA_ROOT="${DATA_ROOT:-spvtp/EgoSPT}"
ANNOTATIONS="${ANNOTATIONS:-annotations_merged.json}"
SPLIT="${SPLIT:-val}"
N_EPISODES="${N_EPISODES:-0}"
OUT_ROOT="${OUT_ROOT:-${RUN_DIR}/eval_metrics}"

for scene in all scene1 scene2 scene3; do
    out_dir="${OUT_ROOT}/${scene}"
    if [ "${scene}" = "all" ]; then
        python -m spvtp.vision_traj.scripts.infer \
            --ckpt "${CKPT}" \
            --data-root "${DATA_ROOT}" \
            --annotations "${ANNOTATIONS}" \
            --split "${SPLIT}" \
            --n-episodes "${N_EPISODES}" \
            --viz-dir "${out_dir}" \
            --viz-samples 0
    else
        python -m spvtp.vision_traj.scripts.infer \
            --ckpt "${CKPT}" \
            --data-root "${DATA_ROOT}" \
            --annotations "${ANNOTATIONS}" \
            --split "${SPLIT}" \
            --n-episodes "${N_EPISODES}" \
            --viz-dir "${out_dir}" \
            --viz-samples 0 \
            --scenes "${scene}"
    fi
done
