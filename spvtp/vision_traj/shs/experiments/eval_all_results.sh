#!/usr/bin/env sh
# Evaluate every completed experiment run and refresh experiment_results.xlsx.
#
# Skips runs without ckpt_final.pt. By default, also skips runs that already
# have eval_metrics/all/summary.json; set FORCE_EVAL=true to recompute.
set -eu

cd "$(git rev-parse --show-toplevel)"

ROOT="${ROOT:-runs/vision_traj_experiments}"
FORCE_EVAL="${FORCE_EVAL:-false}"
EVAL_SCRIPT="spvtp/vision_traj/shs/experiments/run_eval_metrics.sh"
COLLECT_SCRIPT="spvtp/vision_traj/shs/experiments/collect_results.sh"

if [ ! -d "${ROOT}" ]; then
    echo "[eval_all_results] root does not exist: ${ROOT}"
    exit 0
fi

evaluated=0
skipped=0

for run_dir in "${ROOT}"/*; do
    if [ ! -d "${run_dir}" ]; then
        continue
    fi

    run_name="$(basename "${run_dir}")"
    ckpt="${run_dir}/ckpt_final.pt"
    summary="${run_dir}/eval_metrics/all/summary.json"

    if [ ! -f "${ckpt}" ]; then
        echo "[eval_all_results] skip ${run_name}: missing ckpt_final.pt"
        skipped=$((skipped + 1))
        continue
    fi

    if [ "${FORCE_EVAL}" != "true" ] && [ -f "${summary}" ]; then
        echo "[eval_all_results] skip ${run_name}: eval summary already exists"
        skipped=$((skipped + 1))
        continue
    fi

    echo "[eval_all_results] evaluating ${run_name}"
    RUN_DIR="${run_dir}" sh "${EVAL_SCRIPT}"
    evaluated=$((evaluated + 1))
done

echo "[eval_all_results] evaluated=${evaluated} skipped=${skipped}"
echo "[eval_all_results] collecting Excel results"
ROOT="${ROOT}" sh "${COLLECT_SCRIPT}"
