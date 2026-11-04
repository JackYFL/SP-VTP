#!/usr/bin/env bash
# Build runs/vision_traj_experiments/experiment_results.xlsx from eval summaries.

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -eu

cd "$(git rev-parse --show-toplevel)"

if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

ROOT="${ROOT:-runs/vision_traj_experiments}"
OUT="${OUT:-${ROOT}/experiment_results.xlsx}"

python spvtp/vision_traj/shs/experiments/collect_results_xlsx.py \
    --root "${ROOT}" \
    --out "${OUT}"
