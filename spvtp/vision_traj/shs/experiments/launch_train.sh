#!/usr/bin/env bash
# Generic experiment launcher for baseline/ablation runs.
# Pass extra OmegaConf overrides as positional arguments.
set -eu

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

cd "$(git rev-parse --show-toplevel)"

if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

NUM_GPUS="${NUM_GPUS:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"
SCENES="${SCENES:-scene1,scene2,scene3}"
EXPERIMENT_GROUP="${EXPERIMENT_GROUP:?set EXPERIMENT_GROUP}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?set EXPERIMENT_NAME}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT_GROUP}_${EXPERIMENT_NAME}_all_scenes}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/vision_traj_experiments/${RUN_NAME}}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-spvtp_vision_traj}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

mkdir -p "${OUTPUT_DIR}"

python - "${OUTPUT_DIR}" "${EXPERIMENT_GROUP}" "${EXPERIMENT_NAME}" "${RUN_NAME}" "${SCENES}" "$@" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
meta = {
    "experiment_group": sys.argv[2],
    "experiment_name": sys.argv[3],
    "run_name": sys.argv[4],
    "scenes": sys.argv[5],
    "overrides": sys.argv[6:],
}
(out / "experiment_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
PY

launch_args=(
    --num_processes "${NUM_GPUS}"
    --num_machines 1
    --machine_rank 0
    --main_process_port "${MAIN_PROCESS_PORT}"
    --mixed_precision bf16
)
if (( NUM_GPUS > 1 )); then
    launch_args=(--multi_gpu "${launch_args[@]}")
fi

echo "[experiment] group=${EXPERIMENT_GROUP} name=${EXPERIMENT_NAME} run=${RUN_NAME}"
echo "[experiment] scenes=${SCENES} output=${OUTPUT_DIR}"
echo "[experiment] overrides=$*"

accelerate launch \
    "${launch_args[@]}" \
    -m spvtp.vision_traj.scripts.train \
    train.batch_size=16 \
    train.mixed_precision=bf16 \
    train.lr=2.0e-4 \
    train.num_epochs="${NUM_EPOCHS:-30}" \
    train.log_every=50 \
    train.val_every_epochs=2 \
    train.save_every_epochs=10 \
    data.num_workers=4 \
    "data.scenes=[${SCENES}]" \
    train.wandb.enabled="${WANDB_ENABLED}" \
    train.wandb.project="${WANDB_PROJECT}" \
    "train.wandb.run_name=${RUN_NAME}" \
    "train.output_dir=${OUTPUT_DIR}" \
    "$@"
