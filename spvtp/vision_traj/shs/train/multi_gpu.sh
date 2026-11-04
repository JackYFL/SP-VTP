#!/usr/bin/env bash
# Main single-node training launcher.
# Usage:
#   bash spvtp/vision_traj/shs/train/multi_gpu.sh
#   NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 bash spvtp/vision_traj/shs/train/multi_gpu.sh
#   NUM_GPUS=4 bash spvtp/vision_traj/shs/train/multi_gpu.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 bash spvtp/vision_traj/shs/train/multi_gpu.sh
#
# Restrict to one or more scenes via the SCENES env var (comma-separated):
#   SCENES=scene1                    bash spvtp/vision_traj/shs/train/multi_gpu.sh
#   SCENES=scene1,scene2,scene3 bash spvtp/vision_traj/shs/train/multi_gpu.sh
# Add an extra run-name suffix via EXTRA_NAME:
#   EXTRA_NAME=cross_attn SCENES=cene1,scene2,scene3 bash spvtp/vision_traj/shs/train/multi_gpu.sh
# Leave SCENES unset to train on all scenes.

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -eu

cd "$(git rev-parse --show-toplevel)"

if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

NUM_GPUS="${NUM_GPUS:-8}"
HEAD_TYPE="${HEAD_TYPE:-flow_matching}"
FLOW_TIME_SAMPLING="${FLOW_TIME_SAMPLING:-beta}"
FLOW_BETA_ALPHA="${FLOW_BETA_ALPHA:-1.5}"
FLOW_BETA_BETA="${FLOW_BETA_BETA:-1.0}"
SCENES="${SCENES:-}"
EXTRA_NAME="${EXTRA_NAME:-}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

echo "[launch] VIRTUAL_ENV=${VIRTUAL_ENV:-<none>}  accelerate=$(which accelerate)"
echo "[launch] NUM_GPUS=${NUM_GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>} MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT}"
echo "[launch] NCCL_DEBUG=${NCCL_DEBUG} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"

scene_tag=""
extra_name_tag=""
extra_overrides=()
if [[ -n "${SCENES}" ]]; then
    extra_overrides+=("data.scenes=[${SCENES}]")
    scene_tag="_${SCENES//,/+}"
fi
if [[ -n "${EXTRA_NAME}" ]]; then
    extra_name_tag="_${EXTRA_NAME}"
fi

RUN_NAME="${RUN_NAME:-${HEAD_TYPE}${scene_tag}${extra_name_tag}_$(date +%Y%m%d_%H%M%S)}"

# global batch = train.batch_size * NUM_GPUS
# scale LR ~ sqrt(global_batch / base_global_batch)
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

accelerate launch \
    "${launch_args[@]}" \
    -m spvtp.vision_traj.scripts.train \
    train.batch_size=16 \
    train.mixed_precision=bf16 \
    train.lr=2.0e-4 \
    train.num_epochs=30 \
    train.log_every=50 \
    train.val_every_epochs=2 \
    train.save_every_epochs=10 \
    data.num_workers=4 \
    policy.head_type=${HEAD_TYPE} \
    policy.flow_time_sampling=${FLOW_TIME_SAMPLING} \
    policy.flow_beta_alpha=${FLOW_BETA_ALPHA} \
    policy.flow_beta_beta=${FLOW_BETA_BETA} \
    train.wandb.enabled=true \
    train.wandb.project=spvtp_vision_traj \
    "train.wandb.run_name=${RUN_NAME}" \
    "train.output_dir=runs/${RUN_NAME}" \
    "${extra_overrides[@]}"
