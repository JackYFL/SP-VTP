#!/usr/bin/env bash
# Launch DINOv2 small/base/large encoder-size ablations.

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -eu

cd "$(git rev-parse --show-toplevel)"

SCENES="${SCENES:-scene1,scene2,scene3}"
SCRIPT="spvtp/vision_traj/shs/experiments/launch_train_final.sh"

EXPERIMENT_GROUP=ablation_vision_encoder_size EXPERIMENT_NAME=dinov2_small SCENES="${SCENES}" \
    bash "${SCRIPT}" \
    data.prompt_form=vision_bbox_and_bbox \
    policy.vision_encoder.model_name=vit_small_patch14_dinov2 \
    policy.embed_dim=384 \
    policy.fusion_heads=6 \
    train.batch_size=32 \
    policy.head_heads=6

# EXPERIMENT_GROUP=ablation_vision_encoder_size EXPERIMENT_NAME=dinov2_base SCENES="${SCENES}" \
#     bash "${SCRIPT}" \
#     data.prompt_form=vision_bbox_and_bbox \
#     policy.vision_encoder.model_name=vit_base_patch14_dinov2 \
#     policy.embed_dim=768 \
#     policy.fusion_heads=12 \
#     policy.head_heads=12

EXPERIMENT_GROUP=ablation_vision_encoder_size EXPERIMENT_NAME=dinov2_large SCENES="${SCENES}" \
    bash "${SCRIPT}" \
    data.prompt_form=vision_bbox_and_bbox \
    policy.vision_encoder.model_name=vit_large_patch14_dinov2 \
    policy.embed_dim=1024 \
    policy.fusion_heads=16 \
    train.batch_size=32 \
    policy.head_heads=16
