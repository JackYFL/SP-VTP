#!/usr/bin/env bash
# Launch DINOv2-Base tuning ablations.

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -eu

cd "$(git rev-parse --show-toplevel)"

SCENES="${SCENES:-scene1,scene2,scene3}"
SCRIPT="spvtp/vision_traj/shs/experiments/launch_train_final.sh"
GROUP="ablation_dinov2_base_tuning"

COMMON_OVERRIDES=(
    data.prompt_form=vision_bbox_and_bbox
    policy.vision_encoder.type=dinov2
    policy.vision_encoder.model_name=vit_base_patch14_dinov2
    policy.vision_encoder.pretrained=true
    policy.embed_dim=768
    policy.fusion_heads=12
    policy.head_heads=12
    policy.head_type=flow_matching
    policy.task_fusion_type=cross_attention
)

# EXPERIMENT_GROUP="${GROUP}" EXPERIMENT_NAME=frozen SCENES="${SCENES}" \
#     bash "${SCRIPT}" \
#     "${COMMON_OVERRIDES[@]}" \
#     policy.vision_encoder.freeze=true \
#     train.lr=1.0e-4

EXPERIMENT_GROUP="${GROUP}" EXPERIMENT_NAME=unfrozen SCENES="${SCENES}" \
    bash "${SCRIPT}" \
    "${COMMON_OVERRIDES[@]}" \
    policy.vision_encoder.freeze=false \
    train.batch_size=8
