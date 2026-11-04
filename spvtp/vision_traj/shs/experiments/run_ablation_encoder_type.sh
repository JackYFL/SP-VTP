#!/usr/bin/env bash
# Launch supported encoder-family ablations and list pending adapters.

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -eu

cd "$(git rev-parse --show-toplevel)"

SCENES="${SCENES:-scene1,scene2,scene3}"
SCRIPT="spvtp/vision_traj/shs/experiments/launch_train_final.sh"

# EXPERIMENT_GROUP=ablation_vision_encoder_type EXPERIMENT_NAME=dinov2_base SCENES="${SCENES}" \
#     bash "${SCRIPT}" \
#     data.prompt_form=vision_bbox_and_bbox \
#     policy.vision_encoder.type=dinov2 \
#     policy.vision_encoder.model_name=vit_base_patch14_dinov2 \
#     policy.embed_dim=768 \
#     policy.fusion_heads=12 \
#     policy.head_heads=12

# timm's samvit_base_patch16.sa1b pretrained checkpoint is trained at 1024px
# and does not load into the 224px model used by this pipeline.
# EXPERIMENT_GROUP=ablation_vision_encoder_type EXPERIMENT_NAME=sam_base SCENES="${SCENES}" \
#     bash "${SCRIPT}" \
#     data.prompt_form=vision_bbox_and_bbox \
#     policy.vision_encoder.type=sam \
#     policy.vision_encoder.model_name=samvit_base_patch16.sa1b \
#     policy.vision_encoder.pretrained=false \
#     policy.embed_dim=768 \
#     policy.fusion_heads=12 \
#     policy.head_heads=12

# EXPERIMENT_GROUP=ablation_vision_encoder_type EXPERIMENT_NAME=pe_base SCENES="${SCENES}" \
#     bash "${SCRIPT}" \
#     data.prompt_form=vision_bbox_and_bbox \
#     policy.vision_encoder.type=pe \
#     policy.vision_encoder.model_name=vit_pe_core_base_patch16_224.fb \
#     policy.embed_dim=768 \
#     policy.fusion_heads=12 \
#     policy.head_heads=12

# EXPERIMENT_GROUP=ablation_vision_encoder_type EXPERIMENT_NAME=siglip_base SCENES="${SCENES}" \
#     bash "${SCRIPT}" \
#     data.prompt_form=vision_bbox_and_bbox \
#     policy.vision_encoder.type=siglip \
#     policy.vision_encoder.model_name=vit_base_patch16_siglip_224.v2_webli \
#     policy.embed_dim=768 \
#     policy.fusion_heads=12 \
#     policy.head_heads=12

EXPERIMENT_GROUP=ablation_vision_encoder_type EXPERIMENT_NAME=eva2_base SCENES="${SCENES}" \
    bash "${SCRIPT}" \
    data.prompt_form=vision_bbox_and_bbox \
    policy.vision_encoder.type=eva2 \
    policy.vision_encoder.model_name=eva02_base_patch14_224.mim_in22k \
    policy.embed_dim=768 \
    policy.fusion_heads=12 \
    policy.head_heads=12
