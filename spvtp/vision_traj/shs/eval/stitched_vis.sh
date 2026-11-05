#!/usr/bin/env bash
# Stitched full-episode trajectory visualization on an eval split.
#
# Required:
#   CKPT     path to a *.pt checkpoint
# Optional:
#   SCENES   comma-separated scenes to evaluate on.
#   OUT_DIR  output directory for stitched PNGs + CSVs. Auto-derived from CKPT if unset.
#   SPLIT, N_EPISODES, EPISODE_INDICES, STITCH, MAX_SAMPLES_PER_EPISODE
set -eu

cd "$(git rev-parse --show-toplevel)"

if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

CKPT="${CKPT:?set CKPT to a *.pt checkpoint path}"
DATA_ROOT="${DATA_ROOT:-spvtp/EgoSPT}"
ANNOTATIONS="${ANNOTATIONS:-annotations_merged.json}"
SPLIT="${SPLIT:-val}"
N_EPISODES="${N_EPISODES:-3}"
EPISODE_INDICES="${EPISODE_INDICES:-}"
STITCH="${STITCH:-mean_overlap}"
MAX_SAMPLES_PER_EPISODE="${MAX_SAMPLES_PER_EPISODE:-0}"
FIGURE_FORMAT="${FIGURE_FORMAT:-png}"
SCENES="${SCENES:-}"

scene_tag=""
extra_args=()
if [[ -n "${SCENES}" ]]; then
    extra_args+=("--scenes" "${SCENES}")
    scene_tag="_${SCENES//,/+}"
fi
if [[ -n "${EPISODE_INDICES}" ]]; then
    extra_args+=("--episode-indices" "${EPISODE_INDICES}")
fi

ckpt_dir="$(dirname "${CKPT}")"
OUT_DIR="${OUT_DIR:-${ckpt_dir}/${SPLIT}_stitched_viz${scene_tag}}"

python -m spvtp.vision_traj.scripts.vis_stitched_traj \
    --ckpt "${CKPT}" \
    --data-root "${DATA_ROOT}" \
    --annotations "${ANNOTATIONS}" \
    --split "${SPLIT}" \
    --n-episodes "${N_EPISODES}" \
    --out-dir "${OUT_DIR}" \
    --stitch "${STITCH}" \
    --max-samples-per-episode "${MAX_SAMPLES_PER_EPISODE}" \
    --figure-format "${FIGURE_FORMAT}" \
    "${extra_args[@]}"
