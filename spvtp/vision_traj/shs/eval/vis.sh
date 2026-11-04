#!/usr/bin/env bash
# Open-loop evaluation + trajectory visualizations on the val split.
#
# Required:
#   CKPT     path to a *.pt checkpoint
# Optional:
#   SCENES   comma-separated scenes to evaluate on (e.g. "scene1" or "scene1,scene2").
#            Defaults to whatever scenes the checkpoint was trained on.
#   VIZ_DIR  output directory for figures + metrics. Auto-derived from CKPT if unset.
#   N_EPISODES, EPISODE_INDICES, VIZ_SAMPLES, VIZ_SORT, VIZ_SELECTION, SPLIT — passed straight through.
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
N_EPISODES="${N_EPISODES:-10}"
EPISODE_INDICES="${EPISODE_INDICES:-}"
VIZ_SAMPLES="${VIZ_SAMPLES:-12}"
VIZ_SORT="${VIZ_SORT:-total}"
VIZ_SELECTION="${VIZ_SELECTION:-ranked}"
VIZ_FORMAT="${VIZ_FORMAT:-png}"
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
VIZ_DIR="${VIZ_DIR:-${ckpt_dir}/${SPLIT}_viz${scene_tag}}"

python -m spvtp.vision_traj.scripts.infer \
    --ckpt "${CKPT}" \
    --data-root "${DATA_ROOT}" \
    --annotations "${ANNOTATIONS}" \
    --split "${SPLIT}" \
    --n-episodes "${N_EPISODES}" \
    --viz-dir "${VIZ_DIR}" \
    --viz-samples "${VIZ_SAMPLES}" \
    --viz-sort "${VIZ_SORT}" \
    --viz-selection "${VIZ_SELECTION}" \
    --viz-format "${VIZ_FORMAT}" \
    "${extra_args[@]}"
