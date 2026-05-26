#!/usr/bin/env bash
set -euo pipefail

LVLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FULL_SPLIT_DIR="/lby_data01/zhaozy/lby/ChangeRef_Clear/val_FINAL_CLEAN"
VIS_ROOT="${LVLM_ROOT}/vis_segmap"
SUBSET_ROOT="${VIS_ROOT}/subset_data"
SUBSET_SPLIT_DIR="${SUBSET_ROOT}/ChangeRef_Clear/val_FINAL_CLEAN"
SUBSET_JSON_DIR="${SUBSET_SPLIT_DIR}/referring_expression"
MANIFEST_PATH="${VIS_ROOT}/sample_manifest_150.txt"
SAMPLE_COUNT="${SAMPLE_COUNT:-150}"

mkdir -p "${VIS_ROOT}"
mkdir -p "${SUBSET_SPLIT_DIR}"
mkdir -p "${SUBSET_JSON_DIR}"

if [ ! -f "${MANIFEST_PATH}" ]; then
  find "${FULL_SPLIT_DIR}/referring_expression" -maxdepth 1 -type f -name '*.json' -printf '%f\n' \
    | shuf -n "${SAMPLE_COUNT}" \
    | sort > "${MANIFEST_PATH}"
fi

ln -sfn "${FULL_SPLIT_DIR}/A" "${SUBSET_SPLIT_DIR}/A"
ln -sfn "${FULL_SPLIT_DIR}/B" "${SUBSET_SPLIT_DIR}/B"
ln -sfn "${FULL_SPLIT_DIR}/masks" "${SUBSET_SPLIT_DIR}/masks"

find "${SUBSET_JSON_DIR}" -mindepth 1 -maxdepth 1 -type l -delete
while IFS= read -r json_name; do
  [ -n "${json_name}" ] || continue
  ln -sfn "${FULL_SPLIT_DIR}/referring_expression/${json_name}" "${SUBSET_JSON_DIR}/${json_name}"
done < "${MANIFEST_PATH}"

echo "${SUBSET_SPLIT_DIR}"
