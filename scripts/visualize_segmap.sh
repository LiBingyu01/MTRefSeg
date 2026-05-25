#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LVLM_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DATA_ROOT="$(bash "${LVLM_ROOT}/prepare_vis_subset.sh")"
OUTPUT_ROOT="${LVLM_ROOT}/vis_segmap/MTRefSeg-R1"
mkdir -p "${OUTPUT_ROOT}"

source /lby_data01/zhaozy/lby/miniconda3/etc/profile.d/conda.sh
conda activate segr1

(
  cd "${PROJECT_ROOT}"
  export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "${SCRIPT_DIR}/batch_chat_visualize.py" \
    --model_path "${PROJECT_ROOT}/checkpoint/stage2_train_final_val" \
    --mask_config "${PROJECT_ROOT}/segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml" \
    --version "llava_phi" \
    --device "cuda" \
    --json_dir "${DATA_ROOT}/referring_expression" \
    --image_dir_A "${DATA_ROOT}/A" \
    --image_dir_B "${DATA_ROOT}/B" \
    --mask_root "${DATA_ROOT}/masks" \
    --output_dir "${OUTPUT_ROOT}" \
    --save_canvas \
    --save_gt
)
