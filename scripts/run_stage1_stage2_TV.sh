#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="/lby_data01/zhaozy/lby/ChangeRef_Clear"

STAGE1_DATA_PATH="${1:-${DATA_ROOT}/MTRefSeg_pretrain_TV}"
BASE_MODEL="${2:-${PROJECT_ROOT}/pretrained_weights/phi-1_5_dev}"
VISION_TOWER="${3:-${PROJECT_ROOT}/pretrained_weights/model_final_9d7f02.pkl}"
STAGE1_OUTPUT_DIR="${4:-${PROJECT_ROOT}/checkpoint/stage1_visual_change_pretrain_TV}"
STAGE2_OUTPUT_DIR="${5:-${PROJECT_ROOT}/checkpoint/stage2_train_final_val}"
GPU_IDS="${6:-0,1,2,3}"

bash "${SCRIPT_DIR}/run_stage1_visual_pretrain_TV.sh" \
    "${STAGE1_DATA_PATH}" \
    "${BASE_MODEL}" \
    "${VISION_TOWER}" \
    "${STAGE1_OUTPUT_DIR}" \
    "${GPU_IDS}"

bash "${SCRIPT_DIR}/run_stage2_full_finetune_TV.sh" \
    "${BASE_MODEL}" \
    "${VISION_TOWER}" \
    "${STAGE1_OUTPUT_DIR}" \
    "${STAGE2_OUTPUT_DIR}" \
    "${GPU_IDS}"
