#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_ROOT="/lby_data01/zhaozy/lby/ChangeRef_Clear"
PRETRAIN_DATA="${1:-${DATA_ROOT}/MTRefSeg_pretrain_TV}"
BASE_MODEL="${2:-${PROJECT_ROOT}/pretrained_weights/phi-1_5_dev}"
VISION_TOWER="${3:-${PROJECT_ROOT}/pretrained_weights/model_final_9d7f02.pkl}"
OUTPUT_ROOT="${4:-${PROJECT_ROOT}/checkpoint/ablation/e2e_mllm}"
GPU_IDS="${5:-0,1,2,3}"

PRETRAIN_OUT="${OUTPUT_ROOT}/pretrain"
DOWNSTREAM_OUT="${OUTPUT_ROOT}/stage2"
E2E_INIT_PATH="${E2E_INIT_PATH:-${PRETRAIN_OUT}}"

mkdir -p "${OUTPUT_ROOT}" "${PROJECT_ROOT}/logs/ablation"

RUN_NAME="ablation/e2e_mllm_pretrain" \
TEMPORAL_FUSION_TYPE="${TEMPORAL_FUSION_TYPE:-change_aware}" \
bash "${SCRIPT_DIR}/run_e2e_mllm_pretrain.sh" \
    "${PRETRAIN_DATA}" \
    "${BASE_MODEL}" \
    "${VISION_TOWER}" \
    "${PRETRAIN_OUT}" \
    "${GPU_IDS}"

RUN_NAME="ablation/e2e_mllm_stage2" \
USE_STAGE1_PRETRAIN=0 \
FULL_MODEL_INIT_PATH="${E2E_INIT_PATH}" \
TEMPORAL_FUSION_TYPE="${TEMPORAL_FUSION_TYPE:-change_aware}" \
bash "${SCRIPT_DIR}/run_stage2_full_finetune.sh" \
    "${BASE_MODEL}" \
    "${VISION_TOWER}" \
    "" \
    "${DOWNSTREAM_OUT}" \
    "${GPU_IDS}"

python "${SCRIPT_DIR}/summarize_ablation_logs.py" \
    --log-root "${PROJECT_ROOT}/logs/ablation" \
    --output-prefix "${PROJECT_ROOT}/logs/ablation/e2e_mllm_summary"
