#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

BASE_MODEL="${1:-${PROJECT_ROOT}/pretrained_weights/phi-1_5_dev}"
VISION_TOWER="${2:-${PROJECT_ROOT}/pretrained_weights/model_final_9d7f02.pkl}"
OUTPUT_ROOT="${3:-${PROJECT_ROOT}/checkpoint/ablation/no_pretrain}"
GPU_IDS="${4:-0,1,2,3}"

mkdir -p "${OUTPUT_ROOT}" "${PROJECT_ROOT}/logs/ablation"

RUN_NAME="${RUN_NAME:-ablation/no_pretrain_stage2}" \
USE_STAGE1_PRETRAIN=0 \
FULL_MODEL_INIT_PATH="" \
TEMPORAL_FUSION_TYPE="${TEMPORAL_FUSION_TYPE:-change_aware}" \
bash "${SCRIPT_DIR}/run_stage2_full_finetune.sh" \
    "${BASE_MODEL}" \
    "${VISION_TOWER}" \
    "" \
    "${OUTPUT_ROOT}" \
    "${GPU_IDS}"

python "${SCRIPT_DIR}/summarize_ablation_logs.py" \
    --log-root "${PROJECT_ROOT}/logs/ablation" \
    --output-prefix "${PROJECT_ROOT}/logs/ablation/no_pretrain_summary"
