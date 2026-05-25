#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_ROOT="/lby_data01/zhaozy/lby/ChangeRef_Clear"
PRETRAIN_DATA="${1:-${DATA_ROOT}/MTRefSeg_pretrain_TV}"
BASE_MODEL="${2:-${PROJECT_ROOT}/pretrained_weights/phi-1_5_dev}"
VISION_TOWER="${3:-${PROJECT_ROOT}/pretrained_weights/model_final_9d7f02.pkl}"
GPU_IDS="${4:-0,1,2,3}"

FUSION_TYPES_STR="${FUSION_TYPES:-change_aware concat_conv abs_diff avg}"
AB_ROOT="${PROJECT_ROOT}/checkpoint/ablation"
LOG_ROOT="${PROJECT_ROOT}/logs/ablation"

mkdir -p "${AB_ROOT}" "${LOG_ROOT}"

idx=0
for fusion_type in ${FUSION_TYPES_STR}; do
    exp_root="${AB_ROOT}/fusion_${fusion_type}"
    stage1_out="${exp_root}/stage1"
    stage2_out="${exp_root}/stage2"
    stage1_port=$((29521 + idx * 10))
    stage2_port=$((29621 + idx * 10))

    echo "======================================================"
    echo "  Fusion Ablation: ${fusion_type}"
    echo "  STAGE1_OUT : ${stage1_out}"
    echo "  STAGE2_OUT : ${stage2_out}"
    echo "======================================================"

    RUN_NAME="ablation/fusion_${fusion_type}_stage1" \
    TEMPORAL_FUSION_TYPE="${fusion_type}" \
    MASTER_PORT="${stage1_port}" \
    bash "${SCRIPT_DIR}/run_stage1_visual_pretrain_TV.sh" \
        "${PRETRAIN_DATA}" \
        "${BASE_MODEL}" \
        "${VISION_TOWER}" \
        "${stage1_out}" \
        "${GPU_IDS}"

    RUN_NAME="ablation/fusion_${fusion_type}_stage2" \
    TEMPORAL_FUSION_TYPE="${fusion_type}" \
    MASTER_PORT_BASE="${stage2_port}" \
    bash "${SCRIPT_DIR}/run_stage2_full_finetune.sh" \
        "${BASE_MODEL}" \
        "${VISION_TOWER}" \
        "${stage1_out}" \
        "${stage2_out}" \
        "${GPU_IDS}"

    idx=$((idx + 1))
done

python "${SCRIPT_DIR}/summarize_ablation_logs.py" \
    --log-root "${LOG_ROOT}" \
    --output-prefix "${LOG_ROOT}/fusion_sweep_summary"
