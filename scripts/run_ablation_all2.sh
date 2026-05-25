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

RUN_FUSION_SWEEP="${RUN_FUSION_SWEEP:-1}"
RUN_NO_PRETRAIN="${RUN_NO_PRETRAIN:-1}"
RUN_E2E_MLLM="${RUN_E2E_MLLM:-1}"
RUN_LORA_SWEEP="${RUN_LORA_SWEEP:-1}"

# if [ "${RUN_FUSION_SWEEP}" = "1" ]; then
#     bash "${SCRIPT_DIR}/run_ablation_fusion_sweep.sh" \
#         "${PRETRAIN_DATA}" \
#         "${BASE_MODEL}" \
#         "${VISION_TOWER}" \
#         "${GPU_IDS}"
# fi

# if [ "${RUN_NO_PRETRAIN}" = "1" ]; then
#     bash "${SCRIPT_DIR}/run_ablation_no_pretrain.sh" \
#         "${BASE_MODEL}" \
#         "${VISION_TOWER}" \
#         "${PROJECT_ROOT}/checkpoint/ablation/no_pretrain" \
#         "${GPU_IDS}"
# fi

if [ "${RUN_E2E_MLLM}" = "1" ]; then
    bash "${SCRIPT_DIR}/run_ablation_e2e_mllm.sh" \
        "${PRETRAIN_DATA}" \
        "${BASE_MODEL}" \
        "${VISION_TOWER}" \
        "${PROJECT_ROOT}/checkpoint/ablation/e2e_mllm" \
        "${GPU_IDS}"
fi

if [ "${RUN_LORA_SWEEP}" = "1" ]; then
    bash "${SCRIPT_DIR}/run_ablation_lora_sweep.sh" \
        "${BASE_MODEL}" \
        "${VISION_TOWER}" \
        "${PROJECT_ROOT}/checkpoint/stage1_visual_change_pretrain_TV" \
        "${GPU_IDS}"
fi

python "${SCRIPT_DIR}/summarize_ablation_logs.py" \
    --log-root "${PROJECT_ROOT}/logs/ablation" \
    --output-prefix "${PROJECT_ROOT}/logs/ablation/all_ablations_summary"
