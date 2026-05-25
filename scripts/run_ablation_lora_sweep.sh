#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

BASE_MODEL="${1:-${PROJECT_ROOT}/pretrained_weights/phi-1_5_dev}"
VISION_TOWER="${2:-${PROJECT_ROOT}/pretrained_weights/model_final_9d7f02.pkl}"
STAGE1_CKPT="${3:-${PROJECT_ROOT}/checkpoint/stage1_visual_change_pretrain_TV}"
GPU_IDS="${4:-0,1,2,3}"

LORA_RANKS_STR="${LORA_RANKS:-8 16 32 64}"
AB_ROOT="${PROJECT_ROOT}/checkpoint/ablation"
LOG_ROOT="${PROJECT_ROOT}/logs/ablation"

mkdir -p "${AB_ROOT}" "${LOG_ROOT}"

idx=0
for lora_r in ${LORA_RANKS_STR}; do
    exp_root="${AB_ROOT}/lora_r${lora_r}"
    alpha="${FIXED_LORA_ALPHA:-$((lora_r * 2))}"
    master_port=$((29721 + idx * 10))

    echo "======================================================"
    echo "  LoRA Ablation: r=${lora_r}, alpha=${alpha}"
    echo "  OUTPUT_ROOT : ${exp_root}"
    echo "======================================================"

    RUN_NAME="ablation/lora_r${lora_r}" \
    USE_STAGE1_PRETRAIN=1 \
    FULL_MODEL_INIT_PATH="" \
    LORA_ENABLE=True \
    LORA_R="${lora_r}" \
    LORA_ALPHA="${alpha}" \
    TRAIN_BACKBONE=False \
    MASTER_PORT_BASE="${master_port}" \
    bash "${SCRIPT_DIR}/run_stage2_full_finetune.sh" \
        "${BASE_MODEL}" \
        "${VISION_TOWER}" \
        "${STAGE1_CKPT}" \
        "${exp_root}" \
        "${GPU_IDS}"

    idx=$((idx + 1))
done

python "${SCRIPT_DIR}/summarize_ablation_logs.py" \
    --log-root "${LOG_ROOT}" \
    --output-prefix "${LOG_ROOT}/lora_sweep_summary"
