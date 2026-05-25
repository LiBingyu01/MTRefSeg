#!/bin/bash

set -euo pipefail

export WANDB_DISABLED=True
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

source /lby_data01/zhaozy/lby/miniconda3/etc/profile.d/conda.sh
conda activate segr1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

DATA_ROOT="/lby_data01/zhaozy/lby/ChangeRef_Clear"

DATA_PATH="${1:-${DATA_ROOT}/MTRefSeg_pretrain_clean}"
MODEL_PATH="${2:-${PROJECT_ROOT}/pretrained_weights/phi-1_5_dev}"
VISION_TOWER="${3:-${PROJECT_ROOT}/pretrained_weights/model_final_9d7f02.pkl}"
OUTPUT_DIR="${4:-${PROJECT_ROOT}/checkpoint/e2e_mllm_pretrain}"
GPU_IDS="${5:-0,1,2,3}"

MASK_CONFIG="segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
VERSION="llava_phi"
MASTER_PORT="${MASTER_PORT:-29511}"
LOG_DIR="${PROJECT_ROOT}/logs"
RUN_NAME="${RUN_NAME:-e2e_mllm_pretrain}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
TEMPORAL_FUSION_TYPE="${TEMPORAL_FUSION_TYPE:-change_aware}"
E2E_PRETRAIN_EPOCHS="${E2E_PRETRAIN_EPOCHS:-5}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-5000}"
TRAIN_BACKBONE="${TRAIN_BACKBONE:-True}"
LORA_ENABLE="${LORA_ENABLE:-False}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

IFS=',' read -ra GPU_ARR <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_ARR[@]}
INCLUDE_STR="localhost:${GPU_IDS}"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"

echo "======================================================"
echo "  E2E MLLM Pretrain"
echo "  DATA_PATH    : ${DATA_PATH}"
echo "  MODEL_PATH   : ${MODEL_PATH}"
echo "  VISION_TOWER : ${VISION_TOWER}"
echo "  OUTPUT_DIR   : ${OUTPUT_DIR}"
echo "  LOG_FILE     : ${LOG_FILE}"
echo "  FUSION_TYPE  : ${TEMPORAL_FUSION_TYPE}"
echo "  EPOCHS       : ${E2E_PRETRAIN_EPOCHS}"
echo "  MAX_EVAL     : ${MAX_EVAL_SAMPLES}"
echo "  LORA_ENABLE  : ${LORA_ENABLE}"
echo "  GPUs         : ${GPU_IDS} (${NUM_GPUS} GPU(s))"
echo "======================================================"

if [ ! -d "${DATA_PATH}" ]; then
    echo "[ERROR] DATA_PATH not found: ${DATA_PATH}"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "${VISION_TOWER}" ]; then
    echo "[ERROR] VISION_TOWER not found: ${VISION_TOWER}"
    exit 1
fi

deepspeed --master_port "${MASTER_PORT}" \
          --include "${INCLUDE_STR}" \
  segearth_r1/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --data_path "${DATA_PATH}" \
    --dataset_type "change_detection" \
    --val_data_path "${DATA_PATH}" \
    --val_dataset_type "change_detection" \
    --max_eval_samples "${MAX_EVAL_SAMPLES}" \
    --model_name_or_path "${MODEL_PATH}" \
    --version "${VERSION}" \
    --vision_tower "${VISION_TOWER}" \
    --mask_config "${MASK_CONFIG}" \
    --temporal_fusion_type "${TEMPORAL_FUSION_TYPE}" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_projector_type "SparseConv_1" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${E2E_PRETRAIN_EPOCHS}" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --seg_task "referring" \
    --freeze_mm_mlp_adapter False \
    --bf16 True \
    --train_backbone "${TRAIN_BACKBONE}" \
    --save_total_limit 2 \
    --dataloader_drop_last True \
    --dataloader_num_workers 4 \
    --use_seg_query False \
    --lora_enable "${LORA_ENABLE}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
  2>&1 | tee "${LOG_FILE}"

echo "======================================================"
echo "  E2E MLLM pretrain finished."
echo "  Checkpoint: ${OUTPUT_DIR}"
echo "======================================================"
