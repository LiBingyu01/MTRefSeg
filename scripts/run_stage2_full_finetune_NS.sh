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

BASE_MODEL="${1:-${PROJECT_ROOT}/pretrained_weights/phi-1_5_dev}"
VISION_TOWER="${2:-${PROJECT_ROOT}/pretrained_weights/model_final_9d7f02.pkl}"
STAGE1_CKPT_INPUT="${3:-${PROJECT_ROOT}/checkpoint/stage1_visual_change_pretrain_NS}"
OUTPUT_DIR="${4:-${PROJECT_ROOT}/checkpoint/stage2_ns_final}"
GPU_IDS="${5:-0,1,2,3}"

MASK_CONFIG="segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
VERSION="llava_phi"
MASTER_PORT="${MASTER_PORT:-29601}"
LOG_DIR="${PROJECT_ROOT}/logs"
RUN_NAME="${RUN_NAME:-stage2_full_finetune_ns_final}"
TEMPORAL_FUSION_TYPE="${TEMPORAL_FUSION_TYPE:-change_aware}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
USE_STAGE1_PRETRAIN="${USE_STAGE1_PRETRAIN:-1}"
FULL_MODEL_INIT_PATH="${FULL_MODEL_INIT_PATH:-}"
TRAIN_BACKBONE="${TRAIN_BACKBONE:-True}"
LORA_ENABLE="${LORA_ENABLE:-False}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"

TRAIN_FINAL="${DATA_ROOT}/train_FINAL_CLEAN"
VAL_FINAL="${DATA_ROOT}/val_FINAL_CLEAN"
NS_TRAIN="${DATA_ROOT}/NS_FINAL_CLEAN_train"
NS_VAL="${DATA_ROOT}/NS_FINAL_CLEAN_val"
RS_TRAIN="${DATA_ROOT}/RS_FINAL_CLEAN_train"
RS_VAL="${DATA_ROOT}/RS_FINAL_CLEAN_val"

IFS=',' read -ra GPU_ARR <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_ARR[@]}
INCLUDE_STR="localhost:${GPU_IDS}"

resolve_stage1_ckpt() {
    local ckpt_path="$1"
    local latest_ckpt

    if [ -z "${ckpt_path}" ]; then
        return 0
    fi

    if [ -f "${ckpt_path}" ]; then
        echo "${ckpt_path}"
        return 0
    fi

    if [ -d "${ckpt_path}" ]; then
        if [ -f "${ckpt_path}/model.safetensors" ] || [ -f "${ckpt_path}/pytorch_model.bin" ]; then
            echo "${ckpt_path}"
            return 0
        fi

        latest_ckpt="$(find "${ckpt_path}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
        if [ -n "${latest_ckpt}" ]; then
            echo "${latest_ckpt}"
            return 0
        fi
    fi

    echo "${ckpt_path}"
}

STAGE1_CKPT_RESOLVED="$(resolve_stage1_ckpt "${STAGE1_CKPT_INPUT}")"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "======================================================"
echo "  Stage2 Full Finetune - NS"
echo "  BASE_MODEL   : ${BASE_MODEL}"
echo "  VISION_TOWER : ${VISION_TOWER}"
echo "  STAGE1_CKPT  : ${STAGE1_CKPT_INPUT}"
echo "  STAGE1_LOAD  : ${STAGE1_CKPT_RESOLVED}"
echo "  USE_STAGE1   : ${USE_STAGE1_PRETRAIN}"
echo "  FULL_INIT    : ${FULL_MODEL_INIT_PATH:-<none>}"
echo "  FUSION_TYPE  : ${TEMPORAL_FUSION_TYPE}"
echo "  EPOCHS       : ${STAGE2_EPOCHS}"
echo "  LORA_ENABLE  : ${LORA_ENABLE}"
echo "  OUTPUT_DIR   : ${OUTPUT_DIR}"
echo "  LOG_FILE     : ${LOG_DIR}/${RUN_NAME}.log"
echo "  GPUs         : ${GPU_IDS} (${NUM_GPUS} GPU(s))"
echo "======================================================"

for required_path in "${NS_TRAIN}" "${NS_VAL}"; do
    if [ ! -d "${required_path}" ]; then
        echo "[ERROR] Dataset path not found: ${required_path}"
        exit 1
    fi
done

if [ ! -d "${BASE_MODEL}" ]; then
    echo "[ERROR] BASE_MODEL not found: ${BASE_MODEL}"
    exit 1
fi

if [ ! -f "${VISION_TOWER}" ]; then
    echo "[ERROR] VISION_TOWER not found: ${VISION_TOWER}"
    exit 1
fi

if [ "${USE_STAGE1_PRETRAIN}" = "1" ]; then
    if [ ! -d "${STAGE1_CKPT_RESOLVED}" ] && [ ! -f "${STAGE1_CKPT_RESOLVED}" ]; then
        echo "[ERROR] STAGE1_CKPT not found: ${STAGE1_CKPT_INPUT}"
        exit 1
    fi
fi

if [ -n "${FULL_MODEL_INIT_PATH}" ] && [ ! -d "${FULL_MODEL_INIT_PATH}" ] && [ ! -f "${FULL_MODEL_INIT_PATH}" ]; then
    echo "[ERROR] FULL_MODEL_INIT_PATH not found: ${FULL_MODEL_INIT_PATH}"
    exit 1
fi

LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

echo "======================================================"
echo "  Running Stage2-NS"
echo "  TRAIN_DATA : ${NS_TRAIN}"
echo "  VAL_DATA   : ${NS_VAL}"
echo "  OUTPUT_DIR : ${OUTPUT_DIR}"
echo "  LOG_FILE   : ${LOG_FILE}"
echo "  PORT       : ${MASTER_PORT}"
echo "======================================================"

cmd=(
    deepspeed
    --master_port "${MASTER_PORT}"
    --include "${INCLUDE_STR}"
    segearth_r1/train/train.py
    --deepspeed ./scripts/zero2.json
    --data_path "${NS_TRAIN}"
    --dataset_type "change_detection"
    --val_data_path "${NS_VAL}"
    --val_dataset_type "change_detection"
    --model_name_or_path "${BASE_MODEL}"
    --version "${VERSION}"
    --vision_tower "${VISION_TOWER}"
    --mask_config "${MASK_CONFIG}"
    --temporal_fusion_type "${TEMPORAL_FUSION_TYPE}"
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --mm_projector_type "SparseConv_1"
    --output_dir "${OUTPUT_DIR}"
    --num_train_epochs "${STAGE2_EPOCHS}"
    --per_device_train_batch_size 4
    --gradient_accumulation_steps 8
    --evaluation_strategy "no"
    --save_strategy "epoch"
    --learning_rate 5e-5
    --weight_decay 0.0
    --warmup_ratio 0.05
    --lr_scheduler_type "cosine"
    --logging_steps 1
    --model_max_length 2048
    --gradient_checkpointing True
    --lazy_preprocess True
    --seg_task "referring"
    --freeze_mm_mlp_adapter False
    --bf16 True
    --train_backbone "${TRAIN_BACKBONE}"
    --save_total_limit 3
    --dataloader_drop_last True
    --dataloader_num_workers 4
    --lora_enable "${LORA_ENABLE}"
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
)

if [ "${USE_STAGE1_PRETRAIN}" = "1" ]; then
    cmd+=(--stage1_pretrained_path "${STAGE1_CKPT_RESOLVED}")
fi

if [ -n "${FULL_MODEL_INIT_PATH}" ]; then
    cmd+=(--full_model_init_path "${FULL_MODEL_INIT_PATH}")
fi

if [ -n "${MAX_EVAL_SAMPLES}" ]; then
    cmd+=(--max_eval_samples "${MAX_EVAL_SAMPLES}")
fi

"${cmd[@]}" 2>&1 | tee "${LOG_FILE}"

echo "======================================================"
echo "  Stage2-NS finished."
echo "  Output: ${OUTPUT_DIR}"
echo "======================================================"
