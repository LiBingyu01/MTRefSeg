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
STAGE1_CKPT="${3:-${PROJECT_ROOT}/checkpoint/stage1_visual_change_pretrain_converged/checkpoint-1675}"
OUTPUT_ROOT="${4:-${PROJECT_ROOT}/checkpoint}"
GPU_IDS="${5:-0,1,2,3}"

MASK_CONFIG="segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
VERSION="llava_phi"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29501}"
LOG_DIR="${PROJECT_ROOT}/logs"
RUN_NAME="${RUN_NAME:-stage2_full_finetune}"
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

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

echo "======================================================"
echo "  Stage2 Full Finetune - Three Separate Runs"
echo "  BASE_MODEL   : ${BASE_MODEL}"
echo "  VISION_TOWER : ${VISION_TOWER}"
echo "  STAGE1_CKPT  : ${STAGE1_CKPT}"
echo "  USE_STAGE1   : ${USE_STAGE1_PRETRAIN}"
echo "  FULL_INIT    : ${FULL_MODEL_INIT_PATH:-<none>}"
echo "  FUSION_TYPE  : ${TEMPORAL_FUSION_TYPE}"
echo "  EPOCHS       : ${STAGE2_EPOCHS}"
echo "  LORA_ENABLE  : ${LORA_ENABLE}"
echo "  OUTPUT_ROOT  : ${OUTPUT_ROOT}"
echo "  LOG_DIR      : ${LOG_DIR}"
echo "  GPUs         : ${GPU_IDS} (${NUM_GPUS} GPU(s))"
echo "======================================================"

for required_path in \
    "${TRAIN_FINAL}" \
    "${VAL_FINAL}" \
    "${NS_TRAIN}" \
    "${NS_VAL}" \
    "${RS_TRAIN}" \
    "${RS_VAL}"
do
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
    if [ ! -d "${STAGE1_CKPT}" ] && [ ! -f "${STAGE1_CKPT}" ]; then
        echo "[ERROR] STAGE1_CKPT not found: ${STAGE1_CKPT}"
        exit 1
    fi
fi

if [ -n "${FULL_MODEL_INIT_PATH}" ] && [ ! -d "${FULL_MODEL_INIT_PATH}" ] && [ ! -f "${FULL_MODEL_INIT_PATH}" ]; then
    echo "[ERROR] FULL_MODEL_INIT_PATH not found: ${FULL_MODEL_INIT_PATH}"
    exit 1
fi

run_case() {
    local case_name="$1"
    local train_data="$2"
    local val_data="$3"
    local output_dir="$4"
    local log_file="$5"
    local master_port="$6"

    echo "======================================================"
    echo "  Running ${case_name}"
    echo "  TRAIN_DATA : ${train_data}"
    echo "  VAL_DATA   : ${val_data}"
    echo "  OUTPUT_DIR : ${output_dir}"
    echo "  LOG_FILE   : ${log_file}"
    echo "  PORT       : ${master_port}"
    echo "======================================================"

    mkdir -p "${output_dir}" "$(dirname "${log_file}")"

    cmd=(
        deepspeed
        --master_port "${master_port}"
        --include "${INCLUDE_STR}"
        segearth_r1/train/train.py
        --deepspeed ./scripts/zero2.json
        --data_path "${train_data}"
        --dataset_type "change_detection"
        --val_data_path "${val_data}"
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
        --output_dir "${output_dir}"
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
        cmd+=(--stage1_pretrained_path "${STAGE1_CKPT}")
    fi

    if [ -n "${FULL_MODEL_INIT_PATH}" ]; then
        cmd+=(--full_model_init_path "${FULL_MODEL_INIT_PATH}")
    fi

    if [ -n "${MAX_EVAL_SAMPLES}" ]; then
        cmd+=(--max_eval_samples "${MAX_EVAL_SAMPLES}")
    fi

    "${cmd[@]}" 2>&1 | tee "${log_file}"
}

run_case \
    "Stage2-RS" \
    "${RS_TRAIN}" \
    "${RS_VAL}" \
    "${OUTPUT_ROOT}/stage2_rs_final" \
    "${LOG_DIR}/${RUN_NAME}_rs_final.log" \
    "$((MASTER_PORT_BASE + 2))"

echo "======================================================"
echo "  All three stage2 fine-tuning runs finished."
echo "  Outputs:"
echo "    ${OUTPUT_ROOT}/stage2_train_final_val"
echo "    ${OUTPUT_ROOT}/stage2_ns_final"
echo "    ${OUTPUT_ROOT}/stage2_rs_final"
echo "======================================================"
