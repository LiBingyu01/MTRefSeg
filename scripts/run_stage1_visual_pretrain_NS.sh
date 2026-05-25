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

DATA_PATH="${1:-${DATA_ROOT}/MTRefSeg_pretrain_NS}"
MODEL_PATH="${2:-${PROJECT_ROOT}/pretrained_weights/phi-1_5_dev}"
VISION_TOWER="${3:-${PROJECT_ROOT}/pretrained_weights/model_final_9d7f02.pkl}"
OUTPUT_DIR="${4:-${PROJECT_ROOT}/checkpoint/stage1_visual_change_pretrain_NS}"
GPU_IDS="${5:-0,1,2,3}"

MASK_CONFIG="segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
VERSION="llava_phi"
MASTER_PORT="${MASTER_PORT:-29501}"
LOG_DIR="${PROJECT_ROOT}/logs"
RUN_NAME="${RUN_NAME:-stage1_visual_change_pretrain_NS}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
TEMPORAL_FUSION_TYPE="${TEMPORAL_FUSION_TYPE:-change_aware}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-5000}"
STAGE1_TRAIN_MM_PROJECTOR="${STAGE1_TRAIN_MM_PROJECTOR:-False}"
STAGE1_TRAIN_MASK_DECODER="${STAGE1_TRAIN_MASK_DECODER:-True}"
STAGE1_TRAIN_VISION_TOWER="${STAGE1_TRAIN_VISION_TOWER:-True}"
STAGE1_TRAIN_TEMPORAL_MODULES="${STAGE1_TRAIN_TEMPORAL_MODULES:-True}"
STAGE1_TRAIN_SEG_PROJECTORS="${STAGE1_TRAIN_SEG_PROJECTORS:-False}"
STAGE1_TRAIN_EMBEDDINGS="${STAGE1_TRAIN_EMBEDDINGS:-False}"
STAGE1_TRAIN_LM_HEAD="${STAGE1_TRAIN_LM_HEAD:-False}"

IFS=',' read -ra GPU_ARR <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_ARR[@]}
INCLUDE_STR="localhost:${GPU_IDS}"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"

echo "======================================================"
echo "  Stage1 Visual Change Pretrain"
echo "  DATA_PATH    : ${DATA_PATH}"
echo "  MODEL_PATH   : ${MODEL_PATH}"
echo "  VISION_TOWER : ${VISION_TOWER}"
echo "  OUTPUT_DIR   : ${OUTPUT_DIR}"
echo "  LOG_FILE     : ${LOG_FILE}"
echo "  FUSION_TYPE  : ${TEMPORAL_FUSION_TYPE}"
echo "  EPOCHS       : ${STAGE1_EPOCHS}"
echo "  MAX_EVAL     : ${MAX_EVAL_SAMPLES}"
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
  segearth_r1/train/train_stage1.py \
    --deepspeed ./scripts/zero2.json \
    \
    --data_path "${DATA_PATH}" \
    --dataset_type "change_detection_stage1" \
    --val_data_path "${DATA_PATH}" \
    --val_dataset_type "change_detection_stage1" \
    \
    --model_name_or_path "${MODEL_PATH}" \
    --version "${VERSION}" \
    \
    --vision_tower "${VISION_TOWER}" \
    --mask_config "${MASK_CONFIG}" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_projector_type "SparseConv_1" \
    --temporal_fusion_type "${TEMPORAL_FUSION_TYPE}" \
    \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${STAGE1_EPOCHS}" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --max_eval_samples "${MAX_EVAL_SAMPLES}" \
    --seg_task "referring" \
    --freeze_mm_mlp_adapter False \
    --bf16 True \
    --save_total_limit 2 \
    --dataloader_drop_last True \
    --dataloader_num_workers 4 \
    \
    --use_seg_query False \
    --lora_enable False \
    --stage1_train_mm_projector "${STAGE1_TRAIN_MM_PROJECTOR}" \
    --stage1_train_mask_decoder "${STAGE1_TRAIN_MASK_DECODER}" \
    --stage1_train_vision_tower "${STAGE1_TRAIN_VISION_TOWER}" \
    --stage1_train_temporal_modules "${STAGE1_TRAIN_TEMPORAL_MODULES}" \
    --stage1_train_seg_projectors "${STAGE1_TRAIN_SEG_PROJECTORS}" \
    --stage1_train_embeddings "${STAGE1_TRAIN_EMBEDDINGS}" \
    --stage1_train_lm_head "${STAGE1_TRAIN_LM_HEAD}" \
    --stage1_verbose_trainable True \
  2>&1 | tee "${LOG_FILE}"

echo "======================================================"
echo "  Stage1 finished."
echo "  Checkpoint: ${OUTPUT_DIR}"
echo "======================================================"
