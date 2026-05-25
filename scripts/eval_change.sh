#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_change.sh  –  Evaluate SegEarth-R1 on bi-temporal change detection
#
# Usage:
#   bash scripts/eval_change.sh [MODEL_PATH] [DATA_PATH] [SPLIT] [GPU_IDS] [VIS_PATH]
#
# Positional arguments (all optional):
#   1. MODEL_PATH   – trained SegEarth-R1 checkpoint directory
#   2. DATA_PATH    – root dir of the change-detection dataset
#   3. SPLIT        – evaluation split, e.g. "val", "RS", "NS", "TT"  (default: val)
#   4. GPU_IDS      – comma-separated GPU ids, e.g. "0,1,2,3"         (default: 0)
#   5. VIS_PATH     – output directory for visualisations (omit to skip)
#
# Examples:
#   # Single GPU
#   bash scripts/eval_change.sh ./checkpoint/model /data/ChangeRef val 0
#
#   # Four GPUs
#   bash scripts/eval_change.sh ./checkpoint/model /data/ChangeRef val 0,1,2,3
# ─────────────────────────────────────────────────────────────────────────────
set -e

MODEL_PATH="${1:-./checkpoint/SegEarth-R1_ChangeDetection}"
DATA_PATH="${2:-/path/to/change_detection_dataset}"
SPLIT="${3:-val}"
GPU_IDS="${4:-0}"
VIS_PATH="${5:-}"

MASK_CONFIG="segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
VERSION="llava_phi"
IMAGE_SIZE=1024
BATCH_SIZE=16
NUM_WORKERS=4

# ── Derive number of GPUs and CUDA_VISIBLE_DEVICES ───────────────────────────
IFS=',' read -ra GPU_ARR <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARR[@]}
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

echo "======================================================"
echo "  SegEarth-R1 Change Detection Evaluation"
echo "  MODEL_PATH : ${MODEL_PATH}"
echo "  DATA_PATH  : ${DATA_PATH}"
echo "  SPLIT      : ${SPLIT}"
echo "  GPUs       : ${GPU_IDS}  (${NUM_GPUS} GPU(s))"
echo "======================================================"

torchrun --nproc_per_node=${NUM_GPUS} \
         --master_port 29502 \
  segearth_r1/eval_and_test/eval_change.py \
  --model_path         "${MODEL_PATH}" \
  --base_data_path     "${DATA_PATH}" \
  --data_split         "${SPLIT}" \
  --version            "${VERSION}" \
  --mask_config        "${MASK_CONFIG}" \
  --image_size         ${IMAGE_SIZE} \
  --eval_batch_size    ${BATCH_SIZE} \
  --dataloader_num_workers ${NUM_WORKERS} \
  ${VIS_PATH:+--vis_path "${VIS_PATH}"}

echo "Evaluation done."
