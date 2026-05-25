#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/checkpoint/stage2_ns_final}"
BASE_DATA_PATH="${BASE_DATA_PATH:-/lby_data01/zhaozy/lby/ChangeRef_Clear/NS_FINAL_CLEAN_val}"
SPLIT="${SPLIT:-val}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r1/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"
VERSION="${VERSION:-llava_phi}"
DEVICE="${DEVICE:-cuda}"
CAST_DTYPE="${CAST_DTYPE:-float32}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
USE_SEG_QUERY="${USE_SEG_QUERY:-0}"
LOGIT_THRESHOLD="${LOGIT_THRESHOLD:-0.5}"
ALPHA="${ALPHA:-0.45}"

# Batch control
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-31}"
STEP="${STEP:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
RANDOM_SEED="${RANDOM_SEED:-}"

# Visualization detail
LAYER_INDICES="${LAYER_INDICES:-first,2,4,6,8,10,mid,last}"
SAVE_LAYERWISE="${SAVE_LAYERWISE:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/attention_vis/batch_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUTPUT_ROOT"

echo "[visualize_mt_attn] ROOT_DIR=$ROOT_DIR"
echo "[visualize_mt_attn] MODEL_PATH=$MODEL_PATH"
echo "[visualize_mt_attn] BASE_DATA_PATH=$BASE_DATA_PATH"
echo "[visualize_mt_attn] SPLIT=$SPLIT"
echo "[visualize_mt_attn] CAST_DTYPE=$CAST_DTYPE"
echo "[visualize_mt_attn] START_INDEX=$START_INDEX (ignored: random selection uses the full dataset)"
echo "[visualize_mt_attn] END_INDEX=$END_INDEX (ignored: random selection uses the full dataset)"
echo "[visualize_mt_attn] STEP=$STEP (ignored: random selection uses the full dataset)"
echo "[visualize_mt_attn] MAX_SAMPLES=$MAX_SAMPLES"
echo "[visualize_mt_attn] RANDOM_SEED=${RANDOM_SEED:-<unset>}"
echo "[visualize_mt_attn] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[visualize_mt_attn] LAYER_INDICES=$LAYER_INDICES"
echo "[visualize_mt_attn] SAVE_LAYERWISE=$SAVE_LAYERWISE"

count=0

mapfile -t SAMPLE_INDICES < <(
  python - "$ROOT_DIR" "$BASE_DATA_PATH" "$SPLIT" "$IMAGE_SIZE" "$MAX_SAMPLES" "$RANDOM_SEED" <<'PY'
import contextlib
import io
import random
import sys

root_dir = sys.argv[1]
base_data_path = sys.argv[2]
split = sys.argv[3]
image_size = int(sys.argv[4])
max_samples = int(sys.argv[5])
seed = sys.argv[6]

sys.path.insert(0, root_dir)
from segearth_r1.eval_and_test.eval_dataset.change_val_dataset import ChangeValDataset

with contextlib.redirect_stdout(io.StringIO()):
    dataset = ChangeValDataset(
        base_data_path=base_data_path,
        tokenizer=None,
        split=split,
        image_size=image_size,
    )

indices = list(range(len(dataset)))
rng = random.Random()
if seed:
    rng.seed(seed)
rng.shuffle(indices)

if max_samples > 0:
    indices = indices[:max_samples]

for idx in indices:
    print(idx)
PY
)

if [[ "${#SAMPLE_INDICES[@]}" -eq 0 ]]; then
  echo "[visualize_mt_attn] no samples found under BASE_DATA_PATH=$BASE_DATA_PATH SPLIT=$SPLIT"
  exit 0
fi

echo "[visualize_mt_attn] selected_samples=${#SAMPLE_INDICES[@]}"

for idx in "${SAMPLE_INDICES[@]}"; do
  sample_output_dir="$OUTPUT_ROOT/sample_${idx}"
  mkdir -p "$sample_output_dir"

  ARGS=(
    --model-path "$MODEL_PATH"
    --base-data-path "$BASE_DATA_PATH"
    --split "$SPLIT"
    --sample-index "$idx"
    --output-dir "$sample_output_dir"
    --mask-config "$MASK_CONFIG"
    --version "$VERSION"
    --device "$DEVICE"
    --cast-dtype "$CAST_DTYPE"
    --layer-indices "$LAYER_INDICES"
    --image-size "$IMAGE_SIZE"
    --logit-threshold "$LOGIT_THRESHOLD"
    --alpha "$ALPHA"
  )

  if [[ "$USE_SEG_QUERY" == "1" ]]; then
    ARGS+=(--use-seg-query)
  fi

  if [[ "$SAVE_LAYERWISE" == "1" ]]; then
    ARGS+=(--save-layerwise)
  fi

  echo "[visualize_mt_attn] running sample-index=$idx -> $sample_output_dir"
  python "$ROOT_DIR/scripts/visualize_bitemporal_attention.py" "${ARGS[@]}"

  count=$((count + 1))
done

echo "[visualize_mt_attn] finished. processed_samples=$count output_root=$OUTPUT_ROOT"
