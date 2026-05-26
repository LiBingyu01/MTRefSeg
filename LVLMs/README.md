# Unified Guide for Bi-Temporal LVLM Repositories

This directory collects several remote-sensing large vision-language model repositories that have been adapted to support **bi-temporal image input** for change understanding, change referring segmentation, and change localization.

The following repositories have been converted to bi-temporal input:

- `GeoPixel-main`
- `groundingLMM-bi_fusion`
- `GSVA-bi_fusion`
- `LISA-bi_fusion`
- `SegEarth-R1-main-down`
- `UniChange-main`
- `UniGeoSeg-main`

## 1. What Was Modified

Most original repositories were designed for single-image or single-temporal visual input. The versions in this directory were modified to accept dual-time inputs, meaning:

- each sample contains two temporal images, `A` and `B`
- the model encodes both images in the forward pass
- bi-temporal interaction / bi-fusion / temporal fusion is introduced in the visual backbone or multimodal fusion stage
- the downstream task format remains consistent with the original repository, such as referring segmentation, mask prediction, change grounding, or change localization

A common dataset structure is:

```text
dataset_root/
├── A/
├── B/
├── masks/
│   └── <sample_name>/
└── referring_expression/
    └── *.json
```

The script names differ across repositories, but the overall logic is the same: read paired `A/B` images and use the corresponding masks / expressions for change understanding.

## 2. Shared Environment

All repositories are recommended to use a shared environment based on:

```bash
/lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/LISA-bi_fusion/requirements.txt
```

Recommended installation:

```bash
python -m pip install -r /lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/LISA-bi_fusion/requirements.txt
```

For `mmcv`, it is recommended to reinstall it separately with the following commands:

```bash
pip uninstall -y mmcv mmcv-full mmcv-lite openmim

python -m pip install -U pip wheel ninja
python -m pip install "setuptools<82" "packaging<25"

export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MAX_JOBS=8

python -m pip install --no-build-isolation -v "mmcv==2.2.0"
```

Notes:

- the shared environment is mainly based on `torch==2.4.0`, `transformers==4.31.0`, `deepspeed==0.9.5`, and `peft==0.4.0`
- all commands below are assumed to be executed from the **root directory of each repository**
- many scripts contain local absolute paths, so you should check dataset paths, pretrained weights, and output directories before running them

## 3. Unified Checkpoint Export Workflow

Some repositories use LoRA + DeepSpeed ZeRO training. In those cases, checkpoint export usually consists of two steps:

1. Enter the training output directory such as `ckpt_model` or `checkpoint-*`, then run `zero_to_fp32.py` to merge the sharded DeepSpeed checkpoint into a single `pytorch_model.bin`
2. Run the repository-specific merge script to merge LoRA weights back into the base model and export a HuggingFace-style model for inference

If a repository is not trained with LoRA, its checkpoints are usually saved directly under `--output_dir` and no extra merge step is needed.

## 4. Repository-Wise Usage

### 4.1 `GeoPixel-main`

Repository path:

```bash
/lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/GeoPixel-main
```

Required pretrained weights:

- `pretrain_weights/GeoPixel-7B-RES`
- `pretrain_weights/sam2_hiera_large.pt`

Note:

- scripts often refer to the SAM2 visual checkpoint as `pretrain_weights/sam2-hiera-large`
- internally the code maps this to the actual file `pretrain_weights/sam2_hiera_large.pt`

Training:

```bash
cd /lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/GeoPixel-main
bash train.sh
```

If you want to use the combined train + eval workflow:

```bash
bash run_train_and_eval.sh
```

Saving / exporting weights:

```bash
cd output/checkpoint-xxxx
python zero_to_fp32.py . ../pytorch_model.bin

cd /lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/GeoPixel-main
python merge_lora_weights_and_save_hf_model.py \
  --version pretrain_weights/GeoPixel-7B-RES \
  --vision_pretrained pretrain_weights/sam2-hiera-large \
  --weight output/pytorch_model.bin \
  --save_path output/GeoPixel_bitemporal_merged
```

Inference / evaluation:

```bash
bash run_eval_zero_shot_all.sh
```

You can also directly refer to:

- `evaluate_changeref.py`
- `docs/inference.md`

### 4.2 `groundingLMM-bi_fusion`

Repository path:

```bash
/lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/groundingLMM-bi_fusion
```

Required pretrained weights:

- `pretrained_weights/GLaMM-GranD-Pretrained`
- `pretrained_weights/sam_vit_h_4b8939.pth`
- `pretrained_weights/clip-vit-large-patch14`

Training:

```bash
cd /lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/groundingLMM-bi_fusion
bash scripts/train_change_ref.sh
```

Saving / exporting weights:

```bash
bash save.sh
```

This script already includes:

- `zero_to_fp32.py` for merging DeepSpeed shards
- `scripts/merge_lora_weights.py` for merging LoRA into the base model

Inference / evaluation:

```bash
bash scripts/eval_change_ref.sh
```

Visualization:

```bash
bash scripts/vis_change_ref.sh
```

### 4.3 `GSVA-bi_fusion`

Repository path:

```bash
/lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/GSVA-bi_fusion
```

Required pretrained weights:

- `pretrained_weights/weights` for 7B
- `pretrained_weights/weights_13b` for 13B
- `pretrained_weights/gsva-7b-pt.bin`
- `pretrained_weights/gsva-13b-pt.bin`
- `pretrained_weights/clip-vit-large-patch14`
- `pretrained_weights/sam_vit_h_4b8939.pth`

Training:

7B:

```bash
cd /lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/GSVA-bi_fusion
bash scripts/train_demo_7B.sh
```

13B:

```bash
bash scripts/train_demo_13B.sh
```

Saving / exporting weights:

7B:

```bash
bash save_7B.sh
```

13B:

```bash
bash save_13B.sh
```

If you need a HuggingFace-style exported model, you can further use:

```bash
python merge_lora_weights_and_save_hf_model.py
```

Inference / evaluation:

```bash
bash scripts/eval_demo.sh
```

Visualization:

```bash
bash vis_7B.sh
```

### 4.4 `LISA-bi_fusion`

Repository path:

```bash
/lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/LISA-bi_fusion
```

Required pretrained weights:

- `pretrained_weights/LISA-7B-v1`
- `pretrained_weights/LISA-13B-llama2-v1`
- `pretrained_weights/clip-vit-large-patch14`
- `pretrained_weights/sam_vit_h_4b8939.pth`

Training:

7B:

```bash
cd /lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/LISA-bi_fusion
bash run_7B.sh
```

13B:

```bash
bash run_13B.sh
```

Saving / exporting weights:

7B:

```bash
bash save_7B.sh
```

The 13B flow follows the same logic:

- first run `zero_to_fp32.py` inside the corresponding `ckpt_model` directory
- then run `merge_lora_weights_and_save_hf_model.py` to merge LoRA

Inference / evaluation:

7B:

```bash
bash infer_7B.sh
```

13B:

```bash
bash infer_13B.sh
```

### 4.5 `SegEarth-R1-main-down`

Repository path:

```bash
/lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/SegEarth-R1-main-down
```

Required pretrained weights:

- `pretrained_weights/phi-1_5_dev`
- `pretrained_weights/model_final_9d7f02.pkl`

Optional released checkpoint:

- `pretrained_weights/SegEarth-R1-RRSIS-D`

Training:

train/val main setting:

```bash
cd /lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/SegEarth-R1-main-down
bash scripts/train_change_train_val.sh
```

NS:

```bash
bash scripts/train_change_ns.sh
```

RS:

```bash
bash scripts/train_change_rs.sh
```

Saving / exporting weights:

- the main training pipeline in this repository saves checkpoints directly into the configured `--output_dir`
- in the standard workflow, no extra LoRA merge step is required
- if you want a final deployable checkpoint, use the best checkpoint under the training output directory

Inference / evaluation:

```bash
bash scripts/eval_change.sh
```

Batch evaluation:

```bash
bash scripts/eval_all.sh
```

### 4.6 `UniChange-main`

Repository path:

```bash
/lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/UniChange-main
```

Required pretrained weights:

- `pretrained_weights/LLaVA-Lightning-7B-delta-v1-1`
- `pretrained_weights/clip-vit-large-patch14`
- `pretrained_weights/ViT_L.pth`

Training:

```bash
cd /lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/UniChange-main
bash train_UniChange.sh
```

Saving / exporting weights:

```bash
bash save.sh
```

This script already includes:

- `zero_to_fp32.py` for merging shards
- `merge_lora_weights_and_save_hf_model.py` for merging LoRA

Inference / evaluation:

```bash
bash validation_UniChange.sh
```

Visualization:

```bash
bash vis_7B.sh
```

### 4.7 `UniGeoSeg-main`

Repository path:

```bash
/lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/UniGeoSeg-main
```

Required pretrained weights:

- `pretrained_weights/UniGeoSeg`

Current status:

- this adapted bi-temporal directory already provides organized **inference / evaluation** entry points
- unlike the other repositories, it does not currently expose a complete standalone training script and export workflow
- in practice, this directory should be treated as a **bi-temporal evaluation-ready version**

Inference / evaluation:

```bash
cd /lby_data01/zhaozy/lby/ChangeVLM-Other_MLLM_RCD/LVLM/UniGeoSeg-main
bash scripts/eval_change.sh
```

Batch evaluation:

```bash
bash scripts/eval_all.sh
```

## 5. Recommended Usage Order

If you are reproducing experiments in this directory for the first time, the recommended order is:

1. create the shared environment and install `mmcv==2.2.0` as shown above
2. prepare the required base checkpoints for each repository
3. inspect the training scripts and update dataset paths, output paths, GPU settings, and checkpoint paths
4. after training, if the repository uses LoRA + DeepSpeed, first run `zero_to_fp32.py`, then run the merge script
5. finally run the repository-specific `eval`, `infer`, or `vis` scripts for validation and qualitative inspection

## 6. Additional Notes

- `LISA-bi_fusion/requirements.txt` is used as the shared environment baseline for the whole collection
- `SegEarth-R1-main-down` is closer to direct checkpoint saving and is not centered around a default LoRA-merge workflow
- `UniGeoSeg-main` currently focuses on bi-temporal inference / evaluation
- in `GeoPixel-main`, note the naming detail of the SAM2 checkpoint: the script argument is `pretrain_weights/sam2-hiera-large`, while the actual file is `pretrain_weights/sam2_hiera_large.pt`

