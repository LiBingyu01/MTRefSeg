#!/bin/bash

# Arguments instruction:
# --val_dataset="grefcoco|unc|val", the format is Dataset name | version | split, e.g., "grefcoco|unc|val", "refcoco+|unc|testA" .
# --segmentation_model_path="****/sam_vit_h_4b8939.pth", path to the pretrained SAM pth file.
# --mllm_model_path="****/llava-v1_1-7b", path to a directory where LLaVA hugginface model stores.
# --vision-tower="****/clip-vit-large-patch14", path to a directory where CLIP-ViT-L hugginface model stores.
# --dataset_dir="****/data", path to the dataset directory. 
# --weight="****/gsva-7b-ft-gres.bin", path to a GSVA checkpoint.
# --precision="fp32", precision for evaluation.
# --lora_r=8 , r = 8 for 7B model, r = 64 for 13B model.
# --eval_only, use this flag to perform evaluation.

export TRANSFORMERS_OFFLINE=1
export DS_SKIP_CUDA_CHECK=1

deepspeed --master_port=24999 main.py \
  --val_dataset="TV|val"  \
  --segmentation_model_path="/gemini/space/zhaozy/libingyu/LISA-main/pretrained_weights/sam_vit_h_4b8939.pth" \
  --mllm_model_path="weights" \
  --vision-tower="/gemini/space/zhaozy/libingyu/LISA-main-new/pretrained_weights/clip-vit-large-patch14" \
  --dataset_dir="/gemini/space/zhaozy/libingyu/ChangeRef_datasets" \
  --weight="/gemini/space/zhaozy/libingyu/GSVA-main/pretrained_weights/gsva_7b/gsva-7b-pt.bin" \
  --precision="bf16" \
  --lora_r=8 \
  --eval_only \
  --exp_name "gsva-bitemporal-eval-val" \
  --log_base_dir "eval_7B"


# ====================================================================
export TRANSFORMERS_OFFLINE=1
export DS_SKIP_CUDA_CHECK=1

deepspeed --master_port=24999  main.py \
  --val_dataset="TV|RS"  \
  --segmentation_model_path="/gemini/space/zhaozy/libingyu/LISA-main/pretrained_weights/sam_vit_h_4b8939.pth" \
  --mllm_model_path="weights" \
  --vision-tower="/gemini/space/zhaozy/libingyu/LISA-main-new/pretrained_weights/clip-vit-large-patch14" \
  --dataset_dir="/gemini/space/zhaozy/libingyu/ChangeRef_datasets" \
  --weight="/gemini/space/zhaozy/libingyu/GSVA-main/pretrained_weights/gsva_7b/gsva-7b-pt.bin" \
  --precision="bf16" \
  --lora_r=8 \
  --eval_only \
  --exp_name "gsva-bitemporal-eval-RS" \
  --log_base_dir "eval_7B"


# # ====================================================================
export TRANSFORMERS_OFFLINE=1
export DS_SKIP_CUDA_CHECK=1

deepspeed --master_port=24999  main.py \
  --val_dataset="TV|NS"  \
  --segmentation_model_path="/gemini/space/zhaozy/libingyu/LISA-main/pretrained_weights/sam_vit_h_4b8939.pth" \
  --mllm_model_path="weights" \
  --vision-tower="/gemini/space/zhaozy/libingyu/LISA-main-new/pretrained_weights/clip-vit-large-patch14" \
  --dataset_dir="/gemini/space/zhaozy/libingyu/ChangeRef_datasets" \
  --weight="/gemini/space/zhaozy/libingyu/GSVA-main/pretrained_weights/gsva_7b/gsva-7b-pt.bin" \
  --precision="bf16" \
  --lora_r=8 \
  --eval_only \
  --exp_name "gsva-bitemporal-eval-NS" \
  --log_base_dir "eval_7B"