#!/bin/bash

# Arguments instruction:
# --segmentation_model_path="****/sam_vit_h_4b8939.pth", path to the pretrained SAM pth file.
# --mllm_model_path="****/llava-v1_1-7b", path to a directory where LLaVA hugginface model stores.
# --vision-tower="****/clip-vit-large-patch14", path to a directory where CLIP-ViT-L hugginface model stores.
# --dataset_dir="****/data", path to the dataset directory. 
# --weight="****/gsva-7b-ft-gres.bin", path to a pretrained GSVA checkpoint if finetune.
# --precision="bf16", precision for training.
# --lora_r=8 , r = 8 for 7B model, r = 64 for 13B model.
# num_classes_per_sample=5, 5 is one of the optimum values of how many classes / objects sampled in one training example

export TRANSFORMERS_OFFLINE=1
export DS_SKIP_CUDA_CHECK=1

deepspeed --master_port=24999  --master_port=24989 main.py \
  --segmentation_model_path="/gemini/space/zhaozy/libingyu/LISA-main/pretrained_weights/sam_vit_h_4b8939.pth" \
  --mllm_model_path="weights" \
  --vision-tower="/gemini/space/zhaozy/libingyu/LISA-main-new/pretrained_weights/clip-vit-large-patch14" \
  --dataset_dir="/gemini/space/zhaozy/libingyu/ChangeRef_datasets" \
  --change_refer_seg_data "TV|train" \
  --val_dataset "TV|val" \
  --weight="/gemini/space/zhaozy/libingyu/GSVA-main/pretrained_weights/gsva_7b/gsva-7b-pt.bin" \
  --precision="bf16"\
  --lora_r=8 \
  --exp_name "gsva-bitemporal-run-train-val" \
  --log_base_dir "run_7B"

# # ============================================================
export TRANSFORMERS_OFFLINE=1
export DS_SKIP_CUDA_CHECK=1

deepspeed --master_port=24999  --master_port=24989 main.py \
  --segmentation_model_path="/gemini/space/zhaozy/libingyu/LISA-main/pretrained_weights/sam_vit_h_4b8939.pth" \
  --mllm_model_path="weights" \
  --vision-tower="/gemini/space/zhaozy/libingyu/LISA-main-new/pretrained_weights/clip-vit-large-patch14" \
  --dataset_dir="/gemini/space/zhaozy/libingyu/ChangeRef_datasets" \
  --change_refer_seg_data "TV|NS" \
  --val_dataset "TV|RS" \
  --weight="/gemini/space/zhaozy/libingyu/GSVA-main/pretrained_weights/gsva_7b/gsva-7b-pt.bin" \
  --precision="bf16"\
  --lora_r=8 \
  --exp_name "gsva-bitemporal-run-NS-RS" \
  --log_base_dir "run_7B" \



# # ============================================================
export TRANSFORMERS_OFFLINE=1
export DS_SKIP_CUDA_CHECK=1

deepspeed --master_port=24999  --master_port=24989 main.py \
  --segmentation_model_path="/gemini/space/zhaozy/libingyu/LISA-main/pretrained_weights/sam_vit_h_4b8939.pth" \
  --mllm_model_path="weights" \
  --vision-tower="/gemini/space/zhaozy/libingyu/LISA-main-new/pretrained_weights/clip-vit-large-patch14" \
  --dataset_dir="/gemini/space/zhaozy/libingyu/ChangeRef_datasets" \
  --change_refer_seg_data "TV|RS" \
  --val_dataset "TV|NS" \
  --weight="/gemini/space/zhaozy/libingyu/GSVA-main/pretrained_weights/gsva_7b/gsva-7b-pt.bin" \
  --precision="bf16"\
  --lora_r=8 \
  --exp_name "gsva-bitemporal-run-RS-NS" \
  --log_base_dir "run_7B"