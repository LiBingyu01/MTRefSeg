
# CUDA_VISIBLE_DEVICES=0 python batch_chat_visualize.py \
#   --mllm_model_path '/gemini/space/zhaozy/libingyu/GSVA-main/weights_13B' \
#   --weight '/gemini/space/zhaozy/libingyu/GSVA-main/run_13B/gsva-bitemporal-run-train_val/GSVA_bi_fusion_13B_train_val.bin' \
#   --vision-tower="/gemini/space/zhaozy/libingyu/LISA-main-new/pretrained_weights/clip-vit-large-patch14" \
#   --segmentation_model_path '/gemini/space/zhaozy/libingyu/GSVA-main/sam_vit_h_4b8939.pth' \
#   --precision bf16 \
#   --json_dir '/gemini/space/zhaozy/libingyu/ChangeRef_datasets/TV/val/referring_expression' \
#   --image_dir_A '/gemini/space/zhaozy/libingyu/ChangeRef_datasets/TV/val/A' \
#   --image_dir_B '/gemini/space/zhaozy/libingyu/ChangeRef_datasets/TV/val/B' \
#   --mask_root '/gemini/space/zhaozy/libingyu/ChangeRef_datasets/TV/val/masks' \
#   --output_dir './vis_GSVA_output_batch_13B_val' \
#   --save_canvas \
#   --save_gt

CUDA_VISIBLE_DEVICES=1 python batch_chat_visualize.py \
  --mllm_model_path '/gemini/space/zhaozy/libingyu/GSVA-main/weights_13b' \
  --weight '/gemini/space/zhaozy/libingyu/GSVA-main/run_13B/gsva-bitemporal-run-train_val/GSVA_bi_fusion_13B_train_val.bin/GSVA_bi_fusion_13B_train_val.bin' \
  --vision-tower="/gemini/space/zhaozy/libingyu/LISA-main-new/pretrained_weights/clip-vit-large-patch14" \
  --segmentation_model_path '/gemini/space/zhaozy/libingyu/GSVA-main/sam_vit_h_4b8939.pth' \
  --precision bf16 \
  --lora_r=64 \
  --json_dir '/gemini/space/zhaozy/libingyu/ChangeRef_datasets/TV/val/referring_expression' \
  --image_dir_A '/gemini/space/zhaozy/libingyu/ChangeRef_datasets/TV/val/A' \
  --image_dir_B '/gemini/space/zhaozy/libingyu/ChangeRef_datasets/TV/val/B' \
  --mask_root '/gemini/space/zhaozy/libingyu/ChangeRef_datasets/TV/val/masks' \
  --output_dir './vis_GSVA_output_batch_13B_val' \
  --save_canvas \
  --save_gt