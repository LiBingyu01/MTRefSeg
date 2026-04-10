
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
   path_or_fileobj="/gemini/space/zhaozy/libingyu/GSVA-main/run_13B/gsva-bitemporal-run-RS-NS/GSVA_bi_fusion_13B_RS_NS.bin/GSVA_bi_fusion_13B_RS_NS.bin", # 本地文件路径
   path_in_repo="GSVA_bi_fusion_13B_RS_NS.bin", # 仓库中的目标路径
   repo_id="kkk2026/gsva-13b-pt", # 仓库名称
   repo_type="model" # 仓库类型 (dataset/model/space)
)

# huggingface-cli upload kkk2026/gsva-13b-pt /gemini/space/zhaozy/libingyu/GSVA-main/gava_13B_fintuning/GSVA_bi_fusion_13B_NS_RS.bin GSVA_bi_fusion_13B_NS_RS.bin

# huggingface-cli upload kkk2026/gsva-13b-pt /gemini/space/zhaozy/libingyu/GSVA-main/gava_13B_fintuning/GSVA_bi_fusion_13B_RS_NS.bin GSVA_bi_fusion_13B_RS_NS.bin

# huggingface-cli upload kkk2026/gsva-13b-pt /gemini/space/zhaozy/libingyu/GSVA-main/gava_13B_fintuning/GSVA_bi_fusion_13B_train_val.bin GSVA_bi_fusion_13B_train_val.bin

# # --- 配置区 ---
# TOKEN = "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"  # 记得换成你新生成的 Token
# REPO_ID = "kkk2026/gsva-13b-pt"  # 例如 "huohuo/coco-train-parquet"
# FOLDER_PATH = r"./gava_13B_fintuning"  # 你存放 Parquet 文件的文件夹
# # --------------

# api = HfApi()

# print(f"开始上传文件夹 {FOLDER_PATH} 到仓库 {REPO_ID}...")

# try:
#     api.upload_folder(
#         folder_path=FOLDER_PATH,
#         repo_id=REPO_ID,
#         repo_type="model",
#         token=TOKEN,
#     )
#     print("上传成功！")
# except Exception as e:
#     print(f"上传过程中出现错误: {e}")