from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="kkk2026/gsva-7b-pt",
    repo_type="model",
    local_dir="pretrained_weights/gsva_7b",
    local_dir_use_symlinks=False
)
# https://huggingface.co/kkk2026/gsva-7b-pt/tree/main
# kkk2026/ChangeRef_zip