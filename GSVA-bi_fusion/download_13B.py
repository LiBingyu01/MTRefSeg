from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="rjadr/LLaVA-13B-v1-1",
    repo_type="model",
    local_dir="weights_13b",
    local_dir_use_symlinks=False
)

# kkk2026/ChangeRef_zip