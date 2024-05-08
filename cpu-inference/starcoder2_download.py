from huggingface_hub import snapshot_download
model_id="bigcode/starcoder2-7b"
snapshot_download(repo_id=model_id, local_dir="starcoder2",
                  local_dir_use_symlinks=False, revision="main")
