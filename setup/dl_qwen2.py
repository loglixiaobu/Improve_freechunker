#!/usr/bin/env python3
"""下载 Qwen2-1.5B-Instruct 到项目内 models/（PPL / Margin chunker 依赖）。"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from huggingface_hub import snapshot_download

DEST = "/home1/lh/FreeChunker/models/Qwen2-1.5B-Instruct"

print("downloading Qwen/Qwen2-1.5B-Instruct ->", DEST, flush=True)
snapshot_download(
    repo_id="Qwen/Qwen2-1.5B-Instruct",
    local_dir=DEST,
    ignore_patterns=[".gitattributes", ".DS_Store", "**/.DS_Store"],
)
print("DL_DONE", flush=True)
