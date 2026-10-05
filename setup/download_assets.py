#!/usr/bin/env python3
"""把 FreeChunker 复现所需的模型与数据集全部下载到项目文件夹内。

目录约定：
  ~/FreeChunker/models/<name>/      HuggingFace 模型
  ~/FreeChunker/datasets/<name>/    HuggingFace 数据集
"""
import os
import shutil
import sys
import time
import traceback

ROOT = "/home1/lh/FreeChunker"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from huggingface_hub import snapshot_download

# ⚠️ 必须全局忽略这些：hf-mirror 对 `.DS_Store` 返回 **403**，
# snapshot_download 遇到一个文件 403 会整个 job 抛异常退出
# （实测 FreeChunk-corpus 下了 2.7G/3.6G 后全废）。
DEFAULT_IGNORE = [".DS_Store", "**/.DS_Store", ".gitattributes"]


def _merge_ignore(extra):
    pats = list(DEFAULT_IGNORE)
    for p in (extra or []):
        if p not in pats:
            pats.append(p)
    return pats


# (repo_id, repo_type, 目标目录, 额外 ignore_patterns)
JOBS = [
    ("jinaai/jina-embeddings-v2-small-en", "model",
     "models/jina-embeddings-v2-small-en", None),
    ("XiaSheng/FreeChunk-jina", "model",
     "models/FreeChunk-jina", None),
    ("BAAI/bge-m3", "model",
     "models/bge-m3", ["onnx/*", "*.onnx", "imgs/*", ".gitattributes"]),
    ("XiaSheng/FreeChunk-corpus", "dataset",
     "datasets/FreeChunk-corpus", None),
    ("zai-org/LongBench-v2", "dataset",
     "datasets/LongBench-v2", None),
    # 最大的放最后
    ("Qwen/Qwen3-8B", "model",
     "models/Qwen3-8B", None),
]


def du(path):
    total = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            try:
                total += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return total


def main():
    print("=" * 70)
    print("FreeChunker assets download ->", ROOT)
    print("HF_ENDPOINT =", os.environ["HF_ENDPOINT"])
    print("=" * 70, flush=True)

    ok, failed = [], []
    for repo, rtype, rel, ignore in JOBS:
        dest = os.path.join(ROOT, rel)
        os.makedirs(dest, exist_ok=True)
        print(f"\n>>> [{rtype}] {repo}  ->  {dest}", flush=True)
        t0 = time.time()
        ignore = _merge_ignore(ignore)
        print(f"    ignore_patterns = {ignore}", flush=True)
        try:
            snapshot_download(
                repo_id=repo,
                repo_type=None if rtype == "model" else "dataset",
                local_dir=dest,
                ignore_patterns=ignore,
                max_workers=8,
            )
            sz = du(dest)
            dt = time.time() - t0
            print(f"    DONE  {sz/1e9:.2f} GB in {dt/60:.1f} min "
                  f"({sz/1e6/max(dt,1e-6):.1f} MB/s)", flush=True)
            ok.append((repo, sz))
        except Exception:
            print(f"    FAILED after {time.time()-t0:.0f}s", flush=True)
            traceback.print_exc()
            failed.append(repo)

    print("\n" + "=" * 70)
    print("SUMMARY")
    for repo, sz in ok:
        print(f"  OK    {repo:42s} {sz/1e9:7.2f} GB")
    for repo in failed:
        print(f"  FAIL  {repo}")
    print("=" * 70, flush=True)
    print("ASSETS_DOWNLOAD_DONE" if not failed else "ASSETS_DOWNLOAD_PARTIAL")


if __name__ == "__main__":
    main()
