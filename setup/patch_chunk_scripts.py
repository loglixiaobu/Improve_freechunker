#!/usr/bin/env python3
"""批量修补 chunk 脚本 + baseline：把 HF 联网 / 作者机器上的绝对路径换成项目内本地路径。

要改的四件事：
  1. 数据集从 `load_dataset('zai-org/LongBench-v2'[, dn])` 换成
     本地 `datasets/LongBench-v2/split_by_domain/` 的 `load_from_disk`
     （HF 上只有一个 data.json，没有 6 个 config；`load_dataset(dir, cfg)` 也会报
     BuilderConfig not found —— 必须逐域 load_from_disk）
  2. Qwen2-1.5B-Instruct → 项目内 models/Qwen2-1.5B-Instruct（PPL / Margin 用）
  3. baseline/lumberchunker.py 里硬编码的
     `/share/home/ecnuzwx/UnifiedRAG/cache/models--Qwen--Qwen3-8B`
     → 项目内 models/Qwen3-8B
  4. baseline/semantic_chunking.py 的默认 bge-m3 绝对路径 → 项目内 models/bge-m3

幂等：已经改过的文件再跑一遍不会重复插入（看 MARK / 看目标串是否还在）。
"""
from __future__ import annotations

import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TP = os.path.join(ROOT, "train and preprocess")
BP = os.path.join(ROOT, "baseline")

SYS_PATH_SNIPPET = (
    "import sys as _sys, os as _os\n"
    "_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))\n"
    "from src.paths import LONGBENCH_V2 as _LB, MODELS as _MODELS\n"
    "_SPLIT_DIR = _os.path.join(_LB, 'split_by_domain')\n"
)

MARK = "# --- fc-patch: local paths ---"


def _backup(path):
    bak = path + ".orig"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)


def patch_dir_script(fname: str):
    """打补丁：train and preprocess/ 下那 4 个 chunker。"""
    path = os.path.join(TP, fname)
    if not os.path.exists(path):
        print(f"[miss] {fname}")
        return False
    src = open(path, encoding="utf-8").read()
    if MARK in src:
        print(f"[skip] {fname} 已打过补丁")
        return False
    orig = src

    src = src.replace(
        "from datasets import load_dataset, Dataset, DatasetDict",
        "from datasets import load_from_disk, Dataset, DatasetDict",
    )

    # domains[dn] = load_dataset(dataset_path, dn)  ->  load_from_disk(join(dir, dn))
    src = re.sub(
        r"domains\[(\w+)\] = load_dataset\(dataset_path,\s*(\w+)\)",
        r"domains[\1] = load_from_disk(_os.path.join(dataset_path, \2))",
        src,
    )
    src = re.sub(
        r"domains\[(\w+)\] = load_dataset\(dataset_path,\s*'([\w-]+)'\)",
        r"domains[\1] = load_from_disk(_os.path.join(dataset_path, '\2'))",
        src,
    )

    # ppl 的整包加载 -> DatasetDict
    src = src.replace(
        "datasets = load_dataset('zai-org/LongBench-v2')",
        "datasets = DatasetDict({d: load_from_disk(_os.path.join(_SPLIT_DIR, d))\n"
        "                        for d in sorted(_os.listdir(_SPLIT_DIR)) if _os.path.isdir(_os.path.join(_SPLIT_DIR, d))})",
    )

    src = src.replace("dataset_path = 'zai-org/LongBench-v2'", "dataset_path = _SPLIT_DIR")

    # Qwen2-1.5B-Instruct -> 本地目录
    src = src.replace(
        "model_name_or_path = f'Qwen/Qwen2-1.5B-Instruct'",
        "_QWEN2_LOCAL = _os.path.join(_MODELS, 'Qwen2-1.5B-Instruct')\n"
        "model_name_or_path = _QWEN2_LOCAL if _os.path.isdir(_QWEN2_LOCAL) else 'Qwen/Qwen2-1.5B-Instruct'",
    )

    # 顶部插入 sys.path + 常量
    lines = src.split("\n")
    ins = 0
    for i, ln in enumerate(lines[:40]):
        if ln.startswith("import ") or ln.startswith("from "):
            ins = i + 1
    lines[ins:ins] = [MARK] + SYS_PATH_SNIPPET.rstrip("\n").split("\n")
    src = "\n".join(lines)

    if src == orig:
        print(f"[warn] {fname} 无改动 —— 检查模式匹配")
        return False
    _backup(path)
    open(path, "w", encoding="utf-8").write(src)
    print(f"[ok] 已修补 {fname}")
    return True


def patch_baseline_abs_paths():
    """baseline/ 下两处作者机器上的绝对路径。"""
    changed = []

    p = os.path.join(BP, "lumberchunker.py")
    src = open(p, encoding="utf-8").read()
    if "/share/home/ecnuzwx/UnifiedRAG/cache/models--Qwen--Qwen3-8B" in src:
        _backup(p)
        src = src.replace(
            'tokenizer = AutoTokenizer.from_pretrained("/share/home/ecnuzwx/UnifiedRAG/cache/models--Qwen--Qwen3-8B")',
            "# fc-patch: 本地 Qwen3-8B 目录\n"
            "_TK_LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'Qwen3-8B')\n"
            "tokenizer = AutoTokenizer.from_pretrained(_TK_LOCAL if os.path.isdir(_TK_LOCAL) else 'Qwen/Qwen3-8B')",
        )
        # 确保 os 已 import
        if not re.search(r"^import os$", src, re.M):
            src = src.replace("import time\n", "import os\nimport time\n", 1)
        open(p, "w", encoding="utf-8").write(src)
        changed.append("baseline/lumberchunker.py")
        print("[ok] 已修补 baseline/lumberchunker.py 的 tokenizer 路径")
    else:
        print("[skip] baseline/lumberchunker.py 无需修补")

    p = os.path.join(BP, "semantic_chunking.py")
    src = open(p, encoding="utf-8").read()
    if "/share/home/ecnuzwx/UnifiedRAG/cache/models--BAAI--bge-m3" in src:
        _backup(p)
        src = src.replace(
            '"/share/home/ecnuzwx/UnifiedRAG/cache/models--BAAI--bge-m3"',
            "_BGE_LOCAL",
        )
        src = (
            "import os as _os\n"
            "_BGE_LOCAL = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'models', 'bge-m3')\n"
            + src
        )
        open(p, "w", encoding="utf-8").write(src)
        changed.append("baseline/semantic_chunking.py")
        print("[ok] 已修补 baseline/semantic_chunking.py 的 bge-m3 默认路径")
    else:
        print("[skip] baseline/semantic_chunking.py 无需修补")

    return changed


def main():
    for t in ["5_chunk_lumber.py", "6_chunk_ppl.py", "7_chunk_semantic.py",
              "8_chunk_margin.py", "10_chunk_dense_x.py"]:
        patch_dir_script(t)
    patch_baseline_abs_paths()
    print("\nPATCH_DONE")


if __name__ == "__main__":
    main()
