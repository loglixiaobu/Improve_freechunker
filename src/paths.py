#!/usr/bin/env python3
"""项目内路径统一定义。

**所有脚本一律从这里取路径**，避免再出现「造数据脚本写到 A、训练脚本读 B」这类不一致。

可用环境变量覆盖（便于换机器 / 换数据盘）：

    FC_ROOT       仓库根目录（默认按本文件位置自动探测）
    FC_MODELS     模型目录
    FC_DATASETS   数据集目录
    FC_VECTOR     编码产物目录
    FC_SAVED      训练产物目录
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))            # <root>/src
ROOT = os.environ.get("FC_ROOT") or os.path.dirname(_HERE)    # <root>

MODELS = os.environ.get("FC_MODELS") or os.path.join(ROOT, "models")
DATASETS = os.environ.get("FC_DATASETS") or os.path.join(ROOT, "datasets")
VECTOR = os.environ.get("FC_VECTOR") or os.path.join(ROOT, "vector")
SAVED = os.environ.get("FC_SAVED") or os.path.join(ROOT, "saved_models")
LOGS = os.environ.get("FC_LOGS") or os.path.join(ROOT, "logs")

# ---------------- 具体资产（全部在项目目录内，不走 HF 缓存） ----------------
JINA_LOCAL = os.path.join(MODELS, "jina-embeddings-v2-small-en")
BGE_M3_LOCAL = os.path.join(MODELS, "bge-m3")
QWEN3_8B_LOCAL = os.path.join(MODELS, "Qwen3-8B")
FREE_CHUNK_JINA = os.path.join(MODELS, "FreeChunk-jina")

CORPUS = os.path.join(DATASETS, "FreeChunk-corpus")
LONGBENCH_V2 = os.path.join(DATASETS, "LongBench-v2")

# ---------------- 训练 / 推理固定口径 ----------------
BACKBONE = "jina-embeddings-v2-small-en"
# 训练粒度：与 src.utils.generate_shifted_matrix 的默认值、官方 config.json 的
# max_power=4 一致。论文正文写的是 (2,4,8,16,32)，但代码与官方权重都是 [2,4]，
# 以代码为准。
GRANULARITIES = (2, 4)

VECTOR_DIR = os.path.join(VECTOR, BACKBONE)   # 编码产物
SAVE_DIR = os.path.join(SAVED, BACKBONE)      # 训练产物

# 官方训练超参（用于验收门 G2）
OFFICIAL_TOTAL_STEPS = 200_000
OFFICIAL_FIRST1000_MEAN = 0.4497
OFFICIAL_LAST1000_MEAN = 0.0109
OFFICIAL_FINAL_VAL = 0.0103


def load_corpus_split(split: str):
    """按 split 子目录加载 FreeChunk 语料。

    本地布局**没有根 dataset_dict.json**，所以：
      * `load_from_disk(CORPUS)`        -> FileNotFoundError
      * `load_dataset(CORPUS, split=..)` -> ValueError（把 val/ 误判成 json split）
    只能按 split 子目录逐个加载。作者脚本能跑，是因为它在 hub 上加载而不是本地。
    """
    from datasets import load_from_disk

    d = os.path.join(CORPUS, "val" if split == "validation" else split)
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"找不到语料 split 目录 {d}\n"
            f"（{CORPUS} 下现有：{sorted(os.listdir(CORPUS)) if os.path.isdir(CORPUS) else '目录不存在'}）"
        )
    return load_from_disk(d)


def load_vector_split(split: str, split_name: str | None = None):
    """加载编码产物（标准 save_to_disk 目录）。"""
    from datasets import load_from_disk

    d = os.path.join(VECTOR_DIR, split_name or split)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"找不到编码产物 {d}，请先跑 1_build_pretrain_datasets.py + 1b_merge_parts.py")
    return load_from_disk(d)


__all__ = [
    "ROOT", "MODELS", "DATASETS", "VECTOR", "SAVED", "LOGS",
    "JINA_LOCAL", "BGE_M3_LOCAL", "QWEN3_8B_LOCAL", "FREE_CHUNK_JINA",
    "CORPUS", "LONGBENCH_V2",
    "BACKBONE", "GRANULARITIES", "VECTOR_DIR", "SAVE_DIR",
    "OFFICIAL_TOTAL_STEPS", "OFFICIAL_FIRST1000_MEAN",
    "OFFICIAL_LAST1000_MEAN", "OFFICIAL_FINAL_VAL",
    "load_corpus_split", "load_vector_split",
]
