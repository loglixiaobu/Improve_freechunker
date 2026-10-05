#!/usr/bin/env python3
"""探针：验证 FreeChunker 的 vector_store 不变量，并测出单卡能装多少句子。

要验证的事（**必须实测，不能猜**）
---------------------------------
`UnifiedEncoder.query()` 里做的是
    similarities = np.dot(vector_store['embeddings'], query_embedding)
    ...
    grouped_text = vector_store['grouped_texts'][idx]

→ 所以 `embeddings` 的第 0 维必须 == `len(grouped_texts)`，否则索引会错位/越界。
`build_vector_store` 里 `grouped_texts = sentences + groups`，
而 `len(embeddings)` 打印出来是 `len(groups)` —— 看着**对不上**。
必须实测确认到底是哪个。

顺带测：句子数 N 与显存的关系（决定窗口该开多大）。

用法:
    CUDA_VISIBLE_DEVICES=2 python setup/probe_freechunker_window.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.encoder import UnifiedEncoder  # noqa: E402

MODEL = os.path.join(ROOT, "models", "FreeChunk-jina")

enc = UnifiedEncoder(model_name="jina", local_model_path=MODEL,
                     granularities=(2, 4))

text = ("This is a test sentence about chunking long documents. " * 40).strip()
print(f"\n输入句子数（splitter 视角）: {len(enc.sentenceizer.split(text))}")

enc.build_vector_store(text, show_progress=True)

vs = enc.vector_store
n_sent = len(vs["sentences"])
n_grp = len(vs["grouped_sentences"])
n_txt = len(vs["grouped_texts"])
n_emb = vs["embeddings"].shape[0]
dim = vs["embeddings"].shape[1]

print("\n=== vector_store 形状 ===")
print(f"  sentences       = {n_sent}")
print(f"  grouped_sentences = {n_grp}")
print(f"  grouped_texts   = {n_txt}")
print(f"  embeddings      = {n_emb} x {dim}")
print(f"  n_sent + n_grp  = {n_sent + n_grp}")

ok = (n_emb == n_txt)
print(f"\n  不变量 len(embeddings) == len(grouped_texts): {ok}  "
      f"({'✅' if ok else '❌ 索引会错位！'})")

# query 是否可用
try:
    r = enc.query("What is this about?", topk=3)
    print(f"  query(topk=3) -> {len(r)} 段, 首段 {r[0][:60]!r}")
except Exception as e:
    print(f"  query 失败: {type(e).__name__}: {e}")

# 单卡能装多少句：逐步加长，看显存/是否 OOM
print("\n=== 句子数 vs 显存 ===")
if torch.cuda.is_available():
    free0, total = torch.cuda.mem_get_info()
    print(f"  卡上可用 {free0/2**30:.1f} / {total/2**30:.1f} GiB")
for n in (200, 1000, 2000, 4000, 8000):
    t = ("A sentence for memory probing about retrieval augmented reading. " * n)
    try:
        torch.cuda.reset_peak_memory_stats()
        enc.build_vector_store(t, show_progress=False)
        peak = torch.cuda.max_memory_allocated() / 2**30
        real = len(enc.sentenceizer.split(t))
        print(f"  句子≈{real:>6}: 峰值 {peak:6.2f} GiB")
    except torch.OutOfMemoryError:
        print(f"  句子≈{n * 9:>6}: OOM")
        torch.cuda.empty_cache()
        break
    except Exception as e:
        print(f"  句子≈{n * 9:>6}: {type(e).__name__}: {str(e)[:80]}")
        break
