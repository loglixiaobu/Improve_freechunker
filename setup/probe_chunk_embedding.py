#!/usr/bin/env python3
"""探针：定位「给 chunks 算 embedding」为什么会 OOM。

背景
----
2026-10-05 重跑里 ppl / margin / semantic / lumber 大量 worker 报
    test_ppl.py:173  chunk_embeddings = self.embedding_model.encode(chunks)
    → torch.OutOfMemoryError: Tried to allocate 8.00 GiB（本进程已占 18.38 GiB）
**不是挤卡**（报错里没有别人的 pid），是单进程自己超了 24 GB。

假设：某些 chunk 远长于 embedding 模型的 max_seq_length(8192)，
而 `SentenceTransformer.encode` **默认不做截断**，导致注意力 O(N²) 爆炸。

本探针要实测：
  1. 各方法各域的 chunk 长度分布（有多少条超 8192 token）
  2. 对超长文本调用 encode 是否真的爆显存
  3. `SentenceTransformer.encode` 到底截不截断

用法:
    CUDA_VISIBLE_DEVICES=2 python setup/probe_chunk_embedding.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

JINA = os.path.join(ROOT, "models", "jina-embeddings-v2-small-en")
DOMAINS = ["single-doc", "multi-doc", "code-repo", "long-dialogue",
           "long-icl", "long-structured"]
ROOTS = {
    "Traditional/256": ("Traditional", "256"),
    "PPL": ("PPL", "Qwen2.5-1.5B-Instruct"),
    "Margin": ("Margin", "Qwen2.5-1.5B-Instruct"),
    "Lumber": ("Lumber", "qwen3-8b"),
    "Semantic": ("Semantic", "bge-m3"),
}

print("=== 1) chunk 字符长度分布（超长才是凶手）===")
print(f"{'方法':<16}{'域':<16}{'chunk数中位':>11}{'最长chunk':>12}{'>30000字符':>11}")
worst = None
for label, (sub, model) in ROOTS.items():
    for dn in DOMAINS:
        p = os.path.join(ROOT, "LongBench-v2_chunked", sub, model, dn)
        if not os.path.isdir(p):
            continue
        ds = load_from_disk(p)
        lens = []
        for i in range(len(ds)):
            ch = ds[i].get("chunks") or []
            if ch:
                lens.append(max(len(c) for c in ch))
        if not lens:
            continue
        lens.sort()
        med = lens[len(lens) // 2]
        mx = lens[-1]
        over = sum(1 for x in lens if x > 30000)
        print(f"{label:<16}{dn:<16}{med:>11,}{mx:>12,}{over:>11}")
        if worst is None or mx > worst[0]:
            worst = (mx, label, dn, p)

print(f"\n最长的 chunk 出现在: {worst[1]}/{worst[2]}  ({worst[0]:,} 字符)")

print("\n=== 2) encode 到底截不截断？===")
m = SentenceTransformer(JINA)
print(f"  max_seq_length = {m.max_seq_length}")
print(f"  tokenizer.model_max_length = {m.tokenizer.model_max_length}")
for n in (2000, 20000, 200000):
    t = "hello world this is a test sentence. " * (n // 36)
    ids = m.tokenizer(t)["input_ids"]
    print(f"  原文 {len(t):>8,} 字符 → tokenizer 直接给 {len(ids):>8,} 个 id"
          f"（{'未截断 ⚠️' if len(ids) > m.max_seq_length else '已截断'}）")

print("\n=== 3) 实测：encode 超长文本的显存 ===")
torch.cuda.reset_peak_memory_stats()
for n_chars in (50_000, 200_000, 800_000):
    t = "hello world this is a test sentence. " * (n_chars // 36)
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        e = m.encode([t], batch_size=1, show_progress_bar=False,
                     convert_to_numpy=True)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"  {len(t):>9,} 字符: 峰值 {peak:7.2f} GiB  emb={e.shape}")
        del e
    except torch.OutOfMemoryError:
        print(f"  {len(t):>9,} 字符: **OOM**")
        torch.cuda.empty_cache()
        break
    except Exception as ex:
        print(f"  {len(t):>9,} 字符: {type(ex).__name__}: {str(ex)[:60]}")
        torch.cuda.empty_cache()
        break
