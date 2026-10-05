#!/usr/bin/env python3
"""探针 v2：用**真实 splitter + 真实数据**测 FreeChunker 的句子数分布与显存拐点。

为什么重写
----------
v1 用 `"A sentence..." * n` 造数据，结果真实 splitter 把 200 个重复短句
**合并成 8 个**——探针根本没压到模型。
必须用真实 LongBench 样本 + 真实 `sentenceizer.split()`。

要回答两个问题
--------------
1. 各域的句子数 N 分布到底是多少（这决定窗口开多大、多少样本会被切）
2. N 多大时显存爆掉（找安全窗口）

用法:
    CUDA_VISIBLE_DEVICES=2 python setup/probe_freechunker_window2.py
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402

from src.encoder import UnifiedEncoder  # noqa: E402

MODEL = os.path.join(ROOT, "models", "FreeChunk-jina")
DOMAINS = ["single-doc", "multi-doc", "code-repo", "long-dialogue",
           "long-icl", "long-structured"]

enc = UnifiedEncoder(model_name="jina", local_model_path=MODEL,
                     granularities=(2, 4))
splitter = enc.sentenceizer

print("\n=== 各域「真实 splitter 句子数」分布 ===")
sizes = {}
for dn in DOMAINS:
    ds = load_from_disk(os.path.join(ROOT, "datasets", "LongBench-v2",
                                     "split_by_domain", dn))
    ns = []
    for i in range(len(ds)):
        try:
            ns.append(len(splitter.split(ds[i]["context"])))
        except Exception as e:
            ns.append(-1)
    ns = [x for x in ns if x > 0]
    ns.sort()
    sizes[dn] = ns
    over = {t: sum(1 for x in ns if x > t) for t in (2000, 4000, 8000)}
    print(f"  {dn:16s} n={len(ns):3d}  中位={ns[len(ns)//2]:>7,}  "
          f"最大={ns[-1]:>8,}  超2000={over[2000]:3d} 超4000={over[4000]:3d} "
          f"超8000={over[8000]:3d}")

# 找全局最大的样本，做显存拐点测试
big_dn, big_i, big_n = None, None, -1
for dn in DOMAINS:
    ds = load_from_disk(os.path.join(ROOT, "datasets", "LongBench-v2",
                                     "split_by_domain", dn))
    for i in range(len(ds)):
        n = len(splitter.split(ds[i]["context"]))
        if n > big_n:
            big_dn, big_i, big_n = dn, i, n
print(f"\n最大样本: {big_dn}[{big_i}] 句子数={big_n:,}")

ds = load_from_disk(os.path.join(ROOT, "datasets", "LongBench-v2",
                                 "split_by_domain", big_dn))
sents = splitter.split(ds[big_i]["context"])

print("\n=== 显存拐点（用该样本前 N 句）===")
torch.cuda.reset_peak_memory_stats()
base = torch.cuda.memory_allocated() / 2**30
print(f"  模型常驻 {base:.2f} GiB")
for N in (1000, 2000, 4000, 6000, 8000, 12000, 16000):
    if N > len(sents):
        break
    txt = " ".join(sents[:N])
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        t0 = time.time()
        enc.build_vector_store(txt, show_progress=False)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"  N={N:>6,}: 峰值 {peak:6.2f} GiB  用时 {time.time()-t0:5.1f}s")
    except torch.OutOfMemoryError:
        print(f"  N={N:>6,}: **OOM**")
        torch.cuda.empty_cache()
        break
    except Exception as e:
        print(f"  N={N:>6,}: {type(e).__name__}: {str(e)[:70]}")
        torch.cuda.empty_cache()
        break
