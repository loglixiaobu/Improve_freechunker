#!/usr/bin/env python3
"""探针 v3：直接测 FreeChunker 模型对「句子数 N」的显存曲线。

为什么这样测
------------
v1 用重复短句造数据 → 真实 splitter 把 200 句合并成 8 句，没压到模型。
v2 用真实数据 → 光 tokenize 整篇文档就慢到不可接受。

**关键认识**：`UnifiedEncoder.encode()` 里，模型拿到的输入是
    inputs_embeds = sentence_embeddings.unsqueeze(0)      # [1, N, 1024]
句子向量是**外部 sentence transformer 算好的**，模型只关心 N。
所以可以直接造 `torch.randn(1, N, 1024)` 来测模型显存 vs N —— 秒级，
而且测的正是决定窗口大小的那个量。

（注意：正式跑时 `build_vector_store` 还要额外做一遍句子编码，
 那部分是 O(N) 且分 batch，不是瓶颈。）

用法:
    CUDA_VISIBLE_DEVICES=2 python setup/probe_freechunker_window3.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from src.freechunker import FreeChunkerModel  # noqa: E402

MODEL = os.path.join(ROOT, "models", "FreeChunk-jina")
DIM = 1024

model = FreeChunkerModel.from_pretrained(MODEL).to("cuda").eval()

free, total = torch.cuda.mem_get_info()
print(f"卡: {free/2**30:.1f} / {total/2**30:.1f} GiB 可用")
torch.cuda.reset_peak_memory_stats()
print(f"模型常驻: {torch.cuda.memory_allocated()/2**30:.2f} GiB\n")

print(f"{'N':>8}  {'峰值GiB':>9}  {'增量GiB':>9}  备注")
base = torch.cuda.memory_allocated()
last_ok = None
for N in (500, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 24000, 32000):
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        emb = torch.randn(1, N, DIM, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            out = model(inputs_embeds=emb, granularities=(2, 4))
        peak = torch.cuda.max_memory_allocated() / 2**30
        inc = peak - base / 2**30
        sm = out["shift_matrix"].shape
        print(f"{N:>8,}  {peak:>9.2f}  {inc:>9.2f}  shift_matrix={tuple(sm)}")
        del emb, out
        last_ok = N
    except torch.OutOfMemoryError:
        print(f"{N:>8,}  {'—':>9}  {'—':>9}  **OOM**")
        torch.cuda.empty_cache()
        break
    except Exception as e:
        print(f"{N:>8,}  {'—':>9}  {'—':>9}  {type(e).__name__}: {str(e)[:60]}")
        torch.cuda.empty_cache()
        break

print(f"\n最大可用 N ≈ {last_ok:,}")
