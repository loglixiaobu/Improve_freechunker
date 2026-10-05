#!/usr/bin/env python3
"""把 5 个 test_*.py 里「给 chunks 算 embedding」的 batch_size 降下来。

问题（2026-10-05 实测）
----------------------
`chunk_embeddings = self.embedding_model.encode(chunks)` 用的是包装类的默认
batch_size（ppl=4, traditional/lumber/semantic=6, margin=4）。

`SentenceTransformer.encode` 会把一个 batch 里的文本 **pad 到该 batch 最长的那条**，
注意力矩阵是 `batch × heads × L²`。当 batch 里出现 8192 token（模型上限）的长 chunk：

    batch=4 × 8 heads × 8192² × 4 B ≈ 8.6 GB     ← 与实测报错 "Tried to allocate 8.00 GiB" 吻合
    batch=6 × 8 heads × 8192² × 4 B ≈ 12.9 GB

而 PPL / Lumber / Semantic 的 chunk 本来就长（实测 PPL/code-repo 最长 chunk
28 万字符，Lumber/code-repo 中位最大 chunk 高达 161 万字符），
所以 batch 里全是长 chunk 时，单次分配就能吃掉半张卡 → OOM。

实测显存轨迹（ppl × single-doc，limit=60）：
    3809 MiB → 10013 MiB → 17111 MiB → OOM
是**阶梯式跳升**，每次跳升都对应一个"长 chunk 扎堆"的 batch。

修法
----
把包装类的默认 batch_size 改成可配置，默认 **1**：
    batch=1 × 8 × 8192² × 4 ≈ 2.1 GB   ← 安全

代价：embedding 变慢（原来是 4~6 并发）。但 embedding 只占每题 3–5 s 的一部分，
换来的是不再整域归零，划算。

调用点 `encode([question], batch_size=1)` 不受影响（本来就传了 1）。

幂等：已改过会跳过。
"""
from __future__ import annotations

import ast
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 匹配 `def encode(self, texts, batch_size=4):`（数值不同也认）
pat = re.compile(
    r"def encode\(self,\s*texts,\s*batch_size\s*=\s*\d+\s*\):")

NEW = ("def encode(self, texts, batch_size=None):\n"
       "        # batch 里若混入 8192-token 的长 chunk，注意力会按 batch 线性放大：\n"
       "        #   batch=6 → 6 × 8 heads × 8192² × 4B ≈ 12.9 GB（直接吃掉半张 3090）\n"
       "        # 默认降到 1，可用环境变量 FC_EMB_BATCH 调大。\n"
       "        if batch_size is None:\n"
       "            batch_size = int(os.environ.get('FC_EMB_BATCH', '1'))")

changed, skipped = [], []
for p in sorted(glob.glob(os.path.join(ROOT, "test", "test_*.py"))):
    src = open(p, encoding="utf-8").read()
    if "FC_EMB_BATCH" in src:
        skipped.append(os.path.basename(p) + "(已改)")
        continue
    if not pat.search(src):
        skipped.append(os.path.basename(p) + "(无 encode 包装)")
        continue
    if not re.search(r"^\s*import os\s*$", src, re.M):
        m = re.search(r"^(import \S+|from \S+ import .*)$", src, re.M)
        if m:
            src = src[:m.start()] + "import os\n" + src[m.start():]
        else:
            skipped.append(os.path.basename(p) + "(无法插 import os)")
            continue
    src = pat.sub(NEW, src, count=1)
    ast.parse(src)
    open(p, "w", encoding="utf-8").write(src)
    changed.append(os.path.basename(p))

print(f"[patch] 已改: {changed}")
print(f"[patch] 跳过: {skipped}")
for p in sorted(glob.glob(os.path.join(ROOT, "test", "test_*.py"))):
    src = open(p, encoding="utf-8").read()
    m = re.search(r"def encode\(self,\s*texts,\s*batch_size\s*=\s*(\S+?)\):", src)
    print(f"  {os.path.basename(p):28s} batch_size={m.group(1) if m else '(无)'}")
