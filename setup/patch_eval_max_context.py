#!/usr/bin/env python3
"""把 6 个 test_*.py 里的 `max_context_length = 40000` 改成可配置。

为什么必须改
------------
- 评测侧：`self.max_context_length = 40000` → 截到 ~39k token 的 prompt
- vLLM 侧：`--max-model-len 16384`
→ 长 prompt 被 vLLM 以 HTTP 400 拒绝，**整个域归零**：
  `You passed 15873 input tokens and requested 512 output tokens.
   However, the model's context length is only 16384`

实测 vLLM 在 gmu=0.92 下的上限是 **36160**；Qwen3-8B 原生上下文是 **32768**。
所以把评测侧上限设成 32768（保留 1000 reserved + 512 输出，仍有余量）。

改成读环境变量 `FC_EVAL_MAX_CONTEXT`（默认 32768），这样不用改代码就能调。

幂等：已改过会跳过。
"""
from __future__ import annotations

import ast
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = 32768

pat = re.compile(r"^(?P<ind>\s*)self\.max_context_length\s*=\s*40000\s*$", re.M)

changed, skipped = [], []
for p in sorted(glob.glob(os.path.join(ROOT, "test", "test_*.py"))):
    src = open(p, encoding="utf-8").read()
    if "FC_EVAL_MAX_CONTEXT" in src:
        skipped.append(os.path.basename(p))
        continue
    if not pat.search(src):
        skipped.append(os.path.basename(p) + "(无 40000)")
        continue
    if not re.search(r"^\s*import os\s*$", src, re.M):
        # 在第一个 import 之后插入 import os
        m = re.search(r"^(import \S+|from \S+ import .*)$", src, re.M)
        if m:
            src = src[:m.start()] + "import os\n" + src[m.start():]
        else:
            skipped.append(os.path.basename(p) + "(无法插 import os)")
            continue
    src = pat.sub(
        lambda m: (f"{m.group('ind')}self.max_context_length = int("
                   f"os.environ.get('FC_EVAL_MAX_CONTEXT', '{DEFAULT}'))"),
        src)
    ast.parse(src)
    open(p, "w", encoding="utf-8").write(src)
    changed.append(os.path.basename(p))

print(f"[patch] 已改: {changed}")
print(f"[patch] 跳过: {skipped}")

# 复查
for p in sorted(glob.glob(os.path.join(ROOT, "test", "test_*.py"))):
    src = open(p, encoding="utf-8").read()
    m = re.search(r"self\.max_context_length\s*=\s*(.+)", src)
    print(f"  {os.path.basename(p):28s} -> {m.group(1).strip() if m else '(无)'}")
