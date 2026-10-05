#!/usr/bin/env python3
"""修 aggregate_eval.py 里「Traditional 被静默丢弃」的 bug。

症状
----
汇总表里**只有 FreeChunker**，traditional_256 / traditional_512 的 18 个 raw 文件
明明读进来了，却不出现在输出里。

根因
----
`parse_tag()` 把 `raw_traditional_256_...` 解析成
    key = f"{mkey}{size}" = "traditional256"
而筛选方法列表用的是
    methods = [k for k in METHOD_ORDER if any(a[0] == k for a in acc)]
    METHOD_ORDER = ["freechunker", "traditional", "ppl", ...]
`"traditional256" == "traditional"` 为 False → **整个方法被排除**，
而且**不报任何错**（既不打印警告，也不影响退出码）。

顺带：`METHOD_LABEL` 里写的是 `'traditional': 'Traditional(256)'`，
说明作者本意是 size=256 用无后缀 key，但 `parse_tag` 实际加了后缀 —— 两处不一致。

修法
----
以 `parse_tag` 的实际产物为准（`traditional256` / `traditional512`），
把 `METHOD_ORDER` 与 `METHOD_LABEL` 对齐，让二者一致。

幂等：已修过会跳过。
"""
from __future__ import annotations

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "setup", "aggregate_eval.py")

src = open(P, encoding="utf-8").read()

if "traditional256" in src:
    print("[patch] 已修过，跳过")
    sys.exit(0)

old_order = '''METHOD_ORDER = ["freechunker", "traditional", "ppl", "margin", "semantic", "lumber"]'''
new_order = '''# ⚠️ 必须与 parse_tag() 的实际产物一致：
#    raw_traditional_256_... -> key "traditional256"
#    raw_traditional_512_... -> key "traditional512"
# 早期这里写的是 "traditional"，导致 `a[0] == k` 永远不成立，
# **Traditional 被静默排除**（不报错、不影响退出码）。
METHOD_ORDER = ["freechunker", "traditional256", "traditional512",
                "ppl", "margin", "semantic", "lumber"]'''
assert old_order in src, "找不到 METHOD_ORDER"
src = src.replace(old_order, new_order, 1)

old_lab = '''    "traditional": "Traditional(256)",
    "traditional512": "Traditional(512)",'''
new_lab = '''    "traditional256": "Traditional(256)",
    "traditional512": "Traditional(512)",'''
assert old_lab in src, "找不到 METHOD_LABEL"
src = src.replace(old_lab, new_lab, 1)

ast.parse(src)
open(P, "w", encoding="utf-8").write(src)
print("[patch] ✅ 已修 aggregate_eval.py：traditional256/512 现在会被纳入汇总")
