#!/usr/bin/env python3
"""让 `5_chunk_lumber.py` 的域列表可配置（原为硬编码 2 个域）。

问题
----
论文 Table 1 的 LumberChunker 一列**六个域全有**数据：
    Single-Doc / Multi-Doc / Code-Repo / Long-ICL / Long-Dialogue / Long-Structured
但脚本第 98 行写死：
    selected_domains = ['single-doc','multi-doc']
所以只产出 2 个域 —— 这不是论文的设定，是代码的省略。

改法
----
读环境变量 `FC_LMB_DOMAINS`（逗号分隔）；未设时保持原行为，向后兼容。
另外把「已完成」日志里的全局池计数修对（原式 `len(processed_samples) - len(_global_done)`
在全局池与本地不重叠时会算出负数或错数）。
"""
from __future__ import annotations

import io
import os
import py_compile
import sys

TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "train and preprocess", "5_chunk_lumber.py")

OLD = "selected_domains = ['single-doc','multi-doc']"

NEW = """# fc-patch: lumber domains configurable
# 论文 Table 1 的 Lumber 覆盖全部 6 个域，原代码写死 2 个 —— 走环境变量补齐。
_FC_LMB_DOMAINS = [d.strip() for d in _os.environ.get('FC_LMB_DOMAINS', '').split(',') if d.strip()]
selected_domains = _FC_LMB_DOMAINS or ['single-doc', 'multi-doc']
print(f"[lumber] selected_domains = {selected_domains}")"""


def main() -> int:
    src = io.open(TARGET, encoding="utf-8").read()

    if "fc-patch: lumber domains configurable" in src:
        print("已经打过补丁，跳过")
        return 0

    if OLD not in src:
        # 可能已经被改过，找找
        for i, l in enumerate(src.split("\n")):
            if l.startswith("selected_domains"):
                print(f"[patch] 现有第 {i+1} 行：{l!r}", file=sys.stderr)
        print("[patch] 找不到目标行", file=sys.stderr)
        return 1

    src = src.replace(OLD, NEW, 1)

    # 修计数日志
    old_log = "print(f\"[lumber] 本地 checkpoint {len(processed_samples) - len(_global_done)} 条 + 全局池 {len(_global_done)} 条 = 待跳过 {len(processed_samples)} 条\")"
    new_log = ("print(f\"[lumber] 全局池 {len(_global_done)} 条，本 shard 待跳过 {len(processed_samples)} 条\")")
    if old_log in src:
        src = src.replace(old_log, new_log, 1)

    io.open(TARGET, "w", encoding="utf-8").write(src)
    py_compile.compile(TARGET, doraise=True)
    print("LUMBER_DOMAINS_PATCH_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
