#!/usr/bin/env python3
"""统一修 `6_chunk_ppl.py` / `8_chunk_margin.py` 的分片维度：域 → 样本索引。

为什么
------
两个脚本都有一段：
    picked = [d for d in picked if d in _CM_DOMAINS]
    if _CM_NSHARDS > 1:
        picked = picked[_CM_SHARD::_CM_NSHARDS]
「先按域过滤、再按域切片」在**只有一个域**时会退化：
    ['multi-doc'][0::3] = ['multi-doc']
    ['multi-doc'][1::3] = []
    ['multi-doc'][2::3] = []
后两个 worker 拿空列表 → 空转；配合 `export` 作用域 bug（见 4.1），
三个进程还可能全跑成 shard0，重复计算同一批。

改法
----
- `_cm_pick` 只做域过滤，**不切片**；
- 新增 `_cm_keep_idx(n)`：`set(range(shard, n, nshards))`，在各域**内部**按样本挑；
- 空域不落盘（避免 N 个 worker 抢写同一个 `dataset_dict.json`）。

两个脚本的循环写法不同，分别适配：
- PPL:     `for i, sample in enumerate(task_dataset):`  （`_done = _cm_load_done(task_name)` 之后）
- Margin:  `for i, sample in enumerate(tqdm(task_dataset, ...)):`
"""
from __future__ import annotations

import io
import os
import py_compile
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = [
    os.path.join(BASE, "train and preprocess", "6_chunk_ppl.py"),
    os.path.join(BASE, "train and preprocess", "8_chunk_margin.py"),
]
MARKER = "def _cm_keep_idx("

NEW_PICK = '''def _cm_pick(domains_order):
    """只按 FC_CM_DOMAINS 过滤；分片在**样本级**做（见 _cm_keep_idx）。

    单域切 N 片时，若在这里按域切片，只有 shard0 拿得到域，
    其余拿到 []，直接空转。
    """
    if _CM_DOMAINS:
        return [d for d in domains_order if d in _CM_DOMAINS]
    return list(domains_order)


def _cm_keep_idx(n):
    """本分片在长度 n 的域里应取哪些下标。"""
    if _CM_NSHARDS <= 1:
        return None
    return set(range(_CM_SHARD, n, _CM_NSHARDS))
'''


def patch(path: str) -> int:
    src = io.open(path, encoding="utf-8").read()
    name = os.path.basename(path)

    if MARKER in src:
        print(f"[skip] {name} 已修")
        return 0

    lines = src.split("\n")

    # --- 1) 替换 _cm_pick 块（从 def _cm_pick( 到下一个顶层语句） ---
    s = e = None
    for i, l in enumerate(lines):
        if l.startswith("def _cm_pick("):
            s = i
        elif s is not None and l and not l[0].isspace() and not l.startswith("def _cm_pick"):
            e = i
            break
    if s is None or e is None:
        print(f"[FAIL] {name}: 定位 _cm_pick 失败 s={s} e={e}", file=sys.stderr)
        return 1
    print(f"[{name}] 替换 line {s+1}..{e}")
    lines[s:e] = NEW_PICK.split("\n")
    src = "\n".join(lines)

    # --- 2) 找循环行，在循环体第一行插 continue 过滤 ---
    ls = src.split("\n")
    loop_idx = None
    for i, l in enumerate(ls):
        if "for i, sample in enumerate(" in l and "task_dataset" in l:
            loop_idx = i
            break
    if loop_idx is None:
        print(f"[FAIL] {name}: 找不到循环行", file=sys.stderr)
        return 1

    indent = ls[loop_idx][: len(ls[loop_idx]) - len(ls[loop_idx].lstrip())]
    body = indent + "    "
    # 全部插在**循环体内部**：_keep 每轮重算，但 n 只是个 len()，开销可忽略。
    # 放循环外会依赖「循环行前面那行的缩进」，PPL 与 Margin 的嵌套层级不同，容易踩空。
    ls.insert(loop_idx + 1, f"{body}if _cm_keep_idx(len(task_dataset)) is not None "
                            f"and i not in _cm_keep_idx(len(task_dataset)):")
    ls.insert(loop_idx + 2, f"{body}    continue")
    src = "\n".join(ls)
    print(f"[{name}] 在 line {loop_idx+2} 插入循环内样本级过滤")

    io.open(path, "w", encoding="utf-8").write(src)
    py_compile.compile(path, doraise=True)
    return 0


def main() -> int:
    rc = 0
    for t in TARGETS:
        if not os.path.isfile(t):
            print(f"[warn] 不存在：{t}", file=sys.stderr)
            continue
        rc |= patch(t)
    print("CM_SAMPLE_SHARD_PATCH_OK" if rc == 0 else "CM_SAMPLE_SHARD_PATCH_FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
