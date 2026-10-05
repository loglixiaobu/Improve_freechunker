#!/usr/bin/env python3
"""把 `6_chunk_ppl.py` 的分片维度从「域」改成「样本索引」。

问题（2026-10-04 实测抓到的真 bug）
------------------------------------
`rebalance_ppl.py` 把 multi-doc 劈成 3 路时，三个进程日志全是
`[cm] shard=0/1 domains=['multi-doc']` —— 也就是说

1. **`FC_CM_SHARD` / `FC_CM_NSHARDS` 根本没传进进程**：
   `launch_shard` 里写的是
       `export FC_CM_DOMAINS=x && FC_CM_SHARD=0 FC_CM_NSHARDS=3 && cd ...`
   `export` 只作用于紧跟的第一个赋值；后面的 `FC_CM_SHARD=... FC_CM_NSHARDS=...`
   是**普通 shell 变量**，而且这种 `VAR=v cmd` 前缀赋值是给 `cd` 的，
   不会进 python 的环境。于是三个进程都用默认 `SHARD=0 / NSHARDS=1`。
   （`rebalance_ppl.py` 的 env 已另修。）

2. **就算传进去了，按「域」切片也切不开单域**：
       `['multi-doc'][0::3] = ['multi-doc']`
       `['multi-doc'][1::3] = []`
       `['multi-doc'][2::3] = []`
   后两个 worker 拿到空域，一个样本都不算。

修法
----
- `_cm_pick` 只做 `FC_CM_DOMAINS` 过滤，**不再按域切片**；
- 新增 `_cm_keep_idx(n)`，在**各域内部按样本下标** `range(shard, n, nshards)` 挑；
- 空域/空集**不落盘**（否则 N 个 worker 抢写同一个 `dataset_dict.json`）。
"""
from __future__ import annotations

import io
import os
import py_compile
import sys

TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "train and preprocess", "6_chunk_ppl.py")

NEW_PICK_BLOCK = '''def _cm_pick(domains_order):
    """只按 FC_CM_DOMAINS 过滤；分片在**样本级**做（见 _cm_keep_idx）。

    单域切 3 片时，若在这里按域切片，只有 shard0 拿得到域，
    另两个拿到 []，直接空转 —— 这是三个进程重复算同一批的根因。
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


def main() -> int:
    src = io.open(TARGET, encoding="utf-8").read()
    lines = src.split("\n")

    # 1) 替换 _cm_pick 整块：从 `def _cm_pick(` 到 `root = ` 之前
    s = e = None
    for i, l in enumerate(lines):
        if l.startswith("def _cm_pick("):
            s = i
        elif s is not None and l.startswith("root = "):
            e = i
            break
    if s is None or e is None:
        print("[patch] 定位 _cm_pick 块失败", file=sys.stderr)
        return 1
    print("[patch] 替换 line %d..%d" % (s + 1, e))
    lines[s:e] = NEW_PICK_BLOCK.split("\n")
    src = "\n".join(lines)

    # 2) 在每个域开始处算 _keep
    anchor = "    _done = _cm_load_done(task_name)\n"
    if anchor not in src:
        print("[patch] 找不到 _cm_load_done 锚点", file=sys.stderr)
        return 1
    if "_keep = _cm_keep_idx" not in src:
        src = src.replace(
            anchor,
            anchor + "    _keep = _cm_keep_idx(len(task_dataset))\n",
            1,
        )

    # 3) 循环内按样本下标过滤
    loop = "    for i, sample in enumerate(task_dataset):\n"
    if loop not in src:
        print("[patch] 找不到 for-loop 锚点", file=sys.stderr)
        return 1
    if "_keep is not None and i not in _keep" not in src:
        src = src.replace(
            loop,
            loop
            + "        if _keep is not None and i not in _keep:\n"
            + "            continue\n",
            1,
        )

    # 4) 空集不落盘
    mkds = "    new_datasets[task_name] = Dataset.from_list(task_chunk_results)\n"
    if mkds not in src:
        print("[patch] 找不到 Dataset.from_list 锚点", file=sys.stderr)
        return 1
    if "本分片无样本，跳过落盘" not in src:
        src = src.replace(
            mkds,
            mkds
            + "    if new_datasets[task_name].num_rows == 0:\n"
            + '        print(f"[cm] {task_name} 本分片无样本，跳过落盘")\n'
            + "        continue\n",
            1,
        )

    io.open(TARGET, "w", encoding="utf-8").write(src)
    py_compile.compile(TARGET, doraise=True)
    print("PPL_SAMPLE_SHARD_PATCH_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
