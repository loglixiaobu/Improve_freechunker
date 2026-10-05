#!/usr/bin/env python3
"""离线验证评测分片口径：build_tasks × _apply_keep 合起来必须恰好切一次。

为什么需要这个
--------------
`setup/run_eval_parallel.py` 和 `test/eval_downstream.py::_apply_keep()` **都**会做
`domains[shard::nshards]`。若前者预先切好再传后者，域被切两次：
shard0 拿到 ['single-doc'] 还能用，shard1 拿到 ['multi-doc'] 再 `[1::6]` → 空集，
worker 打印"没领到域，跳过"、静默不产出。整晚只会评出 1/6 的域。

本脚本对多个 nshards 值断言三件事：
  1. 每个 shard 都领到**非空**域集
  2. 各 shard 的域集**并集 == 全部域**
  3. 各 shard 的域集**两两不交**

用法:
    python setup/verify_eval_sharding.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rp = load(os.path.join(ROOT, "setup", "run_eval_parallel.py"), "rp")
ed = load(os.path.join(ROOT, "test", "eval_downstream.py"), "ed")

ALL = rp.ALL_DOMAINS


class FakeQA:
    """只需要 .datasets 有 keys，避免真去 load_from_disk。"""
    def __init__(self):
        self.datasets = {d: [d] * 3 for d in ALL}


ok_all = True
for nshards in (1, 2, 3, 6):
    tasks = rp.build_tasks(["margin"], ALL, nshards, "256,512", 1, "5,10", 0)
    got = {}
    for t in tasks:
        doms = t["domains"].split(",")
        keep = ed._apply_keep(FakeQA(), doms, 0, 0, t["shard"], t["nshards"])
        got[t["shard"]] = sorted(keep.keys())

    union = sorted({d for v in got.values() for d in v})
    empty = [s for s, v in got.items() if not v]
    dup = []
    seen = set()
    for s, v in got.items():
        for d in v:
            if d in seen:
                dup.append(d)
            seen.add(d)

    ok = (not empty) and (union == sorted(ALL)) and (not dup) \
        and len(tasks) == min(nshards, len(ALL))
    ok_all &= ok
    print(f"nshards={nshards}  任务数={len(tasks)}  "
          f"空集shard={empty}  并集完整={union == sorted(ALL)}  重叠={dup}  "
          f"{'✅' if ok else '❌'}")
    for s in sorted(got):
        print(f"    shard{s}: {got[s]}")

print()
print("VERIFY_EVAL_SHARDING_OK" if ok_all else "VERIFY_EVAL_SHARDING_FAIL")
sys.exit(0 if ok_all else 1)
