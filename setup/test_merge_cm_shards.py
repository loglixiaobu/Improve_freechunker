#!/usr/bin/env python3
"""在**隔离沙箱**里验证 merge_cm_shards.py v2，不动真实产物。

为什么需要
----------
merge v2 是从「只读目录」改成「目录 + checkpoint 都读」的版本，
还没在真实数据上跑过。夜里无人值守时它要是挂了，早上会一无所有。
所以在真跑之前，用一个假目录把它三种关键路径都覆盖掉：

  A. 目录 + 分片目录 + checkpoint 都有，且**互相重叠** → 应按 _id 去重
  B. **只有本体 checkpoint**、没有任何目录（= code-repo 被 kill 后的状态）
     → v1 会当成"无数据"丢掉，v2 必须能救回来
  C. 已合并态（只有 <dn>/，无分片）→ 应跳过，不重做

用法:
    python setup/test_merge_cm_shards.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB, MODEL = "_mergetest", "M"
BASE = os.path.join(ROOT, "LongBench-v2_chunked", SUB, MODEL)

REC = {"_id": None, "domain": "x", "sub_domain": "s", "difficulty": "easy",
       "length": "short", "question": "q", "choice_A": "a", "choice_B": "b",
       "choice_C": "c", "choice_D": "d", "answer": "A", "chunks": ["c1", "c2"],
       "time": 1.0}


def rec(i):
    r = dict(REC)
    r["_id"] = i
    return r


def write_ckpt(path, recs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build():
    from datasets import Dataset
    if os.path.isdir(BASE):
        shutil.rmtree(BASE)
    ck = os.path.join(BASE, "checkpoints")

    # --- 域 dom：目录 A/B + 分片目录 A + 本体 checkpoint B/C ---
    Dataset.from_list([rec("A"), rec("B")]).save_to_disk(os.path.join(BASE, "dom"))
    Dataset.from_list([rec("A")]).save_to_disk(os.path.join(BASE, "dom.shard0"))
    write_ckpt(os.path.join(ck, "dom.jsonl"), [rec("B"), rec("C")])
    write_ckpt(os.path.join(ck, "dom.shard0.jsonl"), [rec("A")])

    # --- 域 onlyckpt：只有本体 checkpoint（模拟被 kill 的 worker）---
    write_ckpt(os.path.join(ck, "onlyckpt.jsonl"), [rec("X"), rec("Y")])

    # --- 域 merged：已合并态（只有目录）---
    Dataset.from_list([rec("P"), rec("Q"), rec("R")]).save_to_disk(
        os.path.join(BASE, "merged"))


def run_merge():
    spec = importlib.util.spec_from_file_location(
        "mcs", os.path.join(ROOT, "setup", "merge_cm_shards.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    # 把三个测试域塞进 DOMAINS，并把 MODEL_DIR 指到沙箱
    m.DOMAINS = ["dom", "onlyckpt", "merged"]
    m.MODEL_DIR["margin"] = (SUB, MODEL)
    return m.merge_one("margin")


def check():
    from datasets import load_from_disk
    ok = True

    def ids(dn):
        p = os.path.join(BASE, dn)
        if not os.path.isdir(p):
            return None
        return sorted(str(x) for x in load_from_disk(p)["_id"])

    # A: dom 应为 A,B,C（去重）
    got = ids("dom")
    exp = ["A", "B", "C"]
    print(f"  dom      -> {got}  期望 {exp}  {'✅' if got == exp else '❌'}")
    ok &= (got == exp)

    # B: onlyckpt 应为 X,Y（v1 会丢）
    got = ids("onlyckpt")
    exp = ["X", "Y"]
    print(f"  onlyckpt -> {got}  期望 {exp}  {'✅' if got == exp else '❌'}")
    ok &= (got == exp)

    # C: merged 应保持 P,Q,R
    got = ids("merged")
    exp = ["P", "Q", "R"]
    print(f"  merged   -> {got}  期望 {exp}  {'✅' if got == exp else '❌'}")
    ok &= (got == exp)

    # dataset_dict.json 应列出三个域
    idx = os.path.join(BASE, "dataset_dict.json")
    splits = json.load(open(idx, encoding="utf-8"))["splits"] if os.path.isfile(idx) else None
    print(f"  dataset_dict.splits -> {splits}  期望 {['dom','onlyckpt','merged']}  "
          f"{'✅' if splits == ['dom','onlyckpt','merged'] else '❌'}")
    ok &= (splits == ["dom", "onlyckpt", "merged"])

    # 内容合法性
    ds = load_from_disk(os.path.join(BASE, "dom"))
    ch_ok = all(isinstance(r["chunks"], list) and r["chunks"] for r in ds)
    print(f"  chunks 非空 -> {'✅' if ch_ok else '❌'}")
    ok &= ch_ok
    return ok


def main():
    print("=== 构建沙箱 ===")
    build()
    print("=== 跑 merge v2 ===")
    rc = run_merge()
    print(f"  merge_one rc={rc}")
    print("=== 断言 ===")
    ok = check()
    shutil.rmtree(BASE, ignore_errors=True)
    print(f"\n{'TEST_MERGE_OK' if ok else 'TEST_MERGE_FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
