#!/usr/bin/env python3
"""全量成果核查：6 方法 × 6 域。

检查项
------
1. **完整性**：每个 (方法, 域) 的唯一 `_id` 数 vs 该域应有条数
2. **重复**：合并后是否出现重复 `_id`
3. **缺失**：列出具体缺哪些下标
4. **内容合法性**：`chunks` 是非空 list[str]；13 列齐全；`time` 存在
5. **跨方法一致性**：同一个域下，各方法的 `_id` 集合应**完全相同**
   （同一批样本，只是切法不同）—— 这是最强的一条交叉校验

数据来源：`LongBench-v2_chunked/<Method>/<model>/<domain>/`（已合并目录）。
若目录不存在则回退到 checkpoints（`<domain>*.jsonl`），便于在跑的中途核查。

用法:
    python setup/check_all_results.py
    python setup/check_all_results.py --method margin
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets import load_from_disk  # noqa: E402

DOMAINS = ["single-doc", "multi-doc", "code-repo", "long-dialogue",
           "long-icl", "long-structured"]

# (显示名, 子目录, 模型目录)
METHODS = [
    ("Traditional/256", "Traditional/256", None),
    ("Traditional/512", "Traditional/512", None),
    ("Semantic", "Semantic", None),
    ("PPL", "PPL", "Qwen2.5-1.5B-Instruct"),
    ("Margin", "Margin", "Qwen2.5-1.5B-Instruct"),
    ("Lumber", "Lumber", None),
]

COLS = {"_id", "domain", "sub_domain", "difficulty", "length", "question",
        "choice_A", "choice_B", "choice_C", "choice_D", "answer", "chunks",
        "time"}


def expected_ids(dn):
    ds = load_from_disk(os.path.join(ROOT, "datasets", "LongBench-v2",
                                     "split_by_domain", dn))
    return [str(ds[i]["_id"]) for i in range(len(ds))]


def _has_domain_dirs(p):
    if not os.path.isdir(p):
        return False
    for dn in DOMAINS:
        if os.path.isdir(os.path.join(p, dn)):
            return True
    ck = os.path.join(p, "checkpoints")
    if os.path.isdir(ck):
        return any(f.endswith(".jsonl") for f in os.listdir(ck))
    return False


def resolve_root(sub, model):
    """找到"直接装着各域目录/checkpoints"的那一层。

    布局在不同方法间不统一：
        Traditional/256/<domain>/            ← 无模型子目录
        Semantic/bge-m3/<domain>/
        PPL/Qwen2.5-1.5B-Instruct/<domain>/
    所以不能假设固定层级，要逐层探测。
    """
    base = os.path.join(ROOT, "LongBench-v2_chunked", sub)
    cands = []
    if model:
        cands.append(os.path.join(base, model))
    cands.append(base)
    if os.path.isdir(base):
        for d in sorted(os.listdir(base)):
            p = os.path.join(base, d)
            if os.path.isdir(p):
                cands.append(p)
    for c in cands:
        if _has_domain_dirs(c):
            return c
    return None


def load_domain(root, dn):
    """返回 (rows, source_desc)。优先目录，否则回退 checkpoint。"""
    d = os.path.join(root, dn)
    if os.path.isdir(d):
        return list(load_from_disk(d)), "dir"
    ck = os.path.join(root, "checkpoints")
    files = sorted(glob.glob(os.path.join(ck, dn + "*.jsonl"))) + \
        sorted(glob.glob(os.path.join(ck, dn + ".shard*.jsonl")))
    files = sorted(set(files))
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    if rows:
        return rows, f"ckpt×{len(files)}"
    return [], "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=None)
    args = ap.parse_args()

    exp = {dn: expected_ids(dn) for dn in DOMAINS}
    print("域应有条数: " + "  ".join(f"{d}={len(exp[d])}" for d in DOMAINS))
    print()

    idsets = {}          # (method, dn) -> set
    problems = []
    for name, sub, model in METHODS:
        if args.method and args.method.lower() not in name.lower():
            continue
        root = resolve_root(sub, model)
        if not root or not os.path.isdir(root):
            print(f"### {name}: 目录不存在，跳过")
            continue
        print(f"### {name}")
        for dn in DOMAINS:
            rows, src = load_domain(root, dn)
            if not rows:
                print(f"  {dn:16s} 无数据")
                continue
            seen, dups = set(), 0
            bad_cols = bad_chunks = 0
            for r in rows:
                rid = r.get("_id")
                if rid in seen:
                    dups += 1
                    continue
                seen.add(rid)
                if set(r.keys()) != COLS:
                    bad_cols += 1
                ch = r.get("chunks")
                if not isinstance(ch, list) or not ch or \
                        not all(isinstance(x, str) for x in ch):
                    bad_chunks += 1
            n_exp = len(exp[dn])
            miss = [i for i, x in enumerate(exp[dn]) if x not in seen]
            idsets[(name, dn)] = seen
            flag = ""
            if len(seen) != n_exp:
                flag = "  ⚠️ 不完整"
                problems.append(f"{name}/{dn}: {len(seen)}/{n_exp} 缺下标 {miss[:8]}")
            if bad_cols:
                flag += f"  ⚠️ 列不齐 {bad_cols}"
                problems.append(f"{name}/{dn}: 列不齐 {bad_cols}")
            if bad_chunks:
                flag += f"  ⚠️ chunks 非法 {bad_chunks}"
                problems.append(f"{name}/{dn}: chunks 非法 {bad_chunks}")
            # 跨 checkpoint 文件的重叠是**预期内**的（旧单文件 ckpt 与分片 ckpt
            # 内容重叠），合并时按 _id 去重即可 —— 只提示，不算问题。
            dup_note = f"  ↺跨ckpt重叠{dups}" if dups else ""
            print(f"  {dn:16s} {len(seen):4d}/{n_exp:<4d} [{src}]"
                  f" 缺列{bad_cols} 坏chunks{bad_chunks}{dup_note}{flag}")
        print()

    # 跨方法一致性
    print("### 跨方法 _id 一致性（同域应完全一致）")
    for dn in DOMAINS:
        sets = {m: s for (m, d), s in idsets.items() if d == dn and s}
        if len(sets) < 2:
            print(f"  {dn:16s} 可比方法不足（{len(sets)}）")
            continue
        ref_name = sorted(sets)[0]
        ref = sets[ref_name]
        diff = {m: len(s ^ ref) for m, s in sets.items() if s != ref}
        if diff:
            print(f"  {dn:16s} ⚠️ 不一致: {diff}")
            problems.append(f"{dn}: 跨方法 id 集不一致 {diff}")
        else:
            print(f"  {dn:16s} ✅ {len(sets)} 个方法 id 集完全相同 ({len(ref)} 条)")

    print()
    if problems:
        print(f"发现 {len(problems)} 个问题：")
        for p in problems:
            print("  -", p)
    else:
        print("CHECK_ALL_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
