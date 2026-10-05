#!/usr/bin/env python3
"""合并 Lumber 分片输出 -> 标准目录（test_lumber.py 期望 Lumber/qwen3-8b/<domain>/）。"""
from __future__ import annotations

import json
import os
import shutil
import sys

from datasets import load_from_disk, Dataset, concatenate_datasets

BASE = "/home1/lh/FreeChunker/LongBench-v2_chunked/Lumber/qwen3-8b"
DOMAINS = ["single-doc", "multi-doc"]


def main():
    if not os.path.isdir(BASE):
        print("没有 Lumber 输出目录", BASE)
        sys.exit(1)
    entries = sorted(os.listdir(BASE))
    print("现有条目:", entries)

    merged = {}
    for dn in DOMAINS:
        plain = os.path.join(BASE, dn)
        shards = sorted(e for e in entries if e.startswith(dn + ".shard"))
        parts = []
        if os.path.isdir(plain):
            parts.append(load_from_disk(plain))
        for sh in shards:
            p = os.path.join(BASE, sh)
            if os.path.isdir(p):
                d = load_from_disk(p)
                print(f"  {dn}: + {sh} {len(d)}")
                parts.append(d)
        if not parts:
            print(f"  {dn}: 缺失")
            continue
        ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        seen, keep = set(), []
        for row in ds:
            if row["_id"] in seen:
                continue
            seen.add(row["_id"])
            keep.append(row)
        ds = Dataset.from_list(keep)
        print(f"  {dn}: 合并 {len(ds)} 样本")
        if os.path.isdir(plain):
            shutil.rmtree(plain)
        ds.save_to_disk(plain)
        merged[dn] = ds

    for e in os.listdir(BASE):
        if ".shard" in e:
            p = os.path.join(BASE, e)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            print("  清理:", e)

    with open(os.path.join(BASE, "dataset_dict.json"), "w", encoding="utf-8") as f:
        json.dump({"splits": list(merged.keys())}, f, ensure_ascii=False)
    print("\n✅ 合并完成 splits =", list(merged.keys()))
    print("MERGE_DONE")


if __name__ == "__main__":
    main()
