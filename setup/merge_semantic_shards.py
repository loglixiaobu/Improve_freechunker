#!/usr/bin/env python3
"""把 Semantic 的分片输出合并成 test_semantic.py 期望的标准目录结构。

分片跑完后，Semantic/bge-m3/ 下会有：
  single-doc/            (单 shard 直接产出)
  multi-doc/
  code-repo.shard0/      (二级分片)
  code-repo.shard1/
  ...
最终需要合并成：
  Semantic/bge-m3/code-repo/     <- shard0 + shard1 合并
  Semantic/bge-m3/dataset_dict.json
"""
from __future__ import annotations

import json
import os
import shutil
import sys

from datasets import load_from_disk, Dataset, concatenate_datasets

ROOT = "/home1/lh/FreeChunker"
BASE = os.path.join(ROOT, "LongBench-v2_chunked", "Semantic", "bge-m3")

DOMAINS = ["single-doc", "multi-doc", "code-repo", "long-dialogue",
           "long-icl", "long-structured"]


def main():
    if not os.path.isdir(BASE):
        print("没有 Semantic 输出目录", BASE)
        sys.exit(1)

    entries = sorted(os.listdir(BASE))
    print("现有条目:", entries)

    merged = {}
    for dn in DOMAINS:
        plain = os.path.join(BASE, dn)
        shards = sorted(
            [e for e in entries if e.startswith(dn + ".shard")]
        )
        if os.path.isdir(plain) and not shards:
            ds = load_from_disk(plain)
            print(f"  {dn}: 单 shard {len(ds)} 样本")
            merged[dn] = ds
            continue
        parts = []
        if os.path.isdir(plain):
            parts.append(load_from_disk(plain))
        for sh in shards:
            p = os.path.join(BASE, sh)
            if os.path.isdir(p):
                d = load_from_disk(p)
                print(f"  {dn}: + {sh} {len(d)} 样本")
                parts.append(d)
        if not parts:
            print(f"  {dn}: 缺失，跳过")
            continue
        ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        # 去重（按 _id），防止 checkpoint 重跑导致重复
        seen, keep = set(), []
        for row in ds:
            if row["_id"] in seen:
                continue
            seen.add(row["_id"])
            keep.append(row)
        ds = Dataset.from_list(keep)
        print(f"  {dn}: 合并后 {len(ds)} 样本")

        # 写回标准目录
        if os.path.isdir(plain):
            shutil.rmtree(plain)
        ds.save_to_disk(plain)
        merged[dn] = ds

    # 清掉分片残留
    for e in os.listdir(BASE):
        if ".shard" in e and e not in DOMAINS:
            p = os.path.join(BASE, e)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            print("  清理:", e)

    with open(os.path.join(BASE, "dataset_dict.json"), "w", encoding="utf-8") as f:
        json.dump({"splits": list(merged.keys())}, f, ensure_ascii=False)
    print("\n✅ 合并完成，splits =", list(merged.keys()))
    print("MERGE_DONE")


if __name__ == "__main__":
    main()
