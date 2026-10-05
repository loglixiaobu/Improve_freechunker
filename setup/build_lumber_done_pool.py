#!/usr/bin/env python3
"""把 Lumber 历史 shard checkpoint 灌进全局完成池。

改 worker 数（6 -> 48）会让 `i % nshards` 的映射失效，老 shard 文件里的
已完成样本对不上新 worker 的份额。把它们统一灌进 `_done_global.jsonl`，
新 worker 启动时就能跨 shard 跳过，不重跑。

用法:
    python setup/build_lumber_done_pool.py            # 默认 Lumber/qwen3-8b
    python setup/build_lumber_done_pool.py --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "LongBench-v2_chunked", "Lumber", "qwen3-8b", "checkpoints")
POOL = os.path.join(CKPT, "_done_global.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(CKPT):
        print(f"[err] checkpoint 目录不存在: {CKPT}")
        return

    pool_ids = set()
    if os.path.isfile(POOL):
        with open(POOL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("_id") is not None:
                    pool_ids.add(obj["_id"])
    print(f"全局池已有 {len(pool_ids)} 条")

    shard_files = sorted(glob.glob(os.path.join(CKPT, "*.shard*.jsonl")))
    src_ids = set()
    recs = {}
    for p in shard_files:
        n = 0
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                rid = obj.get("_id")
                if rid is None:
                    continue
                recs[rid] = obj
                src_ids.add(rid)
                n += 1
        print(f"  {os.path.basename(p)}: {n} 条")

    new_ids = src_ids - pool_ids
    print(f"shard 文件合计 {len(src_ids)} 条，其中 {len(new_ids)} 条需补进全局池")

    if args.dry_run:
        print("[dry-run] 未写入")
        print("POOL_DRY_DONE")
        return

    with open(POOL, "a", encoding="utf-8") as f:
        for rid in sorted(new_ids):
            f.write(json.dumps(recs[rid], ensure_ascii=False) + "\n")
    print(f"[ok] 全局池现在共 {len(pool_ids) + len(new_ids)} 条 -> {POOL}")
    print("POOL_BUILD_DONE")


if __name__ == "__main__":
    main()
