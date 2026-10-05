#!/usr/bin/env python3
"""把 1_build_pretrain_datasets.py 产出的 part_* 合并成标准 Dataset 目录。

原仓库缺了这一步：造数据脚本把 part 存到 `vector/<backbone>/`（train 直接平铺）
或 `vector/<backbone>/<split>/`，而训练脚本读 `vector/<backbone>/train`
—— 两边路径对不上，直接跑必然报错。本脚本补上这个缺口，并在合并前校验：

  * 行数与各分片 manifest 声明的 n_records 之和一致
  * features 是 {input, label}，dtype 都是 float32（否则磁盘会翻倍）
  * 抽第一条复算 span 数 == label 条数

用法：
    python "train and preprocess/1b_merge_parts.py" --split train --remove-parts
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import concatenate_datasets, load_from_disk

from src.paths import VECTOR_DIR
from src.utils import generate_shifted_matrix


def merge(parts_dir, out_dir, expect=None, remove_parts=False):
    parts = sorted(glob.glob(os.path.join(parts_dir, "part_*")))
    if not parts:
        raise FileNotFoundError(f"{parts_dir} 下没有 part_*，请先跑 1_build_pretrain_datasets.py")
    print(f"合并 {len(parts)} 个 part: {os.path.basename(parts[0])} .. {os.path.basename(parts[-1])}")
    print(f"  -> {out_dir}")

    manifests = sorted(glob.glob(os.path.join(parts_dir, "manifest_shard*.json")))
    declared = None
    if manifests:
        ms = [json.load(open(p, encoding="utf-8")) for p in manifests]
        declared = sum(m["n_records"] for m in ms)
        print(f"  分片 manifest {len(ms)} 份，声明合计 n_records = {declared}")
        for m in ms:
            print(f"    shard {m['shard']:>2}/{m['nshards']} [{m['lo']}:{m['hi']}] "
                  f"records={m['n_records']} (新 {m['n_new']} + 跳过 {m['n_skipped_parts']}) "
                  f"n<2 跳过 {m['n_short_docs']} | {m['elapsed_min']} min | "
                  f"max_batch={m['batch_stats']['max_batch']}")

    ds = concatenate_datasets([load_from_disk(p) for p in parts])
    print(f"  行数 = {len(ds)} | features = {ds.features}")

    for k in ("input", "label"):
        dt = ds.features[k].feature.feature.dtype
        assert dt == "float32", f"{k} dtype = {dt}，应为 float32（否则磁盘会翻倍）"

    if declared is not None and len(ds) != declared:
        raise RuntimeError(f"行数 {len(ds)} != 分片 manifest 声明 {declared}，不要继续")
    if expect is not None and len(ds) != expect:
        raise RuntimeError(f"行数 {len(ds)} != 期望 {expect}，不要继续")

    if os.path.isdir(out_dir):
        print(f"  {out_dir} 已存在，覆盖")
        shutil.rmtree(out_dir)
    ds.save_to_disk(out_dir)
    print(f"  已保存 {out_dir}")

    # 自检：抽第一条，复算 span 数必须等于 label 条数
    row = ds[0]
    n, m = len(row["input"]), len(row["label"])
    m_exp = int(generate_shifted_matrix(n, granularities=[2, 4])[0].shape[1])
    if m != m_exp:
        raise RuntimeError(f"抽检 doc0: input={n} label={m}，但按 [2,4] 应为 {m_exp}")
    print(f"  抽检 doc0: input={n} 条, label={m} 条 (与 [2,4] 粒度一致)")

    summary = {
        "split": os.path.basename(out_dir), "rows": len(ds),
        "features": {k: str(ds.features[k]) for k in ds.features},
        "n_parts": len(parts), "declared_n_records": declared,
        "shards": [json.load(open(p, encoding="utf-8")) for p in manifests],
        "merged_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
    }
    mp = out_dir.rstrip("/") + ".manifest.json"
    json.dump(summary, open(mp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  汇总 manifest -> {mp}")

    if remove_parts:
        shutil.rmtree(parts_dir)
        print(f"  已删除中间产物 {parts_dir}")

    return len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--parts-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--expect", type=int, default=None, help="期望行数，不符就报错")
    ap.add_argument("--remove-parts", action="store_true")
    args = ap.parse_args()

    split = "val" if args.split == "validation" else args.split
    parts_dir = args.parts_dir or os.path.join(VECTOR_DIR, "_parts", split)
    out_dir = args.out_dir or os.path.join(VECTOR_DIR, split)

    total = merge(parts_dir, out_dir, args.expect, args.remove_parts)
    print(f"MERGE_DONE {split} rows={total}")


if __name__ == "__main__":
    main()
