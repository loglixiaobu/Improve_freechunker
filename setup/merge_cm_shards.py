#!/usr/bin/env python3
"""合并 PPL / Margin 的分片输出（v2：同时吃 **目录** 和 **checkpoint**）。

分片跑法会在
    LongBench-v2_chunked/<PPL|Margin>/<model>/<domain>.shardN/
留下 N 份 directory dataset，checkpoints 侧也有 <domain>.shardN.jsonl。

### v1 的缺口（2026-10-04 21:40 发现）
v1 只合并**目录**（`load_from_disk`）。但一个 worker 在**跑完之前**
不会 `save_to_disk`，它的成果**只存在于 checkpoint jsonl** 里。
→ 一旦 kill 掉一个跑到一半的 worker（比如 code-repo 32/50），
  这 32 条在合并时就**彻底丢了**（磁盘上只有 `code-repo.jsonl`，没有目录）。

### v2 修法
按 `_id` 去重，把三类来源合起来：
  1. `<domain>/`（已合并态，可能不存在）
  2. `<domain>.shardN/`、`<domain>.gap*/`（分片目录）
  3. `<domain>.jsonl`、`<domain>.shardN.jsonl`、`<domain>.gap*.jsonl`（checkpoint）

已核实：checkpoint 记录与目录记录的**列名完全一致**（13 列），
`Dataset.from_list` 混合两者不会 schema 冲突。

用法:
    python setup/merge_cm_shards.py --method ppl      # 或 margin
    python setup/merge_cm_shards.py --method both
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets import Dataset, load_from_disk  # noqa: E402

MODEL_DIR = {
    "ppl": ("PPL", "Qwen2.5-1.5B-Instruct"),
    "margin": ("Margin", "Qwen2.5-1.5B-Instruct"),
}
DOMAINS = ["single-doc", "multi-doc", "code-repo", "long-dialogue",
           "long-icl", "long-structured"]


def _is_part_of(stem: str, dn: str) -> bool:
    """stem 是否属于域 dn（本体、.shardN、.gapXXX 都算）。"""
    return (stem == dn
            or stem.startswith(dn + ".shard")
            or stem.startswith(dn + ".gap"))


def _dir_candidates(root: str, dn: str):
    """返回 (plain_dir_or_None, [分片目录...])。"""
    plain = os.path.join(root, dn)
    plain = plain if os.path.isdir(plain) else None
    shards = []
    for entry in sorted(os.listdir(root)):
        if not (entry.startswith(dn + ".shard") or entry.startswith(dn + ".gap")):
            continue
        p = os.path.join(root, entry)
        if os.path.isdir(p):
            shards.append(p)
    return plain, shards


def _ckpt_files(root: str, dn: str):
    """返回该域相关的所有 checkpoint jsonl（含本体 <dn>.jsonl）。

    ⚠️ checkpoint 放在 `<root>/checkpoints/` 子目录（见 chunker 的
    `_CM_CKPT_DIR = <out_root>/checkpoints`），不是 `<root>/` 下。
    为兼容早期布局，两个位置都扫。
    """
    out = []
    for base in (root, os.path.join(root, "checkpoints")):
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base)):
            if not entry.endswith(".jsonl"):
                continue
            stem = entry[:-len(".jsonl")]
            if _is_part_of(stem, dn):
                out.append(os.path.join(base, entry))
    return out


def _read_ckpt_rows(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    # 截断的末行（worker 被 kill）—— 跳过
                    continue
    return rows


def _decide(plain, shard_dirs, shard_cks, ck_files):
    """返回 (是否合并, 说明)。抽出来是为了让干跑脚本能测到**真实**判定。"""
    if plain is None and not shard_dirs and not ck_files:
        return False, "跳过(真的一无所有)"
    if plain is not None and not shard_dirs and not shard_cks:
        return False, "跳过(已合并)"
    return True, "合并"


def merge_one(method: str) -> int:
    sub, model = MODEL_DIR[method]
    root = os.path.join(ROOT, "LongBench-v2_chunked", sub, model)
    if not os.path.isdir(root):
        print(f"[{method}] 目录不存在: {root}")
        return 1

    merged_any = 0
    for dn in DOMAINS:
        plain, shard_dirs = _dir_candidates(root, dn)
        ck_files = _ckpt_files(root, dn)
        # 分片来源：目录 or 分片 checkpoint（本体 checkpoint 不算"分片"）
        shard_cks = [p for p in ck_files
                     if os.path.basename(p)[:-len(".jsonl")] != dn]

        do_merge, why = _decide(plain, shard_dirs, shard_cks, ck_files)
        if not do_merge:
            if why != "跳过(真的一无所有)":
                print(f"  [{dn}] {why}")
            continue

        seen, rows = set(), []
        for p in ([plain] if plain else []) + shard_dirs:
            try:
                ds = load_from_disk(p)
            except Exception as e:
                print(f"  [{dn}] 目录读取失败 {p}: {e}")
                continue
            for r in ds:
                rid = r.get("_id")
                if rid in seen:
                    continue
                seen.add(rid)
                rows.append(r)
        n_from_dir = len(rows)

        for r in _read_ckpt_rows(ck_files):
            rid = r.get("_id")
            if rid is None or rid in seen:
                continue
            seen.add(rid)
            rows.append(r)
        n_from_ck = len(rows) - n_from_dir

        if not rows:
            print(f"  [{dn}] 没有任何样本，跳过")
            continue

        # 原子替换：先写到临时目录，再换名
        tmp = os.path.join(root, dn + ".__merged_tmp")
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        Dataset.from_list(rows).save_to_disk(tmp)
        if os.path.isdir(os.path.join(root, dn)):
            shutil.rmtree(os.path.join(root, dn))
        os.rename(tmp, os.path.join(root, dn))
        merged_any += 1
        print(f"  [{dn}] 合并 {len(shard_dirs)} 片目录 + "
              f"{len(ck_files)} 份 checkpoint -> {len(rows)} 样本 "
              f"(目录贡献 {n_from_dir}，checkpoint 补 {n_from_ck})")

    if merged_any == 0:
        print(f"[{method}] 没有可合并的分片（可能本来就没分片跑）")
    idx = os.path.join(root, "dataset_dict.json")
    with open(idx, "w", encoding="utf-8") as f:
        json.dump({"splits": DOMAINS}, f, ensure_ascii=False)
    print(f"[{method}] 索引已写: {idx}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="both", choices=["ppl", "margin", "both"])
    args = ap.parse_args()
    methods = ["ppl", "margin"] if args.method == "both" else [args.method]
    for m in methods:
        print(f"=== 合并 {m} ===")
        merge_one(m)
    print("MERGE_CM_DONE")


if __name__ == "__main__":
    main()
