#!/usr/bin/env python3
"""G1 数据门禁 —— 编码产物必须与语料**逐条对齐**，否则训练几小时全是白跑。

为什么必须有这一关：训练 loss 只会告诉你「模型学得像不像」，不会告诉你
「label 是不是配错了」。如果 input/label 错位（label 少一条、粒度用错、切片范围错），
loss 照样能降到很低，但模型学到的是错的东西 —— 要到端到端评测才发现，
那时已经烧掉一整晚。所以先卡住。

检查项：
  1. **覆盖范围**：把各分片 manifest 里的 [lo,hi) 区间取并集，与语料逐篇核对，
     期望行数 = 这些区间内 `len(sentences) >= 2` 的篇数（造数据脚本会跳过 n<2）
  2. 向量行数 == 期望行数，且 == manifest 声明之和
  3. features 是 {input, label}，两层嵌套，dtype 都是 float32
  4. 随机抽 20 条：len(input) == n，len(label) == generate_shifted_matrix(n,[2,4]) 的列数
  5. 向量维度 == 512，且全部是有限值（无 NaN / Inf）

用法：
    python setup/g1_check.py --split train --split val --split test
    python setup/g1_check.py --allow-partial        # 冒烟产物（几百行）时跳过行数检查
退出码：0 = 全过；1 = 有检查项失败。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyarrow.compute as pc

from src.paths import GRANULARITIES, VECTOR_DIR, load_corpus_split, load_vector_split
from src.utils import generate_shifted_matrix

EMB_DIM = 512
N_SAMPLE = 20


def sentence_counts(corpus_split):
    """每篇的句数 —— 直接读 Arrow 的 list 长度，**不materialize 文本**，10 万篇也是秒级。"""
    ds = load_corpus_split(corpus_split)
    lens = pc.list_value_length(ds.data.column("sentences"))
    return lens.to_numpy(zero_copy_only=False).astype(np.int64)


def load_manifests(parts_dir, split):
    """优先读分片 manifest；若 merge 时已 --remove-parts，则回退读合并 manifest。"""
    out = []
    for p in sorted(glob.glob(os.path.join(parts_dir, "manifest_shard*.json"))):
        try:
            out.append(json.load(open(p, encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            print(f"  !! 读不了 {p}: {e}")
    if out:
        return out, "分片 manifest"
    merged = os.path.join(VECTOR_DIR, f"{split}.manifest.json")
    if os.path.exists(merged):
        try:
            return json.load(open(merged, encoding="utf-8")).get("shards", []), f"合并 manifest {merged}"
        except Exception as e:  # noqa: BLE001
            print(f"  !! 读不了 {merged}: {e}")
    return [], "无"


def union_ranges(ranges):
    """把 [lo,hi) 区间列表合并成不重叠的区间。"""
    rs = sorted(ranges)
    merged = []
    for lo, hi in rs:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(a, b) for a, b in merged]


def check_split(split, results, allow_partial=False):
    def ok(name, cond, detail=""):
        results.append((f"{split}:{name}", bool(cond), detail))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)

    print(f"\n--- split={split} ---", flush=True)
    parts_dir = os.path.join(VECTOR_DIR, "_parts", split)
    manifests, src = load_manifests(parts_dir, split)
    declared = sum(m["n_records"] for m in manifests) if manifests else None

    vec = load_vector_split(split)
    ok("向量目录存在", True, f"{len(vec)} 行 <- {os.path.join(VECTOR_DIR, split)}")

    # --- 覆盖范围核对 ---
    if manifests:
        by_corpus = {}
        for m in manifests:
            by_corpus.setdefault(m["corpus_split"], []).append((m["lo"], m["hi"]))
        print(f"  {src}：{len(manifests)} 份，声明行数合计 {declared}", flush=True)
        total_exp = 0
        for cs, rs in sorted(by_corpus.items()):
            counts = sentence_counts(cs)
            exp = 0
            for lo, hi in union_ranges(rs):
                hi = min(hi, len(counts))
                exp += int((counts[lo:hi] >= 2).sum())
            total_exp += exp
            print(f"    语料 {cs}: 覆盖 {union_ranges(rs)} -> 期望 {exp} 行", flush=True)
        if allow_partial:
            print(f"  [SKIP] 行数对齐  (--allow-partial，冒烟产物 {len(vec)} 行)", flush=True)
        else:
            ok("行数与覆盖范围对齐", len(vec) == total_exp,
               f"向量 {len(vec)} vs 期望 {total_exp}")
            ok("行数与 manifest 一致", declared is None or len(vec) == declared,
               f"向量 {len(vec)} vs manifest {declared}")
    else:
        print(f"  !! {src}：没有可用 manifest，无法核对覆盖范围", flush=True)
        if not allow_partial:
            ok("manifest 可读", False, "缺 manifest，无法核对覆盖范围")

    # --- 结构核对 ---
    feats = vec.features
    ok("features 键", set(feats.keys()) == {"input", "label"}, str(list(feats.keys())))
    for k in ("input", "label"):
        try:
            inner = feats[k].feature.feature.dtype
            ok(f"{k} 两层嵌套 + float32", inner == "float32", f"内层 dtype={inner}")
        except Exception as e:  # noqa: BLE001
            ok(f"{k} 两层嵌套 + float32", False, f"{type(e).__name__}: {e}")

    # --- 抽样核对 ---
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(vec), size=min(N_SAMPLE, len(vec)), replace=False)
    bad_shape, bad_dim, bad_finite, examples = [], [], [], []
    for i in idxs:
        row = vec[int(i)]
        x = np.asarray(row["input"], dtype=np.float32)
        y = np.asarray(row["label"], dtype=np.float32)
        n, m = x.shape[0], y.shape[0]
        m_exp = int(generate_shifted_matrix(n, granularities=list(GRANULARITIES))[0].shape[1])
        if m != m_exp:
            bad_shape.append((int(i), n, m, m_exp))
        if x.ndim != 2 or y.ndim != 2 or x.shape[1] != EMB_DIM or y.shape[1] != EMB_DIM:
            bad_dim.append((int(i), x.shape, y.shape))
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            bad_finite.append(int(i))
        if len(examples) < 3:
            examples.append((int(i), n, m))

    ok(f"抽 {len(idxs)} 条形状对齐", not bad_shape,
       f"异常 {bad_shape[:5]}" if bad_shape else f"input=n, label=m(粒度) 吻合；例 {examples}")
    ok("维度=512", not bad_dim, f"异常 {bad_dim[:5]}" if bad_dim else "input/label 均 [*, 512]")
    ok("无 NaN/Inf", not bad_finite, f"异常 {bad_finite[:5]}" if bad_finite else "全部有限")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="append", default=None)
    ap.add_argument("--allow-partial", action="store_true",
                    help="跳过「行数对齐」检查（冒烟产物只有几百行时用）")
    args = ap.parse_args()
    splits = args.split or ["train", "val"]

    print("=" * 70)
    print(f"G1 数据门禁  VECTOR_DIR = {VECTOR_DIR}")
    print(f"粒度 = {list(GRANULARITIES)}")
    print("=" * 70, flush=True)

    results = []
    for s in splits:
        try:
            check_split(s, results, args.allow_partial)
        except Exception as e:  # noqa: BLE001
            results.append((f"{s}:加载", False, f"{type(e).__name__}: {e}"))
            print(f"  [FAIL] {s} 加载失败: {type(e).__name__}: {e}", flush=True)

    n_fail = sum(1 for _, c, _ in results if not c)
    print("\n" + "=" * 70)
    for name, c, detail in results:
        if not c:
            print(f"  FAIL  {name}  {detail}")
    print(f"  共 {len(results)} 项，失败 {n_fail} 项")
    print("=" * 70, flush=True)
    print("G1_PASS" if n_fail == 0 else "G1_FAIL")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
