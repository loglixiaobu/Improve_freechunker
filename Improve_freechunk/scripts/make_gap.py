#!/usr/bin/env python3
"""生成「某个变体 vs 基线 vs 论文」的对照表。

用法:
    python scripts/make_gap.py --eval-dir results/I1_gran_norm --out results/I1_gran_norm/gap.md
    python scripts/make_gap.py --eval-dir results/I1_gran_norm        # 打印到 stdout

设计
----
* 基线数字硬编码在 BASELINE（来自 2026-10-05 的实测，见 BASELINE.md）
* 论文数字硬编码在 PAPER（FreeChunker.pdf Table 1，jina-small-en 行）
* 三者并排，让 commit message 可以直接贴
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

DOMAINS = ["single-doc", "multi-doc", "code-repo",
           "long-dialogue", "long-icl", "long-structured"]

# 本次实测基线（FreeChunker / Traditional(256)），来自 test_result/
BASELINE = {
    "FreeChunker": {
        "single-doc": (33.71, 33.71), "multi-doc": (28.00, 25.60),
        "code-repo": (48.00, 46.00), "long-dialogue": (28.21, 35.90),
        "long-icl": (25.93, 28.40), "long-structured": (30.30, 24.24),
        "Overall": (31.81, 31.61),
    },
    "Traditional(256)": {
        "single-doc": (30.86, 35.43), "multi-doc": (24.00, 28.00),
        "code-repo": (44.00, 52.00), "long-dialogue": (25.64, 33.33),
        "long-icl": (27.16, 30.86), "long-structured": (15.15, 24.24),
        "Overall": (28.43, 33.60),
    },
}

# 论文 Table 1（jina-embeddings-v2-small-en）
PAPER = {
    "FreeChunker": {
        "single-doc": (35.43, 32.57), "multi-doc": (26.40, 26.40),
        "code-repo": (46.00, 44.00), "long-dialogue": (28.21, 23.08),
        "long-icl": (32.10, 28.40), "long-structured": (42.42, 45.45),
        "Overall": (33.60, 31.61),
    },
    "Traditional": {
        "single-doc": (30.86, 35.43), "multi-doc": (24.80, 28.00),
        "code-repo": (44.00, 51.33), "long-dialogue": (26.50, 33.33),
        "long-icl": (27.57, 30.86), "long-structured": (15.15, 24.24),
        "Overall": (28.76, 33.53),
    },
}


def load_raw(eval_dir):
    """直接读 raw jsonl，按 (方法, 域) 统计，不依赖 aggregate 的命名约定。"""
    id2dom = {}
    from datasets import load_from_disk
    root = os.path.expanduser("~/FreeChunker")
    for dn in DOMAINS:
        p = os.path.join(root, "datasets", "LongBench-v2", "split_by_domain", dn)
        if os.path.isdir(p):
            ds = load_from_disk(p)
            for x in ds["_id"]:
                id2dom[x] = dn

    acc = defaultdict(lambda: [0, 0])
    for p in sorted(glob.glob(os.path.join(eval_dir, "raw_*.jsonl"))):
        b = os.path.basename(p)
        m = re.match(r"raw_(.+?)_shard\d+_rep\d+_\d{8}_\d{6}\.jsonl$", b)
        if not m:
            continue
        meth = m.group(1)
        seen = set()
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                qid = r.get("question_id")
                k = (meth, id2dom.get(qid, "?"), r.get("topk"))
                if qid in seen and k[0] == meth:
                    pass
                acc[k][0] += 1 if r.get("is_correct") else 0
                acc[k][1] += 1
    return acc


def norm_method(m):
    """raw 文件名里的方法名 -> 展示名"""
    if m.startswith("traditional"):
        return "Traditional(256)" if "256" in m else "Traditional(512)"
    return "FreeChunker"


def overall(acc, meth, topk):
    c = sum(acc[(meth, d, topk)][0] for d in DOMAINS if (meth, d, topk) in acc)
    t = sum(acc[(meth, d, topk)][1] for d in DOMAINS if (meth, d, topk) in acc)
    return (100.0 * c / t) if t else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None, help="变体名，默认取目录名")
    args = ap.parse_args()

    label = args.label or os.path.basename(os.path.abspath(args.eval_dir))
    acc = load_raw(args.eval_dir)

    # 找出本次跑了哪些方法
    meths = sorted({k[0] for k in acc})
    L = [f"# 变体 `{label}` · 对照表\n"]
    L.append("| 指标 | 论文 | 基线 | 本次 | Δ vs 基线 | Δ vs 论文 |")
    L.append("|---|---|---|---|---|---|")

    for raw_meth in meths:
        disp = norm_method(raw_meth)
        base_key = "FreeChunker" if disp == "FreeChunker" else "Traditional(256)"
        paper_key = "FreeChunker" if disp == "FreeChunker" else "Traditional"
        for topk in (5, 10):
            cur = overall(acc, raw_meth, topk)
            if cur is None:
                continue
            b = BASELINE.get(base_key, {}).get("Overall", (None, None))[0 if topk == 5 else 1]
            p = PAPER.get(paper_key, {}).get("Overall", (None, None))[0 if topk == 5 else 1]
            L.append(f"| {disp} Overall Top-{topk} | {p:.2f} | {b:.2f} | "
                     f"**{cur:.2f}** | {cur-b:+.2f} | {cur-p:+.2f} |")

    L.append("\n## 分域（Top-5 / Top-10）\n")
    for raw_meth in meths:
        disp = norm_method(raw_meth)
        L.append(f"### {disp}\n")
        L.append("| 域 | 论文 T5/T10 | 基线 T5/T10 | 本次 T5/T10 |")
        L.append("|---|---|---|---|")
        base_key = "FreeChunker" if disp == "FreeChunker" else "Traditional(256)"
        paper_key = "FreeChunker" if disp == "FreeChunker" else "Traditional"
        for dn in DOMAINS:
            c5 = acc.get((raw_meth, dn, 5))
            c10 = acc.get((raw_meth, dn, 10))
            f = lambda x: f"{100.0*x[0]/x[1]:.2f}" if x and x[1] else "—"
            p = PAPER.get(paper_key, {}).get(dn, (None, None))
            b = BASELINE.get(base_key, {}).get(dn, (None, None))
            L.append(f"| {dn} | {p[0]:.2f} / {p[1]:.2f} | {b[0]:.2f} / {b[1]:.2f} | "
                     f"{f(c5)} / {f(c10)} |")
        L.append("")

    txt = "\n".join(L)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        open(args.out, "w", encoding="utf-8").write(txt)
        print(f"[write] {args.out}")
    print(txt)


if __name__ == "__main__":
    main()
