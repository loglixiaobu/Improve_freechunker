#!/usr/bin/env python3
"""把 eval_downstream.py 产出的 raw_*.jsonl 汇总成一张总表。

设计：
  * raw 行里有 question_id / domain(子域) / topk / is_correct / total_sample_time
  * 子域 -> 顶层域 的映射从 LongBench-v2/split_by_domain 读（按 _id）
  * 输出：
      - results/downstream_all_table.csv   （明细：方法 x 域 x topk）
      - results/downstream_overall.csv     （总体：方法 x topk）
      - results/downstream_report.md       （markdown 对照表，带 FreeChunker 相对提升）

用法：
    python setup/aggregate_eval.py [--eval-dir eval_results] [--out-dir results]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys  # noqa: E402

sys.path.insert(0, ROOT)
from src.paths import LONGBENCH_V2  # noqa: E402

SPLIT = os.path.join(LONGBENCH_V2, "split_by_domain")

# 顶层域顺序（报告里的列顺序）
DOMAINS = ["single-doc", "multi-doc", "code-repo", "long-dialogue",
           "long-icl", "long-structured"]

# 方法显示名与排序（FreeChunker 第一）
METHOD_ORDER = ["freechunker", "traditional", "ppl", "margin", "semantic", "lumber"]
METHOD_LABEL = {
    "freechunker": "FreeChunker",
    "traditional": "Traditional(256)",
    "traditional512": "Traditional(512)",
    "ppl": "PPL",
    "margin": "Margin",
    "semantic": "Semantic",
    "lumber": "Lumber",
    "dense_x": "Dense-X",
}


def load_id2domain():
    """question_id -> 顶层域。"""
    from datasets import load_from_disk

    m = {}
    if not os.path.isdir(SPLIT):
        return m
    for dn in sorted(os.listdir(SPLIT)):
        p = os.path.join(SPLIT, dn)
        if not os.path.isdir(p):
            continue
        try:
            ds = load_from_disk(p)
        except Exception:
            continue
        for _id in ds["_id"]:
            m[_id] = dn
    return m


def parse_tag(fname):
    """raw_<method>[_<size>]_shard<N>_rep<R>_<ts>.jsonl -> (method_key, size)"""
    m = re.match(r"raw_(.+?)_shard(\d+)_rep(\d+)_(\d{8}_\d{6})\.jsonl$", fname)
    if not m:
        return None, None
    base = m.group(1)
    size = None
    ms = re.match(r"^(traditional)_(\d+)$", base)
    if ms:
        return ms.group(1), ms.group(2)
    return base, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=os.path.join(ROOT, "eval_results"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    id2dom = load_id2domain()
    print(f"id->domain 映射 {len(id2dom)} 条")

    files = sorted(f for f in os.listdir(args.eval_dir) if f.startswith("raw_"))
    print(f"发现 {len(files)} 个 raw 文件")

    # acc[(method_key, domain, topk)] = [correct, total]
    acc = collections.defaultdict(lambda: [0, 0])
    # times[(method_key, domain)] = [总时间]
    times = collections.defaultdict(float)
    seen_q = collections.defaultdict(set)  # 去重：同一 (方法,域,题) 只算一次

    skipped = 0
    for fname in files:
        mkey, size = parse_tag(fname)
        if mkey is None:
            skipped += 1
            continue
        key = mkey if size is None else f"{mkey}{size}"
        path = os.path.join(args.eval_dir, fname)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                qid = r.get("question_id")
                dom = id2dom.get(qid, "unknown")
                topk = r.get("topk")
                uniq = (key, dom, topk, qid)
                if qid in seen_q[(key, dom, topk)]:
                    continue
                seen_q[(key, dom, topk)].add(qid)
                c, t = acc[(key, dom, topk)]
                acc[(key, dom, topk)] = [c + (1 if r.get("is_correct") else 0), t + 1]
                times[(key, dom)] += float(r.get("total_sample_time") or 0.0)

    if not acc:
        print("没有任何结果可汇总（raw 文件为空？）")
        return

    methods = [k for k in METHOD_ORDER if any(a[0] == k for a in acc)]
    topks = sorted({k for (_, _, k) in acc})

    # ---- 明细 CSV ----
    detail = os.path.join(args.out_dir, "downstream_all_table.csv")
    with open(detail, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "domain", "topk", "correct", "total", "acc_pct", "total_time_s"])
        for k in methods:
            for dom in DOMAINS + ["unknown"]:
                if not any((k, dom, tk) in acc for tk in topks):
                    continue
                for tk in topks:
                    if (k, dom, tk) not in acc:
                        continue
                    c, t = acc[(k, dom, tk)]
                    w.writerow([METHOD_LABEL.get(k, k), dom, tk, c, t,
                                f"{100.0*c/t:.2f}" if t else "", f"{times.get((k,dom),0):.1f}"])
                    # 每域只写一次时间，避免重复
    print(f"[write] {detail}")

    # ---- 总体 CSV ----
    overall = os.path.join(args.out_dir, "downstream_overall.csv")
    with open(overall, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"TopK{k}_pct" for k in topks] + ["n_questions", "total_time_s"])
        for k in methods:
            row = [METHOD_LABEL.get(k, k)]
            nq = 0
            tt = 0.0
            for tk in topks:
                c = sum(acc[(k, d, tk)][0] for d in DOMAINS if (k, d, tk) in acc)
                t = sum(acc[(k, d, tk)][1] for d in DOMAINS if (k, d, tk) in acc)
                row.append(f"{100.0*c/t:.2f}" if t else "")
                if tk == topks[0]:
                    nq = t
            for d in DOMAINS:
                tt += times.get((k, d), 0.0)
            row += [nq, f"{tt:.1f}"]
            w.writerow(row)
    print(f"[write] {overall}")

    # ---- markdown 报告 ----
    md = os.path.join(args.out_dir, "downstream_report.md")
    L = []
    L.append("# FreeChunker 下游评测 · 对照表\n")
    L.append(f"- 嵌入模型: jina-embeddings-v2-small-en（仅最小模型）")
    L.append(f"- QA 生成: Qwen3-8B @ vLLM")
    L.append(f"- 域: {', '.join(DOMAINS)}")
    L.append(f"- 汇总自 {len(files)} 个 raw 文件\n")

    L.append("## 总体准确率\n")
    L.append("| 方法 | " + " | ".join(f"TopK-{k}" for k in topks) + " | 题数 | 耗时 |")
    L.append("|" + "---|" * (len(topks) + 3))
    base = {}
    for k in methods:
        for tk in topks:
            c = sum(acc[(k, d, tk)][0] for d in DOMAINS if (k, d, tk) in acc)
            t = sum(acc[(k, d, tk)][1] for d in DOMAINS if (k, d, tk) in acc)
            base[(k, tk)] = 100.0 * c / t if t else 0.0
    for k in methods:
        cells = [METHOD_LABEL.get(k, k)]
        for tk in topks:
            v = base[(k, tk)]
            extra = ""
            if k != "freechunker" and base.get(("freechunker", tk)) is not None:
                d = v - base[("freechunker", tk)]
                extra = f" ({d:+.2f})"
            cells.append(f"{v:.2f}{extra}")
        nq = sum(acc[(k, d, topks[0])][1] for d in DOMAINS if (k, d, topks[0]) in acc)
        tt = sum(times.get((k, d), 0.0) for d in DOMAINS)
        cells += [str(nq), f"{tt/3600:.2f}h"]
        L.append("| " + " | ".join(cells) + " |")
    L.append("\n> 括号内为相对 FreeChunker 的差值（百分点）\n")

    for tk in topks:
        L.append(f"## 分域明细 · TopK-{tk}\n")
        L.append("| 域 | " + " | ".join(METHOD_LABEL.get(k, k) for k in methods) + " |")
        L.append("|" + "---|" * (len(methods) + 1))
        for d in DOMAINS:
            cells = [d]
            for k in methods:
                if (k, d, tk) in acc:
                    c, t = acc[(k, d, tk)]
                    cells.append(f"{100.0*c/t:.1f}" if t else "-")
                else:
                    cells.append("-")
            L.append("| " + " | ".join(cells) + " |")
        L.append("")

    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[write] {md}")
    print("\n" + "\n".join(L[:40]))
    print("\nAGGREGATE_DONE")


if __name__ == "__main__":
    main()
