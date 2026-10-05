#!/usr/bin/env python3
"""把 LongBench-v2 的单个 data.json 拆成 6 个 domain 子目录。

为什么需要这一步：HF 上 `zai-org/LongBench-v2` 仓库**只有一个 `data.json`（单 split）**，
但仓库里所有评估脚本都按 domain 当 config 加载：
    load_dataset(dataset_path, 'single-doc')
所以必须先在本地按 `domain` 字段拆成 6 个目录 + 根 dataset_dict.json。

domain 全名 -> 目录名（实测确认）：
    Single-Document QA                 -> single-doc
    Multi-Document QA                  -> multi-doc
    Code Repository Understanding      -> code-repo
    Long In-context Learning           -> long-icl
    Long-dialogue History Understanding-> long-dialogue
    Long Structured Data Understanding -> long-structured

用法：
    python "train and preprocess/0_prepare_longbench.py"
"""
from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset

from src.paths import LONGBENCH_V2

DOMAIN2DIR = {
    "Single-Document QA": "single-doc",
    "Multi-Document QA": "multi-doc",
    "Code Repository Understanding": "code-repo",
    "Long In-context Learning": "long-icl",
    "Long-dialogue History Understanding": "long-dialogue",
    "Long Structured Data Understanding": "long-structured",
}


def main():
    src = os.path.join(LONGBENCH_V2, "data.json")
    if not os.path.exists(src):
        raise FileNotFoundError(f"找不到 {src}")
    out_root = os.path.join(LONGBENCH_V2, "split_by_domain")

    data = json.load(open(src, encoding="utf-8"))
    print(f"读取 {src}: {len(data)} 题")

    groups = {}
    unknown = set()
    for d in data:
        dom = d["domain"]
        if dom not in DOMAIN2DIR:
            unknown.add(dom)
            continue
        groups.setdefault(DOMAIN2DIR[dom], []).append(d)
    if unknown:
        raise RuntimeError(f"出现未映射的 domain: {unknown}")

    if os.path.isdir(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)

    total = 0
    for name, rows in sorted(groups.items()):
        d = os.path.join(out_root, name)
        Dataset.from_list(rows).save_to_disk(d)
        total += len(rows)
        print(f"  {name:16s} {len(rows):4d} 题 -> {d}")

    with open(os.path.join(out_root, "dataset_dict.json"), "w", encoding="utf-8") as f:
        json.dump({"splits": sorted(groups.keys())}, f, ensure_ascii=False)

    assert total == len(data), f"{total} != {len(data)}"
    print(f"共 {total} 题，6 个 domain，已写入 {out_root}")
    print("LONGBENCH_PREP_DONE")


if __name__ == "__main__":
    main()
