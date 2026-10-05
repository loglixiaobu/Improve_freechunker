#!/usr/bin/env python3
"""把 Margin 的 code-repo 尾部（跑得慢的那 18 条）重分片到多张空卡上。

为什么
------
code-repo 单进程在 GPU6 上 32/50，平均 468 s/样本，剩 18 条 ≈ 2.3 h。
而 GPU 0/1/5/7 全空着 → 2.3 h 的瓶颈完全可以用 5 张卡压到 ~31 min。

为什么安全（不会丢已完成的 32 条）
--------------------------------
32 条成果只在 checkpoint `checkpoints/code-repo.jsonl` 里（worker 没跑完，
还没 `save_to_disk`）。`merge_cm_shards.py` **v2** 已支持从 checkpoint 读，
所以 kill 掉 worker 后，那 32 条在合并时会被自动补回。
（v1 只读目录，会丢 —— 这是 v2 存在的理由。）

分片方式
--------
用 `FC_CM_INDEXES` 显式指定下标（chunker 脚本已支持），并按 **context 长度
做贪心装箱**，让 5 个 worker 的预计耗时尽量均衡（样本耗时 ≈ 正比于长度）。

输出文件用 `.shard5xx` 后缀（借 `FC_CM_SHARD=500+i, FC_CM_NSHARDS=1000`），
这样合并脚本按 `<domain>.shard*` 规则自然捡得到，不需要新规则。

用法:
    python setup/reshard_margin_tail.py --dry    # 只打印计划
    python setup/reshard_margin_tail.py          # 真跑
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
PY = "/home1/lh/miniconda/envs/freechunker/bin/python"
SCRIPT = os.path.join(ROOT, "train and preprocess", "8_chunk_margin.py")
CK = os.path.join(ROOT, "LongBench-v2_chunked", "Margin",
                  "Qwen2.5-1.5B-Instruct", "checkpoints")

DOMAIN = "code-repo"
GPUS = [0, 1, 5, 6, 7]
KILL_SESS = "m_code-repo_g6"      # 正在跑的旧 worker 会话
SHARD_BASE = 500                  # 借 shard 后缀，产出 <domain>.shard50x/


def remaining_indexes():
    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(ROOT, "datasets", "LongBench-v2",
                                     "split_by_domain", DOMAIN))
    rows = [str(ds[i]["_id"]) for i in range(len(ds))]
    lens = [len(ds[i].get("context") or "") for i in range(len(ds))]
    have = set()
    for f in glob.glob(os.path.join(CK, DOMAIN + "*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        have.add(json.loads(line)["_id"])
                    except Exception:
                        pass
    rem = [i for i, r in enumerate(rows) if r not in have]
    return rem, lens


def pack(rem, lens, k):
    """按长度贪心装箱（LPT）：每轮把最长的放进当前最轻的箱子。"""
    order = sorted(rem, key=lambda i: -lens[i])
    bins = [[] for _ in range(k)]
    load = [0] * k
    for i in order:
        j = load.index(min(load))
        bins[j].append(i)
        load[j] += lens[i]
    return bins, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    rem, lens = remaining_indexes()
    print(f"[tail] {DOMAIN} 剩余 {len(rem)} 条: {rem}")
    if not rem:
        print("[tail] 没有剩余，退出")
        return 0
    k = min(len(GPUS), len(rem))
    bins, load = pack(rem, lens, k)
    print(f"[tail] 按 context 长度贪心装箱到 {k} 组（目标：各组总长接近）：")
    for j, (b, L) in enumerate(zip(bins, load)):
        print(f"   组{j} GPU{GPUS[j]}: {len(b)} 条 idx={sorted(b)} 总长={L}")

    if args.dry:
        print("[tail] --dry，不实际启动")
        return 0

    # 1) 停掉旧 worker
    r = subprocess.run(["tmux", "has-session", "-t", KILL_SESS],
                       stderr=subprocess.DEVNULL)
    if r.returncode == 0:
        subprocess.run(["tmux", "kill-session", "-t", KILL_SESS], check=False)
        print(f"[tail] 已停掉旧 worker {KILL_SESS}")
        time.sleep(5)
    else:
        print(f"[tail] 旧 worker {KILL_SESS} 不在（可能已退出）")

    # 2) 铺分片
    for j, (b, L) in enumerate(zip(bins, load)):
        if not b:
            continue
        gpu = GPUS[j]
        shard = SHARD_BASE + j
        sess = f"mt_{DOMAIN}_s{shard}_g{gpu}"
        log = os.path.join(LOGS, f"margin_{DOMAIN}_sh{shard}_gpu{gpu}.log")
        inner = (
            "source ~/miniconda/etc/profile.d/conda.sh && "
            "conda activate freechunker && "
            f"export PYTHONPATH='{ROOT}' && "
            f"export CUDA_VISIBLE_DEVICES='{gpu}' && "
            "export TOKENIZERS_PARALLELISM=false && "
            "export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1 && "
            f"export FC_CM_DOMAINS='{DOMAIN}' && "
            f"export FC_CM_SHARD='{shard}' && export FC_CM_NSHARDS='1000' && "
            f"export FC_CM_INDEXES='{','.join(str(i) for i in sorted(b))}' && "
            f"cd '{ROOT}' && {PY} '{SCRIPT}' "
            f"> {log} 2>&1; echo EXIT=$? >> {log}; sleep 99999")
        subprocess.run(["tmux", "kill-session", "-t", sess],
                       stderr=subprocess.DEVNULL)
        subprocess.run(["tmux", "new-session", "-d", "-s", sess, inner],
                       check=False)
        print(f"[tail] 启动 {DOMAIN} 组{j} {len(b)}条 -> GPU{gpu}  ({sess})")
        time.sleep(4)

    print("[tail] RESHARD_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
