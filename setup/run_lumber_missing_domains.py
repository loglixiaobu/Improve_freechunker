#!/usr/bin/env python3
"""补齐 Lumber 的 4 个缺失域（论文 Table 1 的 Lumber 覆盖全部 6 个域）。

背景
----
`5_chunk_lumber.py` 第 98 行硬编码 `selected_domains = ['single-doc','multi-doc']`，
所以只产出 2 个域。已用 `patch_lumber_domains.py` 改为读 `FC_LMB_DOMAINS`。

本脚本按域启动 N 路分片（样本级 `i % NSHARDS == SHARD`），
全部走 HTTP 打 vLLM（GPU0:8888 / GPU1:8889），**不占显存**。
进度靠全局池 `_done_global.jsonl` 跨分片续跑，改分片数不丢进度。

与 PPL 的关系
-------------
- Lumber worker 用 `CUDA_VISIBLE_DEVICES=`（空）→ 0 MiB 显存，不会抢 PPL 的卡。
- 只吃 CPU（每 worker 约 1 核）与内存（几百 MB）。72 核 / 481 GB 余量充足。
- 上限：不把 vLLM 打爆。实测 96 并发仍线性（25.7 req/s），所以 40 个 worker 是安全的。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "LongBench-v2_chunked", "Lumber", "qwen3-8b", "checkpoints")
LOGS = os.path.join(ROOT, "logs")
PY = "/home1/lh/miniconda/envs/freechunker/bin/python"
SCRIPT = os.path.join(ROOT, "train and preprocess", "5_chunk_lumber.py")
PORTS = ["8888", "8889"]

TOTALS = {"single-doc": 175, "multi-doc": 125, "long-icl": 81,
          "code-repo": 50, "long-dialogue": 39, "long-structured": 33}

# 目标：每片 <= 25 分钟。实测 long-dialogue 中位 43K tok -> 50.4s/sample。
# 其余域按 token 比例外推。
PLAN = {
    "long-dialogue": 2,
    "long-structured": 6,
    "long-icl": 12,
    "code-repo": 18,
}


def now() -> str:
    return time.strftime("%H:%M:%S")


def log(m: str) -> None:
    print(f"[lmb] {now()} {m}", flush=True)


def pool_done() -> dict[str, dict]:
    p = os.path.join(CKPT, "_done_global.jsonl")
    d: dict[str, dict] = {}
    if not os.path.isfile(p):
        return d
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                d[r["_id"]] = r
            except Exception:
                pass
    return d


def domain_done() -> dict[str, int]:
    """按 domain 字段统计全局池里的完成数。"""
    from collections import Counter
    return Counter(r.get("domain", "?") for r in pool_done().values())


def launch(dn: str, shard: int, nshards: int) -> None:
    session = f"lmb_{dn}_{shard}"
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
    port = PORTS[shard % len(PORTS)]
    inner = (
        "source ~/miniconda/etc/profile.d/conda.sh && conda activate freechunker && "
        f"export PYTHONPATH='{ROOT}' && "
        "export CUDA_VISIBLE_DEVICES= && "
        "export TOKENIZERS_PARALLELISM=false && "
        "export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1 && "
        f"export FC_LMB_DOMAINS='{dn}' && "
        f"export FC_LMB_SHARD='{shard}' && "
        f"export FC_LMB_NSHARDS='{nshards}' && "
        f"export FC_VLLM_PORT='{port}' && "
        f"cd '{ROOT}' && {PY} '{SCRIPT}'"
    )
    logf = os.path.join(LOGS, f"lmb_{dn}_sh{shard}.log")
    full = f"{inner} > {logf} 2>&1; echo EXIT=$? >> {logf}; sleep 99999"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, full])
    log(f"启动 {dn} shard{shard}/{nshards} -> vLLM:{port}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="long-dialogue,long-structured,long-icl,code-repo")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="分片数缩放（<1 少开 worker，>1 多开）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll", type=int, default=180)
    args = ap.parse_args()

    doms = [d.strip() for d in args.domains.split(",") if d.strip()]
    os.makedirs(LOGS, exist_ok=True)

    got = domain_done()
    total_workers = 0
    tasks: list[tuple[str, int, int]] = []
    for dn in doms:
        ns = max(1, int(round(PLAN.get(dn, 1) * args.scale)))
        tasks += [(dn, s, ns) for s in range(ns)]
        total_workers += ns

    log(f"目标域 {doms}，共 {total_workers} 个 worker")
    for dn in doms:
        log(f"    {dn:<16} 完成 {got.get(DOMAIN_LABEL.get(dn, dn), 0)}/{TOTALS.get(dn)}  "
            f"分片 {PLAN.get(dn,1)}")
    if args.dry_run:
        log("DRY_RUN_DONE")
        return 0

    for dn, s, ns in tasks:
        launch(dn, s, ns)

    log(f"全部启动，看护中（每 {args.poll}s 报一次）")
    while True:
        time.sleep(args.poll)
        g = domain_done()
        left = {}
        for dn in doms:
            c = g.get(DOMAIN_LABEL.get(dn, dn), 0)
            if c < TOTALS[dn]:
                left[dn] = f"{c}/{TOTALS[dn]}"
        wk = subprocess.run(["pgrep", "-fc", "5_chunk_lumber.py"],
                            capture_output=True, text=True).stdout.strip()
        log(f"worker={wk}  剩余 {left if left else '全部完成 ✅'}")
        if not left:
            log("LUMBER4_DONE")
            return 0


# Lumber 记录的 domain 字段是论文里的长名，不是目录名
DOMAIN_LABEL = {
    "single-doc": "Single-Document QA",
    "multi-doc": "Multi-Document QA",
    "code-repo": "Code Repository Understanding",
    "long-icl": "Long In-context Learning",
    "long-dialogue": "Long-dialogue History Understanding",
    "long-structured": "Long Structured Data Understanding",
}

if __name__ == "__main__":
    sys.exit(main())
