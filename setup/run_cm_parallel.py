#!/usr/bin/env python3
"""并行跑 PPL / Margin chunker，**显式指定每个 worker 跑哪些域**。

为什么不用 `i % nshards`
------------------------
PPL 和 Margin 各自的域顺序不一样（PPL 走 `sorted()` 字典序，Margin 走手写顺序），
按位置分片很容易两张卡分到同一批大域、算力白浪费。
这里改成**直接写死「哪张卡跑哪几个域」**，谁能干谁多干，一目了然。

GPU 数 < 域数时，把多个域塞给同一 worker（顺序执行，checkpoint 保证不丢）。

用法
----
    # 6 张卡各跑 1 个域（最均衡，推荐）
    python setup/run_cm_parallel.py --method ppl --plan balanced

    # 自定义：gpu2 跑 code-repo，gpu3 跑 single-doc+multi-doc，...
    python setup/run_cm_parallel.py --method ppl \
        --assign "2=code-repo;3=single-doc,multi-doc;4=long-icl;5=long-dialogue;6=long-structured"

    # 两种方法同时铺满 6 张卡（各 3 张）
    python setup/run_cm_parallel.py --method both --gpus 2,3,4,5,6,7

负载依据（LongBench-v2 本仓库 503 样本，字符量）
------------------------------------------------
    code-repo      178.0M  ← 最大，单卡独占
    single-doc      77.1M  ← 次大
    long-icl        65.9M
    multi-doc       65.0M
    long-structured 39.9M
    long-dialogue   12.7M  ← 最小
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")

SCRIPTS = {
    "ppl": "6_chunk_ppl.py",
    "margin": "8_chunk_margin.py",
}

ALL_DOMAINS = ["single-doc", "multi-doc", "code-repo", "long-dialogue",
               "long-icl", "long-structured"]

# 按估算字符量「贪心」分配到 N 组：每次把当前最重的域塞给累计最轻的那组。
# 用字符量而不是样本数 —— code-repo 只有 50 题但 178M 字符，按题数分会被严重低估。
DOMAIN_CHARS = {
    "code-repo": 178.0,
    "single-doc": 77.1,
    "long-icl": 65.9,
    "multi-doc": 65.0,
    "long-structured": 39.9,
    "long-dialogue": 12.7,
}


def balanced_plan(gpus: list[str]) -> dict[str, list[str]]:
    """把 6 个域按字符量贪心分到 len(gpus) 组。"""
    n = len(gpus)
    groups: list[list[str]] = [[] for _ in range(n)]
    load = [0.0] * n
    for dn in sorted(ALL_DOMAINS, key=lambda d: -DOMAIN_CHARS[d]):
        i = load.index(min(load))
        groups[i].append(dn)
        load[i] += DOMAIN_CHARS[dn]
    plan = {}
    for gpu, g in zip(gpus, groups):
        if g:
            plan[gpu] = g
    return plan


def parse_assign(spec: str) -> dict[str, list[str]]:
    """'2=code-repo;3=single-doc,multi-doc' -> {'2': [...], '3': [...]}"""
    plan = {}
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"[err] --assign 片段缺 '=': {chunk!r}")
        gpu, doms = chunk.split("=", 1)
        plan[gpu.strip()] = [d.strip() for d in doms.split(",") if d.strip()]
    return plan


def launch(method: str, plan: dict[str, list[str]], dry_run: bool):
    script = os.path.join(ROOT, "train and preprocess", SCRIPTS[method])
    procs = []
    for gpu, doms in plan.items():
        env = (
            "source ~/miniconda/etc/profile.d/conda.sh && conda activate freechunker && "
            f"export PYTHONPATH={ROOT} && "
            f"export CUDA_VISIBLE_DEVICES={gpu} && "
            "export TOKENIZERS_PARALLELISM=false && "
            "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && "
            f"export FC_CM_DOMAINS={','.join(doms)} && "
            "export FC_CM_SHARD=0 FC_CM_NSHARDS=1 && "
        )
        tag = "-".join(d[:4] for d in doms)
        log = os.path.join(LOGS, f"chunk_{method}_gpu{gpu}_{tag}.log")
        cmd = f'bash -lc "{env} cd {ROOT} && python \'{script}\'"'
        chars = sum(DOMAIN_CHARS.get(d, 0) for d in doms)
        print(f"[launch] {method:6s} gpu={gpu:>2s} chars={chars:6.1f}M  {doms}")
        if dry_run:
            print("         ", cmd)
            continue
        f = open(log, "w")
        procs.append((f"{method}@gpu{gpu}", subprocess.Popen(
            cmd, shell=True, stdout=f, stderr=subprocess.STDOUT), f, log, time.time()))
    return procs


def wait_all(procs):
    rc = {}
    while procs:
        for item in list(procs):
            name, p, f, log, t0 = item
            r = p.poll()
            if r is not None:
                f.close()
                rc[name] = r
                print(f"[done] {name} rc={r}  {(time.time()-t0)/60:.1f} min")
                procs.remove(item)
        time.sleep(5)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="ppl", choices=["ppl", "margin", "both"])
    ap.add_argument("--gpus", default="2,3,4,5,6,7", help="逗号分隔")
    ap.add_argument("--assign", default=None,
                    help="显式指派 '2=code-repo;3=single-doc,multi-doc'")
    ap.add_argument("--plan", default="balanced", choices=["balanced"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    if args.assign:
        plan = parse_assign(args.assign)
    elif len(gpus) >= len(ALL_DOMAINS):
        plan = {g: [d] for g, d in zip(gpus, ALL_DOMAINS)}
    else:
        plan = balanced_plan(gpus)

    print("[plan]")
    for g, d in plan.items():
        print(f"   gpu{g}: {d}  ({sum(DOMAIN_CHARS.get(x,0) for x in d):.1f}M chars)")

    methods = ["ppl", "margin"] if args.method == "both" else [args.method]
    procs = []
    for m in methods:
        procs += launch(m, plan, args.dry_run)

    if args.dry_run:
        print("\n--- dry-run ---")
        return

    print(f"\n[cm] 已启动 {len(procs)} 个 worker ...")
    rc = wait_all(procs)
    print("\n=== 汇总 ===")
    for k, v in sorted(rc.items()):
        print(f"  {k}: {'OK' if v == 0 else 'FAIL rc=' + str(v)}")
    print("CM_DONE")
    sys.exit(0 if all(v == 0 for v in rc.values()) else 1)


if __name__ == "__main__":
    main()
