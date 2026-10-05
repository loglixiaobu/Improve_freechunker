#!/usr/bin/env python3
"""按域分片并行跑 Semantic chunker（bge-m3）。

为什么：Semantic 实测 ~16s/样本，503 样本单卡要 2.2h。
按域分到 6 张卡（GPU2-7），每片只剩自己那几个域，wall-clock 降到 ~25min。

实现：不改 7_chunk_semantic.py 的逻辑，只给它注入 FC_SEM_DOMAINS 环境变量，
让它只处理指定域。每个 worker 一个独立进程 + 独立 CUDA_VISIBLE_DEVICES。
写 checkpoint 是 per-domain 文件，所以多进程写不同域不会互相覆盖。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")

ALL_DOMAINS = ["single-doc", "multi-doc", "code-repo", "long-dialogue",
               "long-icl", "long-structured"]
GPUS = ["2", "3", "4", "5", "6", "7"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nshards", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    script = os.path.join(ROOT, "train and preprocess", "7_chunk_semantic.py")

    procs = []
    for shard in range(args.nshards):
        picked = ALL_DOMAINS[shard::args.nshards]
        if not picked:
            continue
        gpu = GPUS[shard % len(GPUS)]
        env = (
            "source ~/miniconda/etc/profile.d/conda.sh && conda activate freechunker && "
            f"export PYTHONPATH={ROOT} && export CUDA_VISIBLE_DEVICES={gpu} && "
            "export TOKENIZERS_PARALLELISM=false && export HF_ENDPOINT=https://hf-mirror.com && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && "
            f"export FC_SEM_DOMAINS='{','.join(picked)}' && "
        )
        log = os.path.join(LOGS, f"chunk_semantic_shard{shard}.log")
        cmd = f'bash -lc "{env} cd {ROOT} && python \'{script}\'"'
        print(f"[launch] shard{shard} gpu={gpu} domains={picked} -> {log}")
        if args.dry_run:
            print("        ", cmd)
            continue
        f = open(log, "w")
        procs.append((shard, subprocess.Popen(cmd, shell=True, stdout=f,
                                              stderr=subprocess.STDOUT), f, log, time.time()))

    if args.dry_run:
        print("\n--- dry-run ---")
        return

    print(f"\n[semantic] 已启动 {len(procs)} 片，等待 ...")
    rc = {}
    while procs:
        for item in list(procs):
            shard, p, f, log, t0 = item
            r = p.poll()
            if r is not None:
                f.close()
                rc[shard] = r
                print(f"[done] shard{shard} rc={r}  {(time.time()-t0)/60:.1f} min")
                procs.remove(item)
        time.sleep(5)

    print("\n=== 汇总 ===")
    for s, r in sorted(rc.items()):
        print(f"  shard{s}: {'OK' if r == 0 else 'FAIL rc=' + str(r)}")
    print("SEMANTIC_SHARD_DONE")
    sys.exit(0 if all(v == 0 for v in rc.values()) else 1)


if __name__ == "__main__":
    main()
