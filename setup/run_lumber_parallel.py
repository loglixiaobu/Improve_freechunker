#!/usr/bin/env python3
"""并行跑 Lumber chunker：按样本分片，多 worker 打同一个/多个 vLLM 实例。

Lumber 每篇要发 ~100+ 次 vLLM 请求且**串行**（vLLM `Running: 0~1`，GPU 空转），
所以把 300 篇（single-doc 175 + multi-doc 125）按样本切开，多个 worker 并发打 vLLM。

worker 轮流用 8888 / 8889 两个 vLLM 实例，避免单实例排队。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
SCRIPT = os.path.join(ROOT, "train and preprocess", "5_chunk_lumber.py")
PORTS = ["8888", "8889"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)

    procs = []
    for w in range(args.workers):
        port = PORTS[w % len(PORTS)]
        env = (
            "source ~/miniconda/etc/profile.d/conda.sh && conda activate freechunker && "
            f"export PYTHONPATH={ROOT} && export CUDA_VISIBLE_DEVICES= && "
            "export TOKENIZERS_PARALLELISM=false && "
            "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && "
            f"export FC_LMB_SHARD={w} && export FC_LMB_NSHARDS={args.workers} && "
            f"export FC_VLLM_PORT={port} && "
        )
        log = os.path.join(LOGS, f"chunk_lumber_w{w}.log")
        cmd = f'bash -lc "{env} cd {ROOT} && python \'{SCRIPT}\'"'
        print(f"[launch] worker{w} shard={w}/{args.workers} vllm={port} -> {log}")
        if args.dry_run:
            print("        ", cmd)
            continue
        f = open(log, "w")
        procs.append((w, subprocess.Popen(cmd, shell=True, stdout=f,
                                          stderr=subprocess.STDOUT), f, log, time.time()))

    if args.dry_run:
        print("\n--- dry-run ---")
        return

    print(f"\n[lumber] {len(procs)} 个 worker 已启动 ...")
    rc = {}
    while procs:
        for item in list(procs):
            w, p, f, log, t0 = item
            r = p.poll()
            if r is not None:
                f.close()
                rc[w] = r
                print(f"[done] worker{w} rc={r}  {(time.time()-t0)/60:.1f} min")
                procs.remove(item)
        time.sleep(5)

    print("\n=== 汇总 ===")
    for w, r in sorted(rc.items()):
        print(f"  worker{w}: {'OK' if r == 0 else 'FAIL rc=' + str(r)}")
    print("LUMBER_DONE")
    sys.exit(0 if all(v == 0 for v in rc.values()) else 1)


if __name__ == "__main__":
    main()
