#!/usr/bin/env python3
"""Semantic chunker 的显式分片计划（按字符量均衡，而不是简单轮转）。

实测吞吐 ~16s/样本（bge-m3 + percentile），所以按"域 + 样本二级分片"铺满 6 张卡。
字符量参考（split_by_domain）：
  code-repo 178.0M / long-icl 65.9M / multi-doc 65.0M / single-doc 77.1M
  long-structured 39.9M / long-dialogue 12.7M
code-repo 一家占 40%，所以单独劈成 2 片。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
SCRIPT = os.path.join(ROOT, "train and preprocess", "7_chunk_semantic.py")

# (gpu, [domains], sample_shard, sample_nshards, 备注)
PLAN = [
    ("2", ["single-doc"], 0, 1, "single-doc 175样本"),
    ("3", ["multi-doc"], 0, 1, "multi-doc 125样本"),
    ("4", ["code-repo"], 0, 2, "code-repo 前半"),
    ("5", ["code-repo"], 1, 2, "code-repo 后半"),
    ("6", ["long-icl", "long-dialogue"], 0, 1, "long-icl + long-dialogue"),
    ("7", ["long-structured"], 0, 1, "long-structured"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)

    procs = []
    for idx, (gpu, domains, shard, nshards, note) in enumerate(PLAN):
        env = (
            "source ~/miniconda/etc/profile.d/conda.sh && conda activate freechunker && "
            f"export PYTHONPATH={ROOT} && export CUDA_VISIBLE_DEVICES={gpu} && "
            "export TOKENIZERS_PARALLELISM=false && export HF_ENDPOINT=https://hf-mirror.com && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && "
            f"export FC_SEM_DOMAINS='{','.join(domains)}' && "
            f"export FC_SEM_SHARD={shard} && export FC_SEM_NSHARDS={nshards} && "
        )
        log = os.path.join(LOGS, f"chunk_semantic_g{gpu}.log")
        cmd = f'bash -lc "{env} cd {ROOT} && python \'{SCRIPT}\'"'
        print(f"[launch] gpu={gpu} domains={domains} shard={shard}/{nshards}  ({note})")
        if args.dry_run:
            print("        ", cmd)
            continue
        f = open(log, "w")
        procs.append((gpu, subprocess.Popen(cmd, shell=True, stdout=f,
                                            stderr=subprocess.STDOUT), f, log, time.time()))

    if args.dry_run:
        print("\n--- dry-run ---")
        return

    print(f"\n[semantic] {len(procs)} 片已启动 ...")
    rc = {}
    while procs:
        for item in list(procs):
            gpu, p, f, log, t0 = item
            r = p.poll()
            if r is not None:
                f.close()
                rc[gpu] = r
                print(f"[done] gpu={gpu} rc={r}  {(time.time()-t0)/60:.1f} min")
                procs.remove(item)
        time.sleep(5)

    print("\n=== 汇总 ===")
    for g, r in sorted(rc.items()):
        print(f"  gpu{g}: {'OK' if r == 0 else 'FAIL rc=' + str(r)}")
    print("SEMANTIC_PLAN_DONE")
    sys.exit(0 if all(v == 0 for v in rc.values()) else 1)


if __name__ == "__main__":
    main()
