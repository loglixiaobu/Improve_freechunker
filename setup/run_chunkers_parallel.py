#!/usr/bin/env python3
"""并行跑 4 个 chunker（PPL / Margin / Semantic / Lumber）。

分工（对应小布的要求：一张卡跑一个任务）：
  * lumber   -> GPU2, 纯 vLLM 客户端（打 http://localhost:8888），几乎不占显存
  * ppl      -> GPU3, 本地 Qwen2-1.5B on cuda
  * margin   -> GPU4, 本地 Qwen2-1.5B on cuda
  * semantic -> GPU5, 本地 bge-m3 on cuda

GPU 0/1 留给两个 vLLM 实例，2-7 是空闲卡。
先做小样本冒烟（--limit），通过后再全量。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")

# name -> (script, gpu, 额外 env)
JOBS = {
    "ppl": ("6_chunk_ppl.py", "3", {}),
    "margin": ("8_chunk_margin.py", "4", {}),
    "semantic": ("7_chunk_semantic.py", "5", {}),
    "lumber": ("5_chunk_lumber.py", "2", {}),
}


def build_cmd(name: str, limit: int | None):
    script, gpu, extra = JOBS[name]
    script_path = os.path.join(ROOT, "train and preprocess", script)
    env = (
        "source ~/miniconda/etc/profile.d/conda.sh && conda activate freechunker && "
        f"export PYTHONPATH={ROOT} && "
        f"export CUDA_VISIBLE_DEVICES={gpu} && "
        f"export TOKENIZERS_PARALLELISM=false && "
        f"export HF_ENDPOINT=https://hf-mirror.com && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && "
    )
    for k, v in extra.items():
        env += f"export {k}={v} && "
    cmd = f'bash -lc "{env} cd {ROOT} && python \'{script_path}\'"'
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="ppl,margin,semantic,lumber")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="保留参数；chunker 脚本本身不支持 limit，仅供说明")
    args = ap.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    names = [n.strip() for n in args.jobs.split(",") if n.strip()]
    bad = [n for n in names if n not in JOBS]
    if bad:
        print(f"[err] 未知任务: {bad}，可选 {list(JOBS)}")
        sys.exit(2)

    procs = {}
    for name in names:
        script, gpu, _ = JOBS[name]
        cmd = build_cmd(name, args.limit)
        log = os.path.join(LOGS, f"chunk_{name}.log")
        print(f"[launch] {name:9s} gpu={gpu}  -> {log}")
        if args.dry_run:
            print(f"         {cmd}")
            continue
        f = open(log, "w")
        p = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        procs[name] = (p, f, log, time.time())

    if args.dry_run:
        print("\n--- dry-run ---")
        return

    print(f"\n[parallel] 已启动 {len(procs)} 个 chunker，等待全部完成 ...")
    rc = {}
    while procs:
        for name in list(procs):
            p, f, log, t0 = procs[name]
            r = p.poll()
            if r is not None:
                f.close()
                rc[name] = r
                dt = time.time() - t0
                print(f"[done] {name:9s} rc={r}  {dt/60:.1f} min  <- {log}")
                del procs[name]
        time.sleep(5)

    print("\n=== 汇总 ===")
    for name, r in rc.items():
        print(f"  {name:9s} {'OK' if r == 0 else 'FAIL rc=' + str(r)}")
    print("CHUNK_PARALLEL_DONE")
    sys.exit(0 if all(v == 0 for v in rc.values()) else 1)


if __name__ == "__main__":
    main()
