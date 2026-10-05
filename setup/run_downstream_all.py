#!/usr/bin/env python3
"""下游评测总编排：数据分块 -> 合并 -> 并行评测 -> 汇总。

阶段（每步幂等，可重复跑）：
  [0] 检查 vLLM 两实例健康
  [1a] Semantic 分片切分（6 卡）          -> merge_semantic_shards.py
  [1b] Lumber 分片切分（6 worker / 2 vLLM）-> merge_lumber_shards.py
  [1c] Traditional 512（若缺）             已由 9b_chunk_traditional_512.py 产出
  [2] PPL / Margin 后台慢跑（可选，长）      --with-slow
  [3] 并行评测 6 方法 x 6 域                run_eval_parallel.py
  [4] 汇总                                  aggregate_eval.py

用法：
    python setup/run_downstream_all.py --stage chunk      # 只做分块
    python setup/run_downstream_all.py --stage eval       # 只做评测
    python setup/run_downstream_all.py --stage all        # 全流程
    python setup/run_downstream_all.py --stage all --with-slow
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
PY = f"source ~/miniconda/etc/profile.d/conda.sh && conda activate freechunker && cd {ROOT} && export PYTHONPATH={ROOT}"


def run(name, cmd, background=False, log=None):
    """跑一条远端 bash 命令（本地由 python 驱动，命令里已是远端语义）。"""
    print(f"\n{'='*70}\n>>> {name}\n{'='*70}", flush=True)
    t0 = time.time()
    if background:
        log = log or os.path.join(LOGS, f"{name}.log")
        print(f"[bg] {log}")
        return subprocess.Popen(cmd, shell=True)
    r = subprocess.run(cmd, shell=True)
    print(f"<<< {name} rc={r.returncode}  {time.time()-t0:.0f}s", flush=True)
    return r.returncode


def health():
    out = subprocess.run(
        'for p in 8888 8889; do printf "%s=" $p; curl -s -m 3 -o /dev/null -w "%{http_code}\\n" '
        f'http://localhost:{p}/health; done', shell=True, capture_output=True, text=True)
    print(out.stdout)
    return "200" in out.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["chunk", "eval", "agg", "all"])
    ap.add_argument("--with-slow", action="store_true", help="一并后台启动 PPL / Margin")
    ap.add_argument("--semantic-shards", type=int, default=6)
    ap.add_argument("--lumber-workers", type=int, default=6)
    ap.add_argument("--nshards", type=int, default=6)
    ap.add_argument("--max-parallel", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--topk", default="5,10")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)

    if args.dry_run:
        print("[dry-run] 计划：")
        print("  chunk: semantic(6) + lumber(6) [+ ppl/margin if --with-slow]")
        print(f"  eval : {args.nshards} shards, max-parallel={args.max_parallel}, repeats={args.repeats}, topk={args.topk}")
        print("  agg  : aggregate_eval.py")
        return

    if not health():
        print("⚠️  vLLM 未就绪，先起实例：bash setup/start_vllm.sh（GPU0/8888）与 FC_VLLM_GPU=1 FC_VLLM_PORT=8889")
        sys.exit(1)

    if args.stage in ("chunk", "all"):
        # semantic：分片 + 合并（合并放在 runner 返回后）
        r = run("semantic", f'{PY} && python setup/run_semantic_plan.py')
        if r == 0:
            run("semantic_merge", f'{PY} && python setup/merge_semantic_shards.py')
        # lumber：分片 + 合并
        r = run("lumber", f'{PY} && python setup/run_lumber_parallel.py --workers {args.lumber_workers}')
        if r == 0:
            run("lumber_merge", f'{PY} && python setup/merge_lumber_shards.py')
        if args.with_slow:
            run("ppl_bg", f'{PY} && export CUDA_VISIBLE_DEVICES=3 && python -u "train and preprocess/6_chunk_ppl.py"',
                background=True, log=os.path.join(LOGS, "chunk_ppl.log"))
            run("margin_bg", f'{PY} && export CUDA_VISIBLE_DEVICES=4 && python -u "train and preprocess/8_chunk_margin.py"',
                background=True, log=os.path.join(LOGS, "chunk_margin.log"))

    if args.stage in ("eval", "all"):
        run("eval", f'{PY} && python setup/run_eval_parallel.py --nshards {args.nshards} '
                    f'--max-parallel {args.max_parallel} --repeats {args.repeats} '
                    f'--topk {args.topk}')

    if args.stage in ("agg", "all"):
        run("aggregate", f'{PY} && python setup/aggregate_eval.py')

    print("\nDOWNSTREAM_ALL_DONE")


if __name__ == "__main__":
    main()
