#!/usr/bin/env python3
"""下游评测总编排：分块 → 评测（FreeChunker + Traditional）。

回答的问题：「论文 §5 的下游评测能不能跑通，FreeChunker 比传统分块好在哪」。

范围（按小布的决定）：
  - 方法：FreeChunker + Traditional（最小闭环，先验证链路）
  - 嵌入模型：只用 jina（另两个 bge-m3 / nomic 没训）
  - 域：single-doc + multi-doc（传统分块的 LongBench-v2 里这两个最有代表性）
  - TopK：5, 10（仓库默认）
  - repeats：1（官方是 3，但 deterministic temperature=0，第 2/3 次是纯重复）

阶段：
  0. 起 vLLM            （setup/start_vllm.sh，单卡，端口 8888）
  1. 生成 Traditional 分块（9_chunk_traditional.py，chunk_size=256）
  2. FreeChunker 评测    （test/eval_downstream.py --method freechunker）
  3. Traditional 评测    （test/eval_downstream.py --method traditional）

每个阶段幂等：产物存在就跳过（分块目录 / eval_summary 文件）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.expanduser("~/FreeChunker")
CONDA = os.path.expanduser("~/miniconda/etc/profile.d/conda.sh")
PY_ENV = "freechunker"

DOMAINS = "single-doc,multi-doc"
TOPK = "5,10"
REPEATS = 1
CHUNK_SIZE = 256

# vLLM 独占 GPU 0（22G），嵌入模型/编码器必须换一张卡，否则 OOM。
# 其余 7 张 3090 全空。
GPU_FOR_ENCODER = os.environ.get("FC_EVAL_GPU", "1")


def sh(cmd: str, log: str | None = None, check: bool = True):
    """在已激活 conda 环境的子 shell 里跑命令。"""
    full = (
        f"source {CONDA} && conda activate {PY_ENV} && export PYTHONPATH={ROOT} "
        f"&& export HF_ENDPOINT=https://hf-mirror.com "
        f"&& export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 "
        f"&& export TOKENIZERS_PARALLELISM=false "
        f"&& export CUDA_VISIBLE_DEVICES={GPU_FOR_ENCODER} "
        f"&& cd {ROOT} && {cmd}"
    )
    print(f"[run] {cmd}")
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n===== {time.strftime('%F %T')} :: {cmd} =====\n")
            p = subprocess.run(["bash", "-lc", full], stdout=f, stderr=subprocess.STDOUT)
    else:
        p = subprocess.run(["bash", "-lc", full])
    if check and p.returncode != 0:
        raise SystemExit(f"[FAIL] 命令退出码 {p.returncode}: {cmd}")
    return p.returncode


def wait_vllm(timeout=900):
    import urllib.request

    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen("http://localhost:8888/health", timeout=4) as r:
                if r.status == 200:
                    print(f"[vllm] 就绪（{time.time()-t0:.0f}s）")
                    return True
        except Exception:
            pass
        # 服务挂了就别傻等
        if os.system("pgrep -f 'vllm.entrypoints' > /dev/null") != 0:
            print("[vllm] 进程不在，尝试拉起")
            sh(f"bash setup/start_vllm.sh > logs/vllm.log 2>&1 &", check=False)
        time.sleep(15)
    raise SystemExit("[FAIL] vLLM 超时未就绪")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-vllm", action="store_true")
    ap.add_argument("--skip-chunk", action="store_true")
    ap.add_argument("--only", default="", help="freechunker / traditional")
    ap.add_argument("--domains", default=DOMAINS)
    args = ap.parse_args()

    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    os.makedirs(f"{ROOT}/eval_results", exist_ok=True)

    if not args.skip_vllm:
        print("=== 阶段 0: vLLM ===")
        sh("bash setup/start_vllm.sh > logs/vllm.log 2>&1 &", check=False)
        wait_vllm()

    if args.only not in ("traditional",) and not args.skip_chunk:
        print("=== 阶段 1: Traditional 分块 ===")
        done = os.path.isdir(f"{ROOT}/LongBench-v2_chunked/Traditional/{CHUNK_SIZE}/dataset_dict.json")
        if done:
            print(f"[skip] 分块已存在 {CHUNK_SIZE}")
        else:
            sh('python "train and preprocess/9_chunk_traditional.py"', log=f"{ROOT}/logs/chunk_traditional.log")

    if args.only in ("", "freechunker"):
        print("=== 阶段 2: FreeChunker 评测 ===")
        sh(f"python test/eval_downstream.py --method freechunker --domains {args.domains} "
           f"--topk {TOPK} --repeats {REPEATS}",
           log=f"{ROOT}/logs/eval_freechunker.log")

    if args.only in ("", "traditional"):
        print("=== 阶段 3: Traditional 评测 ===")
        sh(f"python test/eval_downstream.py --method traditional --domains {args.domains} "
           f"--topk {TOPK} --repeats {REPEATS} --chunk-sizes {CHUNK_SIZE}",
           log=f"{ROOT}/logs/eval_traditional.log")

    print("EVAL_ALL_DONE")


if __name__ == "__main__":
    main()
