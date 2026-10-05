#!/usr/bin/env python3
"""启动 Margin 剩余两个域（single-doc / code-repo）到空卡。

背景：`run_margin_short.py` 已经占了 GPU3/4/7 跑 long-dialogue / multi-doc / long-icl。
本脚本只负责把**剩余未启动**的两个域铺到空卡上：
    single-doc -> GPU2   (175 条，最大域)
    code-repo  -> GPU6   (50 条)

Margin 一进程占满一张卡（模型 load 进进程，~16-24 GB），所以一卡一进程。
跑完后需手工 `merge_cm_shards.py --method margin`。

⚠️ 计数要同时认 {d}.jsonl 与 {d}.shard*.jsonl 两种形态（踩过坑）。
"""
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

PLAN = [
    ("single-doc", 2),
    ("code-repo", 6),
]
TOTALS = {"single-doc": 175, "code-repo": 50}


def done_ids(domain):
    """按 _id 去重计完成数；两种文件名形态都认。"""
    if not os.path.isdir(CK):
        return 0
    ids = set()
    for fn in os.listdir(CK):
        if not fn.endswith(".jsonl"):
            continue
        stem = fn[:-len(".jsonl")]
        if stem != domain and not stem.startswith(domain + ".shard"):
            continue
        with open(os.path.join(CK, fn)) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line)["_id"])
                    except Exception:
                        pass
    return len(ids)


def launch(domain, gpu):
    env = (
        "source ~/miniconda/etc/profile.d/conda.sh && conda activate freechunker && "
        f"export PYTHONPATH='{ROOT}' && "
        f"export CUDA_VISIBLE_DEVICES='{gpu}' && "
        "export TOKENIZERS_PARALLELISM=false && "
        "export HF_HUB_OFFLINE=1 && "
        "export TRANSFORMERS_OFFLINE=1 && "
        f"export FC_CM_DOMAINS='{domain}' && "
        "export FC_CM_SHARD='0' && "
        "export FC_CM_NSHARDS='1' && "
    )
    log = os.path.join(LOGS, f"margin_{domain}_gpu{gpu}_sh0.log")
    inner = (f"bash -lc \"{env} cd '{ROOT}' && {PY} '{SCRIPT}'\" "
             f"> {log} 2>&1; echo EXIT=$? >> {log}; sleep 99999")
    sess = f"m_{domain}_g{gpu}"
    subprocess.run(["tmux", "kill-session", "-t", sess],
                   stderr=subprocess.DEVNULL)
    subprocess.run(["tmux", "new-session", "-d", "-s", sess, inner], check=False)
    print(f"[margin2] 启动 {domain} -> GPU{gpu}  tmux={sess}", flush=True)


def main():
    already = {d: done_ids(d) for d, _ in PLAN}
    print(f"[margin2] 启动前进度 {already}", flush=True)
    for domain, gpu in PLAN:
        if already[domain] >= TOTALS[domain]:
            print(f"[margin2] {domain} 已完成，跳过", flush=True)
            continue
        launch(domain, gpu)
        time.sleep(3)

    print("[margin2] 全部启动，看护中（每 180s 报一次）", flush=True)
    while True:
        st = {d: f"{done_ids(d)}/{TOTALS[d]}" for d, _ in PLAN}
        left = [d for d in TOTALS if done_ids(d) < TOTALS[d]]
        stamp = time.strftime("%H:%M:%S")
        tail = "" if left else "  全部完成 ✅"
        print(f"[margin2] {stamp} 进度 {st}{tail}", flush=True)
        if not left:
            print("[margin2] MARGIN_REST_DONE", flush=True)
            return 0
        time.sleep(180)


if __name__ == "__main__":
    sys.exit(main())
