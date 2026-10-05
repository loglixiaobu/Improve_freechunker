#!/usr/bin/env python3
"""在空出的 GPU 上跑 Margin 的「短」域。

实测速率（logs/timing_margin.log）：38.6 s/sample on single-doc
→ 0.2783 s/K token。据此估算各域单卡工期：

    long-dialogue   12s/sample * 39  ≈  8 min
    long-structured 59s/sample * 33  ≈ 32 min
    long-icl        51s/sample * 81  ≈ 68 min
    multi-doc       65s/sample * 125 ≈ 135 min
    single-doc      39s/sample * 175 ≈ 114 min
    code-repo      123s/sample * 50  ≈ 102 min

本脚本只跑前三个「短」域，各占一张空卡（GPU3/5/7），nshards=1。
Margin 跟 PPL 一样把模型 load 进进程（device="cuda"），一个进程占满一张卡，
所以必须一卡一进程。跑完后手动 merge_cm_shards.py --method margin。
"""
import json
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
PY = "/home1/lh/miniconda/envs/freechunker/bin/python"
SCRIPT = os.path.join(ROOT, "train and preprocess", "8_chunk_margin.py")

# (domain, gpu) —— 按实测工期从短到长占卡
PLAN = [
    ("long-dialogue",   3),
    ("long-structured", 5),
    ("long-icl",        7),
]

TOTALS = {"long-dialogue": 39, "long-structured": 33, "long-icl": 81}


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
    cmd = (f"bash -lc \"{env} cd '{ROOT}' && "
           f"{PY} '{SCRIPT}'\" > {log} 2>&1; "
           f"echo EXIT=$? >> {log}; sleep 99999")
    subprocess.Popen(["bash", "-c", cmd], cwd=ROOT)
    print(f"[margin] 启动 {domain} -> GPU{gpu}  日志 {os.path.basename(log)}", flush=True)


def done_ids(domain):
    p = os.path.join(ROOT, "LongBench-v2_chunked", "Margin", "Qwen2.5-1.5B-Instruct",
                     "checkpoints", f"{domain}.shard0.jsonl")
    if not os.path.isfile(p):
        return 0
    ids = set()
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["_id"])
                except Exception:
                    pass
    return len(ids)


def main():
    print("[margin] 只跑短域（long-dialogue / long-structured / long-icl）", flush=True)
    for domain, gpu in PLAN:
        launch(domain, gpu)
        time.sleep(2)

    print("[margin] 全部启动，看护中（每 120s 报一次）", flush=True)
    while True:
        st = {d: f"{done_ids(d)}/{TOTALS[d]}" for d, _ in PLAN}
        left = [d for d in TOTALS if done_ids(d) < TOTALS[d]]
        stamp = time.strftime("%H:%M:%S")
        tail = "" if left else "  全部完成 ✅"
        print(f"[margin] {stamp} 进度 {st}{tail}", flush=True)
        if not left:
            print("[margin] MARGIN_SHORT_DONE —— 请手动 "
                  "python setup/merge_cm_shards.py --method margin", flush=True)
            return 0
        time.sleep(120)


if __name__ == "__main__":
    raise SystemExit(main())
