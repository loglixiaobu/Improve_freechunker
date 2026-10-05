#!/usr/bin/env python3
"""PPL 编排器：看护到底，不再手工插手。

背景（2026-10-04 的教训）
-------------------------
上一轮我手工干掉了 6 个 worker、手工把 multi-doc 劈 3 路、手工起 6 个进程，
还把专职做这件事的 `rebal` 会话杀了 —— 路子太野，且**没有任何东西在看护**。
这个脚本把「看护 + 自动再平衡 + 收尾合并」收进一个常驻进程。

职责
----
1. **守住 6 张卡**：CUDA_VISIBLE_DEVICES=2..7，一卡一 worker，绝不叠卡。
2. **按域排班**：每个域一个 worker（`nshards=1`）；长杆域可劈 N 路。
3. **短域跑完 → 自动把卡让给长杆**：这是 `rebalance_ppl.py` 的原理，
   但改成「长杆域预先就多分片」而不是「跑一半杀掉重启」，
   避免分片口径变化导致的重复计算。
4. **全部完成后自动合并**（调用 `merge_cm_shards.py --method ppl`）。
5. **磁盘水位检查**：`/home1` 低于阈值就报警并停下。

用法
----
    python setup/ppl_supervisor.py --dry-run     # 只打印排班，不真跑
    python setup/ppl_supervisor.py               # 常驻看护
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "LongBench-v2_chunked", "PPL",
                    "Qwen2.5-1.5B-Instruct", "checkpoints")
LOGS = os.path.join(ROOT, "logs")
PY = "/home1/lh/miniconda/envs/freechunker/bin/python"

TOTALS = {
    "single-doc": 175, "multi-doc": 125, "long-icl": 81,
    "code-repo": 50, "long-dialogue": 39, "long-structured": 33,
}

# 实测 mean/sample（秒），用于排班时估工期
SEC_PER_SAMPLE = {
    "single-doc": 33.7, "multi-doc": 72.4, "long-icl": 55.6,
    "code-repo": 169.2, "long-dialogue": 23.0, "long-structured": 26.0,
}


def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[sup] {now()} {msg}", flush=True)


def done_ids(dn: str, suffix: str = "") -> set[str]:
    p = os.path.join(CKPT, dn + suffix + ".jsonl")
    if not os.path.isfile(p):
        return set()
    ids: set[str] = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["_id"])
            except Exception:
                pass
    return ids


def domain_done(dn: str) -> int:
    """一个域在所有分片里去重后的完成数。"""
    ids: set[str] = set()
    for suf in [""] + [f".shard{i}" for i in range(16)]:
        ids |= done_ids(dn, suf)
    return len(ids)


def disk_ok(min_free_gb: float = 20.0) -> bool:
    st = os.statvfs(ROOT)
    free_gb = st.f_bavail * st.f_frsize / 1024 ** 3
    if free_gb < min_free_gb:
        log(f"⚠️  磁盘告急：/home1 仅剩 {free_gb:.1f} GiB (< {min_free_gb})")
        return False
    return True


def running_workers() -> list[tuple[int, str, str, int]]:
    """[(pid, gpu, domain, nshards)]"""
    out = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True).stdout
    res = []
    for line in out.splitlines():
        if "6_chunk_ppl.py" not in line or "bash -lc" not in line:
            continue
        pid = int(line.strip().split()[0])
        try:
            env = open(f"/proc/{pid}/environ", "rb").read().decode(errors="ignore")
        except Exception:
            continue
        gpu = dom = "?"
        ns = 1
        for tok in env.split("\x00"):
            if tok.startswith("CUDA_VISIBLE_DEVICES="):
                gpu = tok.split("=", 1)[1]
            elif tok.startswith("FC_CM_DOMAINS="):
                dom = tok.split("=", 1)[1]
            elif tok.startswith("FC_CM_NSHARDS="):
                ns = int(tok.split("=", 1)[1] or 1)
        res.append((pid, gpu, dom, ns))
    return res


def plan(n_gpus: int = 6) -> list[tuple[str, int, int]]:
    """按「最长工期优先分到更多卡」排班 -> [(domain, shard, nshards)]。

    做法：先给每个未完成的域 1 路，算出各自工期，把剩余卡追加给
    当前工期最长的域（贪心），劈成更多分片。
    """
    remain = {d: TOTALS[d] - domain_done(d) for d in TOTALS}
    todo = {d: n for d, n in remain.items() if n > 0}
    if not todo:
        return []

    # 每域的「单路工期」
    eta = {d: n * SEC_PER_SAMPLE[d] for d, n in todo.items()}
    shards = {d: 1 for d in todo}

    # n_gpus 张卡，域数可能少于卡数 → 把多出来的卡给工期最长的
    used = len(todo)
    spare = n_gpus - used
    while spare > 0:
        worst = max(shards, key=lambda d: eta[d] / shards[d])
        shards[worst] += 1
        spare -= 1

    out = []
    for d in sorted(todo, key=lambda x: -eta[x]):
        for s in range(shards[d]):
            out.append((d, s, shards[d]))
    return out


def launch(gpu: str, dn: str, shard: int, nshards: int) -> None:
    session = f"ppl_{dn}_{shard}"
    subprocess.run(["tmux", "kill-session", "-t", session],
                   capture_output=True)
    script = os.path.join(ROOT, "train and preprocess", "6_chunk_ppl.py")
    inner = (
        "source ~/miniconda/etc/profile.d/conda.sh && conda activate freechunker && "
        f"export PYTHONPATH='{ROOT}' && "
        f"export CUDA_VISIBLE_DEVICES='{gpu}' && "
        "export TOKENIZERS_PARALLELISM=false && "
        "export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1 && "
        f"export FC_CM_DOMAINS='{dn}' && "
        f"export FC_CM_SHARD='{shard}' && "
        f"export FC_CM_NSHARDS='{nshards}' && "
        f"cd '{ROOT}' && {PY} '{script}'"
    )
    logf = os.path.join(LOGS, f"ppl_{dn}_sh{shard}_gpu{gpu}.log")
    full = f"{inner} > {logf} 2>&1; echo EXIT=$? >> {logf}; sleep 99999"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, full])
    log(f"启动 {dn} shard{shard}/{nshards} @ GPU{gpu}")


def seed(dn: str, nshards: int) -> None:
    """把已完成的 <dn>.jsonl 播种到 N 个分片（已存在的不动）。"""
    src = os.path.join(CKPT, dn + ".jsonl")
    data = open(src, "rb").read() if os.path.isfile(src) else b""
    for i in range(nshards):
        dst = os.path.join(CKPT, f"{dn}.shard{i}.jsonl")
        if os.path.isfile(dst):
            continue
        with open(dst, "wb") as f:
            f.write(data)
        log(f"播种 {dn}.shard{i} ← {len(done_ids(dn))} 条基线")


def merge() -> None:
    log("全部完成 → 合并分片")
    r = subprocess.run([PY, os.path.join(ROOT, "setup", "merge_cm_shards.py"),
                        "--method", "ppl"],
                       capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    log("合并结束 MERGE_CM_DONE" if "MERGE_CM_DONE" in r.stdout else "合并未见完成标记")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="2,3,4,5,6,7")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--watch-only", action="store_true",
                    help="不重排、不启动任何 worker，只观察进度并在全部完成后合并。"
                         "用于「已有 worker 在跑，别打扰它们」的场景。")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    if args.watch_only:
        log("watch-only：不重排、不启动，只观察 + 收尾合并")
        if not disk_ok(args.min_free_gb):
            return 2
        while True:
            st = {d: (domain_done(d), TOTALS[d]) for d in TOTALS}
            left = {d: f"{c}/{t}" for d, (c, t) in st.items() if c < t}
            wk = running_workers()
            log(f"worker={len(wk)}  进度 {left if left else '全部完成 ✅'}")
            if not left:
                merge()
                log("SUPERVISOR_DONE")
                return 0
            time.sleep(args.poll)

    p = plan(len(gpus))
    log(f"排班（{len(gpus)} 卡，{len(p)} 路）：")
    for d, s, ns in p:
        log(f"    {d:<16} shard{s}/{ns}  剩余 {TOTALS[d] - domain_done(d)}")
    if args.dry_run:
        log("DRY_RUN_DONE")
        return 0

    if not disk_ok(args.min_free_gb):
        return 2

    # 播种所有多分片域
    for d, s, ns in p:
        if ns > 1:
            seed(d, ns)

    # 启动
    for (d, s, ns), gpu in zip(p, gpus):
        launch(gpu, d, s, ns)

    # 看护
    while True:
        time.sleep(args.poll)
        if not disk_ok(args.min_free_gb):
            log("磁盘不够，停止看护（worker 仍在跑）")
            return 3
        st = {d: (domain_done(d), TOTALS[d]) for d in TOTALS}
        left = {d: f"{c}/{t}" for d, (c, t) in st.items() if c < t}
        log(f"进度 {left if left else '全部完成 ✅'}")
        if not left:
            merge()
            log("SUPERVISOR_DONE")
            return 0


if __name__ == "__main__":
    sys.exit(main())
