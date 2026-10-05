#!/usr/bin/env python3
"""PPL 动态再平衡：等短域跑完，把空出来的卡拿去劈长杆域。

背景
----
PPL 6 卡按域固定分工后，各卡完工时间差极大（实测 mean/sample）：

    long-structured  ~26s  ->  ~5 min 完     (GPU7)
    long-dialogue    ~23s  ->  ~7 min 完     (GPU5)
    code-repo       ~169s  ->  ~48 min（已有 33 个老 checkpoint）(GPU4)
    long-icl         ~56s  ->  ~70 min       (GPU6)
    single-doc       ~34s  ->  ~91 min       (GPU2)
    multi-doc        ~72s  ->  ~142 min      (GPU3)  ← 长杆

GPU7 / GPU5 五到七分钟就跑完，白晾两小时。做法：
等短域达标 -> 杀掉长杆域的原 worker -> 改成 N 路分片重跑（按 `_id` 续跑，零重算）。

为什么必须「杀掉原 worker 再重启」
----------------------------------
分片口径是 `index % nshards == shard`。
原 `nshards=1` 的 worker 是「全都要」，新 worker 是「只要我那份」——
两者不一致的话，原 worker 会把新 shard 的样本又算一遍（结果不错但白烧算力）。
所以必须让**所有** worker 统一到同一个 `nshards`。

checkpoint 播种
---------------
`<target>.jsonl`（无后缀，原 worker 写的）里已有 K 条。
复制成 `<target>.shard{0..N-1}.jsonl`，每个分片启动时按 `_id` 跳过已完成的，
只算自己那份新样本 —— 一个样本都不重算。

用法
----
    python setup/rebalance_ppl.py --target multi-doc --free-gpus 5,7 --dry-run
    python setup/rebalance_ppl.py --target multi-doc --free-gpus 5,7
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "LongBench-v2_chunked", "PPL",
                    "Qwen2.5-1.5B-Instruct", "checkpoints")
LOGS = os.path.join(ROOT, "logs")

TOTALS = {
    "single-doc": 175, "multi-doc": 125, "long-icl": 81,
    "code-repo": 50, "long-dialogue": 39, "long-structured": 33,
}


def now() -> str:
    return time.strftime("%H:%M:%S")


def count(dn: str, suffix: str = "") -> int:
    """数某个域 checkpoint 里去重后的样本数。"""
    p = os.path.join(CKPT, dn + suffix + ".jsonl")
    if not os.path.isfile(p):
        return 0
    ids = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["_id"])
            except Exception:
                pass
    return len(ids)


def _proc_env(pid: int) -> str | None:
    try:
        return open(f"/proc/{pid}/environ", "rb").read().decode(errors="ignore")
    except Exception:
        return None


def scan_workers() -> list[tuple[int, str, str, str]]:
    """扫描所有 6_chunk_ppl.py 进程 -> [(pid, gpu, domains, nshards)]。"""
    out = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True).stdout
    res = []
    for line in out.splitlines():
        if "6_chunk_ppl.py" not in line or "[6]" in line:
            continue
        pid = int(line.strip().split()[0])
        env = _proc_env(pid)
        if env is None:
            continue
        gpu = dom = ns = "?"
        for tok in env.split("\x00"):
            if tok.startswith("CUDA_VISIBLE_DEVICES="):
                gpu = tok.split("=", 1)[1]
            elif tok.startswith("FC_CM_DOMAINS="):
                dom = tok.split("=", 1)[1]
            elif tok.startswith("FC_CM_NSHARDS="):
                ns = tok.split("=", 1)[1]
        res.append((pid, gpu, dom, ns))
    return res


def kill_target_single(target: str) -> list[int]:
    """杀掉 target 域、nshards=1 的 worker。"""
    killed = []
    for pid, gpu, dom, ns in scan_workers():
        if dom == target and ns == "1":
            try:
                os.kill(pid, 9)
                killed.append(pid)
                print(f"[rebal] killed pid={pid} gpu={gpu} dom={dom}")
            except Exception as e:
                print(f"[rebal] kill {pid} 失败: {e}")
    return killed


def seed_shards(target: str, nshards: int) -> None:
    """把 <target>.jsonl 复制到 N 个分片文件（已存在的分片不覆盖）。"""
    src = os.path.join(CKPT, target + ".jsonl")
    if not os.path.isfile(src):
        print(f"[rebal] 警告：{src} 不存在，分片将从零开始")
        data = b""
    else:
        data = open(src, "rb").read()
    for i in range(nshards):
        dst = os.path.join(CKPT, f"{target}.shard{i}.jsonl")
        if os.path.isfile(dst):
            print(f"[rebal] shard{i} 已存在，保留（{count(target, f'.shard{i}')} 条）")
            continue
        with open(dst, "wb") as f:
            f.write(data)
        print(f"[rebal] 播种 shard{i} -> {count(target, f'.shard{i}')} 条")


def launch_shard(gpu: str, dn: str, shard: int, nshards: int) -> subprocess.Popen:
    script = os.path.join(ROOT, "train and preprocess", "6_chunk_ppl.py")
    # ⚠️ 坑：`export A=1 && B=2 C=3 && cmd` 里 export 只管 A；
    # B/C 是普通 shell 变量，且 `B=2 C=3 cmd` 这种前缀赋值作用在 `cd` 上，
    # 不会传进 python —— 结果三个分片全跑成 shard0/nshards=1。
    # 必须每个都写 export。
    env = (
        "source ~/miniconda/etc/profile.d/conda.sh && conda activate freechunker && "
        f"export PYTHONPATH='{ROOT}' && "
        f"export CUDA_VISIBLE_DEVICES='{gpu}' && "
        "export TOKENIZERS_PARALLELISM=false && "
        "export HF_HUB_OFFLINE=1 && "
        "export TRANSFORMERS_OFFLINE=1 && "
        f"export FC_CM_DOMAINS='{dn}' && "
        f"export FC_CM_SHARD='{shard}' && "
        f"export FC_CM_NSHARDS='{nshards}' && "
    )
    log = os.path.join(LOGS, f"ppl_rebal_{dn}_gpu{gpu}_sh{shard}.log")
    cmd = f"bash -lc \"{env} cd '{ROOT}' && python '{script}'\" 2>&1 | tee {log}"
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="multi-doc", help="要劈开的长杆域")
    ap.add_argument("--free-gpus", default="5,7", help="短域跑完后空出的 GPU")
    ap.add_argument("--donor-domains", default="long-dialogue,long-structured",
                    help="这些域达标即视为卡已空出")
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    free = [g.strip() for g in args.free_gpus.split(",") if g.strip()]
    donors = [d.strip() for d in args.donor_domains.split(",") if d.strip()]
    target = args.target

    print(f"[rebal] {now()} target={target}  free_gpus={free}  等 {donors} 达标")

    # 1) 等 donors 全部达标
    while True:
        st = {d: (count(d), TOTALS[d]) for d in donors}
        done = [d for d, (c, t) in st.items() if c >= t]
        print(f"[rebal] {now()} {st} done={done}", flush=True)
        if len(done) >= len(donors):
            break
        time.sleep(args.poll)

    if args.dry_run:
        print("[dry-run] 到此为止")
        print("REBAL_DRY_DONE")
        return

    # 2) 找出 target 原 worker 所在的那张卡（它继续当 shard0，不浪费）
    old_gpus = []
    for pid, gpu, dom, ns in scan_workers():
        if dom == target and ns == "1":
            old_gpus.append(gpu)
    kill_target_single(target)
    time.sleep(5)

    # 3) nshards = 原卡(shard0) + 新空出的卡(shard1..N)
    nshards = len(free) + len(old_gpus)   # 通常 = 1 + 2 = 3
    seed_shards(target, nshards)

    # 4) shard0 回到原卡，其余铺到 free
    plan = []
    if old_gpus:
        plan.append((old_gpus[0], 0))
    for i, gpu in enumerate(free, start=len(plan)):
        plan.append((gpu, i))

    procs = []
    for gpu, shard in plan:
        p = launch_shard(gpu, target, shard, nshards)
        print(f"[rebal] {now()} 启动 {target} shard{shard}/{nshards} "
              f"on GPU{gpu} pid={p.pid}")
        procs.append((gpu, p))

    while procs:
        for item in list(procs):
            gpu, p = item
            if p.poll() is not None:
                print(f"[rebal] {now()} GPU{gpu} 结束 rc={p.returncode}")
                procs.remove(item)
        time.sleep(15)
    print("REBAL_DONE")


if __name__ == "__main__":
    main()
