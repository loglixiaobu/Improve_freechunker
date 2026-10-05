#!/usr/bin/env python3
"""下游评测并行编排：把 6 方法 × 6 域 的任务切成 N 份并发跑。

为什么需要它：单进程评测是**串行**的 —— 每题「编码 chunk → 等 LLM → 判分」，
而 vLLM 一次只收到 1 个请求（实测 Running: 0~1 reqs，GPU 利用率才 50% 上下）。
两个 GPU 都没跑满，纯粹是被串行循环拖住。

所以并行度加在**进程**这一层：
  - 每个 worker 负责一部分「域」（worker k 领 domains[k::nshards]）
  - worker 再按 vLLM 端口轮转，把请求分摊到 2 个实例（GPU0 / GPU1）
  - 每个 worker 用独立 GPU 跑编码器（嵌入模型只要 ~1.5G，一张卡放好几个）

分片按「域」而不是按「题」，因为每个 worker 都要各自 build 一次 vector store，
按域切零浪费。

⚠️ **分片只做一次**（2026-10-04 修）：`test/eval_downstream.py::_apply_keep()`
内部已经实现了 `domains[shard::nshards]`。本脚本**不要再预先切好再传**，
否则域被切两次 —— shard1 起全部拿到空集、静默跳过、整晚只评出 1/6 的域。
所以这里传**完整** domains 列表 + shard/nshards，让 `_apply_keep` 做唯一那次切分。

用法：
  python setup/run_eval_parallel.py                      # 全量 6 方法 × 6 域
  python setup/run_eval_parallel.py --methods freechunker,traditional --domains single-doc,multi-doc
  python setup/run_eval_parallel.py --nshards 6 --dry-run
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.expanduser("~/FreeChunker")
CONDA = os.path.expanduser("~/miniconda/etc/profile.d/conda.sh")
PY_ENV = "freechunker"

ALL_DOMAINS = ["single-doc", "multi-doc", "code-repo", "long-dialogue", "long-icl", "long-structured"]
ALL_METHODS = ["freechunker", "traditional", "ppl", "margin", "semantic", "lumber"]

# 2 个 vLLM 实例：GPU0:8888 / GPU1:8889
VLLM_PORTS = [8888, 8889]
# 编码器/评测进程可以用的卡（vLLM 占 0/1）。
# ⚠️ 实测每个 eval worker 会吃 **10–20 GiB** 显存（不是我以为的 1.5 GiB），
# 所以 **一张卡只能放一个 worker**，绝不能按编号硬分配。
ENC_GPUS = [2, 3, 4, 5, 6, 7]
# 判定"这张卡是空的"的显存阈值（MiB）。vLLM 常驻 ~22 GiB 会被排除。
FREE_GPU_MIB = 2000


def build_tasks(methods, domains, nshards, chunk_sizes, repeats, topk, limit):
    """生成任务清单。

    ⚠️ 这里**不分配 GPU** —— gpu 在真正启动的那一刻由 `pick_free_gpu()` 动态决定。
    早期版本用 `ENC_GPUS[wid % 6]` 硬映射，出过大事故（2026-10-05）：
    worker 早退后，调度器把下一个 worker 派到"编号对应"的卡上，而那张卡上
    原来的 worker 还在跑 → 两进程挤一张 24 GiB 卡 → 级联 CUDA OOM，
    36 个任务里 23 个静默失败。
    """
    tasks = []
    wid = 0
    for m in methods:
        for shard in range(nshards):
            picked = domains[shard::nshards]
            if not picked:
                continue
            port = VLLM_PORTS[wid % len(VLLM_PORTS)]
            tasks.append({
                "wid": wid,
                "method": m,
                "shard": shard,
                "nshards": nshards,
                # ⚠️ 这里必须传**完整的** domains 列表，不能传 picked！
                # `test/eval_downstream.py::_apply_keep()` 内部**也会**做
                # `domains[shard::nshards]` 的分片。如果这里先切好再传进去，
                # 域就被切了两次 —— shard0 拿到 ['single-doc'] 还能用，
                # 但 shard1 拿到 ['multi-doc'] 再 `[1::6]` 就变成 **空集**，
                # worker 直接打印"没领到域，跳过"、静默不产出。
                # 实测（2026-10-04）：6 个 shard 里只有 shard0 有活，其余全空。
                # 正确做法：把分片这件事**只交给 _apply_keep 做一次**。
                "domains": ",".join(domains),
                "picked": ",".join(picked),   # 仅供日志显示
                "gpu": None,                  # 启动时再定
                "port": port,
                "chunk_sizes": chunk_sizes,
                "repeats": repeats,
                "topk": topk,
                "limit": limit,
            })
            wid += 1
    return tasks


def cmd_for(t):
    return (
        f"source {CONDA} && conda activate {PY_ENV} "
        f"&& export PYTHONPATH={ROOT} && export HF_ENDPOINT=https://hf-mirror.com "
        # ⚠️ 必须 OFFLINE=1：jina 的 trust_remote_code 会去 HF 发 HEAD 请求，
        # 在不通外网的服务器上会挂死（实测 130 线程卡在 poll，CPU 不再增长）。
        f"&& export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 "
        f"TOKENIZERS_PARALLELISM=false "
        # 评测侧的检索上下文上限必须 <= vLLM 的 --max-model-len，否则超长 prompt
        # 会被 vLLM 以 HTTP 400 拒绝、整个域归零（2026-10-05 实测）。
        # vLLM 实测上限 36160；Qwen3-8B 原生 32768 → 两边都取 32768。
        f"&& export FC_EVAL_MAX_CONTEXT=32768 "
        # FreeChunker 的句子窗口大小（见 src/encoder.py）。4000 句时峰值 ~4.7 GiB。
        f"&& export FC_MAX_SENTENCES=4000 "
        # 降碎片：embedding 阶段会反复申请/释放大块（长 chunk 的注意力矩阵）
        f"&& export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"&& export CUDA_VISIBLE_DEVICES={t['gpu']} "
        f"&& cd {ROOT} && python test/eval_downstream.py "
        f"--method {t['method']} --domains {t['domains']} --topk {t['topk']} "
        f"--repeats {t['repeats']} --shard {t['shard']} --nshards {t['nshards']} "
        f"--vllm-port {t['port']} --chunk-sizes {t['chunk_sizes']} "
        + (f"--limit {t['limit']} " if t['limit'] else "")
    )


def gpu_mem_used():
    """{gpu_index: 已用 MiB}。"""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True).stdout
    m = {}
    for line in out.splitlines():
        if "," in line:
            i, v = line.split(",", 1)
            try:
                m[int(i.strip())] = int(v.strip().split()[0])
            except ValueError:
                pass
    return m


def pick_free_gpu(in_use):
    """挑一张「既没被我们自己占用、显存也确实低」的卡。

    两个条件缺一不可：
      * `in_use` —— 我们自己的 worker 还没退出的卡
      * nvidia-smi 显存 < FREE_GPU_MIB —— 排除别人/vLLM 占着的卡
    """
    used = gpu_mem_used()
    for g in ENC_GPUS:
        if g in in_use:
            continue
        if used.get(g, 10 ** 9) < FREE_GPU_MIB:
            return g
    return None


def wait_vllm(ports, timeout=600):
    import urllib.request
    for p in ports:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                with urllib.request.urlopen(f"http://localhost:{p}/health", timeout=4) as r:
                    if r.status == 200:
                        print(f"[vllm:{p}] 就绪")
                        break
            except Exception:
                time.sleep(10)
        else:
            raise SystemExit(f"[FAIL] vLLM {p} 未就绪")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="all")
    ap.add_argument("--domains", default="all")
    ap.add_argument("--nshards", type=int, default=6)
    ap.add_argument("--chunk-sizes", default="256,512")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--topk", default="5,10")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-parallel", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    methods = ALL_METHODS if args.methods == "all" else [m.strip() for m in args.methods.split(",")]
    domains = ALL_DOMAINS if args.domains == "all" else [d.strip() for d in args.domains.split(",")]

    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    os.makedirs(f"{ROOT}/eval_results", exist_ok=True)

    # 清掉上一轮的 worker 日志：否则本轮没启动到的 worker 会留着旧日志，
    # 排查时会看到"明明没跑却报 OOM"的假象（2026-10-05 踩过）。
    for old in glob.glob(f"{ROOT}/logs/eval_w*.log"):
        try:
            os.remove(old)
        except OSError:
            pass

    tasks = build_tasks(methods, domains, args.nshards, args.chunk_sizes,
                        args.repeats, args.topk, args.limit)

    print(f"[parallel] {len(tasks)} 个任务, 最多 {args.max_parallel} 并发 "
          f"(GPU 启动时动态分配)")
    for t in tasks:
        print(f"  w{t['wid']:02d} {t['method']:12s} shard{t['shard']}/{t['nshards']} "
              f"domains={t['picked']:20s} port={t['port']}")
    if args.dry_run:
        print("\n--- dry-run，命令示例（gpu 由 pick_free_gpu 决定）---")
        t0 = dict(tasks[0])
        t0["gpu"] = ENC_GPUS[0]
        print(cmd_for(t0))
        return

    wait_vllm(VLLM_PORTS)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    procs = []
    logs = {}

    def launch(t):
        logp = f"{ROOT}/logs/eval_w{t['wid']:02d}_{t['method']}_s{t['shard']}.log"
        f = open(logp, "w", encoding="utf-8")
        f.write(f"===== {time.strftime('%F %T')} w{t['wid']} =====\n{cmd_for(t)}\n\n")
        f.flush()
        p = subprocess.Popen(["bash", "-lc", cmd_for(t)], stdout=f, stderr=subprocess.STDOUT)
        procs.append((t, p, f, logp))
        logs[t["wid"]] = logp
        print(f"[launch] w{t['wid']:02d} {t['method']} shard{t['shard']} gpu={t['gpu']} port={t['port']} pid={p.pid}")

    queue = list(tasks)
    running = []
    in_use = set()          # 我们自己的 worker 正在用的卡
    t0 = time.time()

    while queue or running:
        # 回收已退出 worker 占的卡
        alive = []
        for (tt, pp, ff, lp) in procs:
            if pp.poll() is None:
                alive.append((tt, pp, ff, lp))
            else:
                in_use.discard(tt["gpu"])
        running = alive

        # 有活 + 有空卡 + 没超并发 → 启动
        while queue and len(running) < args.max_parallel:
            gpu = pick_free_gpu(in_use)
            if gpu is None:
                break                      # 没有真空卡，等下一轮
            t = queue.pop(0)
            t["gpu"] = gpu
            in_use.add(gpu)
            launch(t)
            running = [(tt, pp, ff, lp) for (tt, pp, ff, lp) in procs
                       if pp.poll() is None]
            time.sleep(3)

        done = len(procs) - len([1 for (_, pp, _, _) in procs if pp.poll() is None])
        print(f"[status] 完成 {done}/{len(tasks)}  运行 {len(running)}  "
              f"占用卡 {sorted(in_use)}  已过 {(time.time()-t0)/60:.1f} min")
        if queue and not running and pick_free_gpu(in_use) is None:
            print("[warn] 有任务待跑但所有评测卡都被占用，等待中 …")
        time.sleep(20)

    # 收尾
    for (tt, pp, ff, lp) in procs:
        rc = pp.wait()
        ff.close()
        print(f"[done] w{tt['wid']:02d} {tt['method']} shard{tt['shard']} "
              f"gpu={tt['gpu']} rc={rc} log={lp}")

    print(f"\n[parallel] 全部进程结束，耗时 {(time.time()-t0)/60:.1f} min")

    # ---------- 覆盖度硬门 ----------
    # rc=0 **不代表**跑成功：eval_downstream.py 会吞异常、照样 EVAL_DONE。
    # 2026-10-05 就是因此 36 个任务里 23 个静默失败、外层还报 PARALLEL_DONE。
    # 所以这里必须**直接数产物**。
    ok = verify_coverage(methods, domains, t0, args.limit)
    if ok:
        print("PARALLEL_DONE")
    else:
        print("PARALLEL_INCOMPLETE")
        raise SystemExit(9)


def verify_coverage(methods, domains, t0, limit):
    """数本轮 raw 产物，逐 (方法, 域) 断言题数。返回是否全部达标。"""
    import glob
    import re as _re
    from collections import defaultdict
    from datasets import load_from_disk

    id2dom = {}
    for dn in ALL_DOMAINS:
        ds = load_from_disk(os.path.join(ROOT, "datasets", "LongBench-v2",
                                         "split_by_domain", dn))
        for x in ds["_id"]:
            id2dom[x] = dn
    exp = {}
    for dn in ALL_DOMAINS:
        ds = load_from_disk(os.path.join(ROOT, "datasets", "LongBench-v2",
                                         "split_by_domain", dn))
        exp[dn] = min(len(ds), limit) if limit else len(ds)

    cov = defaultdict(set)
    n_files = 0
    for p in sorted(glob.glob(f"{ROOT}/eval_results/raw_*.jsonl")):
        if os.path.getmtime(p) < t0 - 5:
            continue
        n_files += 1
        base = os.path.basename(p)
        m = _re.match(r"raw_(.+?)_shard\d+_rep\d+_\d{8}_\d{6}\.jsonl$", base)
        if not m:
            continue
        key = m.group(1)
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    import json as _j
                    r = _j.loads(line)
                    cov[(key, id2dom.get(r.get("question_id"), "?"))].add(
                        r.get("question_id"))
        except Exception as e:
            print(f"[verify] 读 {base} 失败: {e}")

    print(f"\n[verify] 本轮 raw 文件 {n_files} 个，逐 (方法,域) 题数：")
    bad = []
    for meth in methods:
        keys = [f"{meth}_{s}" for s in (256, 512)] if meth == "traditional" else [meth]
        for k in keys:
            for dn in domains:
                got = len(cov.get((k, dn), ()))
                want = exp[dn]
                flag = "" if got == want else "  ⚠️"
                if got != want:
                    bad.append(f"{k}/{dn}: {got}/{want}")
                print(f"    {k:20s} {dn:16s} {got:4d}/{want:<4d}{flag}")
    if bad:
        print(f"\n[verify] ❌ 覆盖不全（{len(bad)} 项）: {bad[:10]}")
        return False
    print("\n[verify] ✅ 覆盖完整")
    return True


if __name__ == "__main__":
    main()
