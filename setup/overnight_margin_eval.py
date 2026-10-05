#!/usr/bin/env python3
"""无人值守流水线：等 Margin 满 503 → 合并 → 校验 → 起 vLLM → 并行评测 → 汇总。

阶段（每步都写日志，失败会明确标记，不静默）
--------------------------------------------
  [0] 等 Margin 6 域全部齐（轮询，超时 MAX_WAIT_H 小时则中止并报错）
  [1] 合并 Margin 分片        -> setup/merge_cm_shards.py --method margin
  [2] 校验合并结果            -> setup/check_all_results.py（必须 CHECK_ALL_OK）
  [3] 归档历史 raw 文件        -> eval_results/_pre_parallel/
      （10:49–12:16 的旧冒烟产物，虽不匹配汇总正则，但归档后更干净）
  [4] 等 GPU 0..7 空出
  [5] 起 2 个 vLLM（GPU0:8888 / GPU1:8889），等 /health
  [6] 并行评测                -> setup/run_eval_parallel.py --nshards 6
  [7] 汇总                    -> setup/aggregate_eval.py
  [8] 写 DONE 标记 + 结果摘要

用法（在 tmux 里跑）：
    tmux new-session -d -s night \
      "cd ~/FreeChunker && python setup/overnight_margin_eval.py > logs/overnight.log 2>&1"
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "setup"))

LOGS = os.path.join(ROOT, "logs")
PY = "/home1/lh/miniconda/envs/freechunker/bin/python"
MARGIN_ROOT = os.path.join(ROOT, "LongBench-v2_chunked", "Margin",
                           "Qwen2.5-1.5B-Instruct")
EVAL_DIR = os.path.join(ROOT, "eval_results")

MAX_WAIT_H = 5.0          # 等 Margin 的上限
POLL_S = 120
VLLM = [(0, 8888, "vllm"), (1, 8889, "vllm1")]
# 评测 worker 用的卡（与 run_eval_parallel.py 的 ENC_GPUS 一致）。
# 只等这几张空出来；0/1 是留给 vLLM 的，不算在内。
EVAL_GPUS = [2, 3, 4, 5, 6, 7]


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, **kw)


def margin_counts():
    """返回 (总唯一 id 数, {域: 数})，目录 + checkpoint 全算上。"""
    import check_all_results as car
    detail = {}
    tot = 0
    for dn in car.DOMAINS:
        ids = set()
        d = os.path.join(MARGIN_ROOT, dn)
        if os.path.isdir(d):
            try:
                from datasets import load_from_disk
                ds = load_from_disk(d)
                ids |= {str(x) for x in ds["_id"]}
            except Exception as e:
                log(f"  !! 读目录失败 {dn}: {e}")
        ck = os.path.join(MARGIN_ROOT, "checkpoints")
        if os.path.isdir(ck):
            for fn in os.listdir(ck):
                if not fn.endswith(".jsonl"):
                    continue
                stem = fn[:-len(".jsonl")]
                if stem != dn and not stem.startswith(dn + ".shard") \
                        and not stem.startswith(dn + ".gap"):
                    continue
                with open(os.path.join(ck, fn), encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                ids.add(str(json.loads(line)["_id"]))
                            except Exception:
                                pass
        detail[dn] = len(ids)
        tot += len(ids)
    return tot, detail


def expected_total():
    import check_all_results as car
    return sum(len(car.expected_ids(d)) for d in car.DOMAINS)


def gpu_used():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True).stdout
    m = {}
    for line in out.splitlines():
        if "," in line:
            i, v = line.split(",", 1)
            m[int(i.strip())] = int(v.strip().split()[0])
    return m


def wait_vllm(ports, timeout=900):
    for p in ports:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                with urllib.request.urlopen(
                        f"http://localhost:{p}/health", timeout=5) as r:
                    if r.status == 200:
                        log(f"  vLLM:{p} 就绪")
                        break
            except Exception:
                time.sleep(10)
        else:
            raise RuntimeError(f"vLLM {p} 在 {timeout}s 内未就绪")


def main():
    os.makedirs(LOGS, exist_ok=True)
    exp = expected_total()
    log(f"=== 无人值守流水线启动，Margin 目标 {exp} 条 ===")

    # ---------- [0] 等 Margin ----------
    t0 = time.time()
    while True:
        tot, detail = margin_counts()
        log(f"[0] Margin {tot}/{exp}  " +
            " ".join(f"{k}={v}" for k, v in detail.items()))
        if tot >= exp:
            log("[0] Margin 已满")
            break
        if (time.time() - t0) / 3600 > MAX_WAIT_H:
            log(f"[FATAL] 等 Margin 超过 {MAX_WAIT_H}h 仍未满，中止")
            return 2
        time.sleep(POLL_S)

    # ---------- [1] 合并 ----------
    log("[1] 合并 Margin 分片 ...")
    r = sh(f"cd {ROOT} && {PY} setup/merge_cm_shards.py --method margin "
           f">> {LOGS}/overnight_merge.log 2>&1")
    log(f"[1] merge rc={r.returncode}  (日志 logs/overnight_merge.log)")
    if r.returncode != 0:
        log("[FATAL] 合并失败")
        return 3

    # ---------- [2] 硬门：合并后每域条数必须精确匹配 ----------
    # 这是唯一会让流水线中止的检查 —— 因为它是**无歧义**的：
    # 合并要么产出 175/125/50/39/81/33，要么没有。
    # （更宽的全量核查只记录、不中止，避免我自己的脚本误报害你白等一晚。）
    log("[2] 硬门：校验合并后每域条数 ...")
    exp_detail = {}
    import check_all_results as car
    for dn in car.DOMAINS:
        exp_detail[dn] = len(car.expected_ids(dn))
    tot, detail = margin_counts()
    log(f"[2] 实际 {detail}")
    log(f"[2] 期望 {exp_detail}")
    bad = {d: (detail.get(d, 0), exp_detail[d]) for d in exp_detail
           if detail.get(d, 0) != exp_detail[d]}
    if bad:
        log(f"[FATAL] 合并后条数不符: {bad}，中止以免产出错结果")
        return 4
    log(f"[2] ✅ 每域条数精确匹配，合计 {tot}/{exp}")

    # ---------- [2b] 全量核查（只记录，不中止）----------
    log("[2b] 全量核查（记录用）...")
    sh(f"cd {ROOT} && {PY} setup/check_all_results.py "
       f"> {LOGS}/overnight_check.log 2>&1")
    chk = open(f"{LOGS}/overnight_check.log", encoding="utf-8").read()
    check_ok = "CHECK_ALL_OK" in chk
    log(f"[2b] CHECK_ALL_OK = {check_ok}（详见 logs/overnight_check.log）")
    log("[2b] 尾部:\n" + chk[-1200:])

    # ---------- [3] 归档历史 raw ----------
    arch = os.path.join(EVAL_DIR, "_pre_parallel")
    os.makedirs(arch, exist_ok=True)
    moved = 0
    for fn in sorted(os.listdir(EVAL_DIR)):
        if fn.startswith("raw_") and fn.endswith(".jsonl"):
            shutil.move(os.path.join(EVAL_DIR, fn), os.path.join(arch, fn))
            moved += 1
    log(f"[3] 归档 {moved} 个历史 raw 文件 -> eval_results/_pre_parallel/")

    # ---------- [4] 等 GPU 空出 ----------
    log(f"[4] 等评测卡 GPU{EVAL_GPUS} 显存回落 ...")
    t0 = time.time()
    while time.time() - t0 < 1800:
        m = gpu_used()
        busy = {i: v for i, v in m.items() if i in EVAL_GPUS and v > 2000}
        if not busy:
            log("[4] 评测卡已全部空出")
            break
        log(f"[4] 仍占用 {busy}")
        time.sleep(60)
    else:
        log("[WARN] 评测卡未完全空出，仍继续（可能只是残留缓存）")

    # ---------- [5] 起 vLLM ----------
    log("[5] 启动 vLLM 实例 ...")
    for gpu, port, sess in VLLM:
        sh(f"tmux kill-session -t {sess} 2>/dev/null")
        inner = (f"source ~/miniconda/etc/profile.d/conda.sh && "
                 f"conda activate freechunker && cd {ROOT} && "
                 f"FC_VLLM_GPU={gpu} FC_VLLM_PORT={port} "
                 f"FC_VLLM_MAXLEN=32768 "          # Qwen3-8B 原生上下文
                 f"bash setup/start_vllm.sh "
                 f"> {LOGS}/vllm_{port}.log 2>&1")
        sh(f'tmux new-session -d -s {sess} "{inner}"')
        log(f"  起 {sess}: GPU{gpu} -> :{port}")
        time.sleep(3)
    wait_vllm([p for _, p, _ in VLLM])
    log("[5] vLLM 全部就绪")

    # ---------- [6] 并行评测 ----------
    log("[6] 并行评测 6 方法 × 6 域（nshards=6）...")
    # ⚠️ 必须**轮转**旧日志：gate 是"日志里有没有 PARALLEL_DONE"，
    # 若用追加（>>），上一轮残留的 PARALLEL_DONE 会让 gate 永远通过。
    ev_log = f"{LOGS}/overnight_eval.log"
    if os.path.isfile(ev_log):
        shutil.move(ev_log, ev_log + ".prev")
    r = sh(f"cd {ROOT} && {PY} setup/run_eval_parallel.py --nshards 6 "
           f"> {ev_log} 2>&1")
    log(f"[6] eval rc={r.returncode}  (日志 {ev_log})")

    # ⚠️ 硬门必须卡在**最终产物**上。
    # `test/eval_downstream.py` 会吞异常、照样 EVAL_DONE 返回 0，
    # 所以 rc=0 毫无意义 —— 2026-10-05 就是因此 36 个任务里 23 个静默失败，
    # 流水线还写了 ALL_DONE。现在由 run_eval_parallel 数产物并打印
    # PARALLEL_DONE / PARALLEL_INCOMPLETE。
    eval_log = open(f"{LOGS}/overnight_eval.log", encoding="utf-8").read()
    if "PARALLEL_DONE" not in eval_log:
        log("[FATAL] 评测覆盖不全（未出现 PARALLEL_DONE），**不出汇总表**")
        log("[FATAL] 见 logs/overnight_eval.log 里的 [verify] 段落")
        return 6
    log("[6] ✅ 覆盖度硬门通过（PARALLEL_DONE）")

    # ---------- [7] 汇总 ----------
    log("[7] 汇总 ...")
    r = sh(f"cd {ROOT} && {PY} setup/aggregate_eval.py "
           f"> {LOGS}/overnight_agg.log 2>&1")
    log(f"[7] aggregate rc={r.returncode}")
    try:
        print(open(f"{LOGS}/overnight_agg.log", encoding="utf-8").read()[-3000:])
    except Exception:
        pass

    # ---------- [8] 收尾：写一份早上一眼看懂的汇总 ----------
    n_raw = len([f for f in os.listdir(EVAL_DIR)
                 if f.startswith("raw_") and f.endswith(".jsonl")])
    log(f"[8] eval_results 下新 raw 文件 {n_raw} 个")

    agg = ""
    try:
        agg = open(f"{LOGS}/overnight_agg.log", encoding="utf-8").read()
    except Exception:
        pass
    rep = os.path.join(ROOT, "results", "downstream_report.md")
    rep_txt = open(rep, encoding="utf-8").read() if os.path.isfile(rep) else "(未生成)"

    summ = os.path.join(ROOT, "results", "OVERNIGHT_SUMMARY.md")
    with open(summ, "w", encoding="utf-8") as f:
        f.write(f"# 夜间流水线结果 · {datetime.now():%F %T}\n\n")
        f.write("## 阶段状态\n\n")
        f.write(f"- [0] 等 Margin 满 {exp} 条：✅\n")
        f.write(f"- [1] 合并 Margin 分片：✅\n")
        f.write(f"- [2] 硬门（每域条数精确匹配）：✅ "
                f"{detail}\n")
        f.write(f"- [2b] 全量核查 CHECK_ALL_OK："
                f"{'✅' if check_ok else '❌ 见 logs/overnight_check.log'}\n")
        f.write(f"- [3] 归档历史 raw：{moved} 个 -> eval_results/_pre_parallel/\n")
        f.write(f"- [4] 等评测卡空出：✅\n")
        f.write(f"- [5] 起 2 个 vLLM（GPU0:8888 / GPU1:8889）：✅\n")
        f.write(f"- [6] 并行评测 6 方法 × 6 域：rc={r.returncode}，"
                f"新 raw {n_raw} 个；**覆盖度硬门 PARALLEL_DONE ✅**\n")
        f.write(f"- [7] 汇总：✅ -> results/downstream_report.md\n\n")
        f.write("## 汇总表\n\n")
        f.write(rep_txt)
        f.write("\n\n## aggregate 原始输出（尾部）\n\n```\n")
        f.write(agg[-2000:])
        f.write("\n```\n")
    log(f"[8] 汇总已写: {summ}")
    log("=== ALL_DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
