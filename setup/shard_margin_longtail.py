#!/usr/bin/env python3
"""把 Margin 的长尾域拆成多分片，铺到空闲 GPU 上，并在空卡出现时**续切**剩余缺口。

背景与教训
----------
之前 `run_margin_short.py` 用 `subprocess.Popen` 起 worker —— 那些 worker 是
**supervisor 的子进程**。我后来 `tmux kill-session -t mgshort` 清理会话时，
把 supervisor 及其整个进程组一起杀了，连带干掉了还在跑的
long-dialogue(38/39) 和 long-icl(7/81)。

→ 本脚本改为：**每个分片一个独立 tmux 会话**（`tmux new-session -d`），
   彼此没有父子关系。杀任何一个都不会牵连其他。
→ checkpoint + `_id` 级 resume 保证重跑无损（已验证）。

硬约束：Margin 把模型 load 进进程（~10-24 GB），**一个进程独占一张卡**。
所以「分配了几个分片」= 「占了几张卡」。

### v1 的盲区（2026-10-04 20:45 踩到）
原 `pending_shards()` 只认 `PLAN` 里声明过的 (domain, shard) 组合，
一旦 4 个分片全部启动过，`out=[]` —— 哪怕此时 4 张卡全空，
也无法把「在跑分片里剩下的那几十条」再摊开。结果：
  - 空卡 0/1/5/7 白等 ~1 小时
  - 而 single-doc 剩 22 条、long-icl 剩 14 条挤在两片里慢慢跑

### v2 曾尝试「缺口续切（gap reshard）」—— **已回退，不要重新启用**

想法：空卡出现时，把在跑分片剩下的几十条再摊到空卡上。

实测结论（2026-10-04 20:45）：**不值得**。
  - 当时 single-doc 剩 22 条 ≈ 33 min，long-icl 剩 14 条 ≈ 21 min，
    两片并行 → 约 35 min 全部结束
  - 续切最多省 ~20 min，却要改 chunker 脚本 + 重启调度器 + 处理
    "gap worker 看不到在跑 worker 的 checkpoint 导致重复算" 的副作用
  - `_cm_load_done()` **写了但主循环没读**（resume 实际靠 `_cm_keep_idx` 下标过滤），
    所以 gap worker 一定会重算已完成的样本 —— 只能靠 `_id` 去重兜底
  → 收益小、风险大、违背「别让错误无声发生」的教训。**回退。**

⚠️ 另有一个容易误判的点：**改磁盘上的 .py 对已在跑的进程无效**。
   当时 `mz` 进程是 20:25 启动的，我 20:40 改文件，旧进程仍按内存里的旧代码跑，
   续切从未生效。**改脚本后必须重启进程才算数。**（判断方法见 skill §4.1.1）

⚠️ 先停掉 GPU 上那个旧 single-doc worker（nshards=1，才 1/175）。
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

# 想跑的分片总数（不含已完成的域）
PLAN = {
    "single-doc": 4,
    "long-icl": 4,
}
TOTALS = {"single-doc": 175, "long-icl": 81}
# 允许使用的 GPU（4/6 被 multi-doc/code-repo 占着，不碰）
USABLE_GPUS = [0, 1, 2, 3, 5, 7]

# 续切开关（回退后恒为 False；保留代码以备将来真需要时再评估）
GAP_RESHARD = False
GAP_MIN_FREE_GPUS = 2      # 空卡少于这个数就不续切（不值当）
GAP_MIN_REMAIN = 8         # 剩余缺口少于这个数就不续切

# 已在本脚本中启动过的 (domain, shard) -> gpu，避免重复起
launched = {}
# 续切 worker：session -> (domain, [idx...])
gap_launched = {}


def ck_files(domain):
    if not os.path.isdir(CK):
        return []
    out = []
    for fn in os.listdir(CK):
        if not fn.endswith(".jsonl"):
            continue
        stem = fn[:-len(".jsonl")]
        if stem != domain and not stem.startswith(domain + ".shard"):
            continue
        out.append(os.path.join(CK, fn))
    return out


def done_ids(domain):
    ids = set()
    for p in ck_files(domain):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line)["_id"])
                    except Exception:
                        pass
    return ids


def domain_row_ids(domain):
    """域里每一行的 _id，按位置排列。"""
    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(ROOT, "datasets", "LongBench-v2",
                                     "split_by_domain", domain))
    return [str(ds[i]["_id"]) for i in range(len(ds))]


def gap_indexes(domain, done):
    """还没写出结果的样本下标。"""
    rows = domain_row_ids(domain)
    return [i for i, rid in enumerate(rows) if rid not in done]


def live_claimed_indexes(domain):
    """正在跑的 worker 认领的下标（避免续切重复劳动）。"""
    n = PLAN.get(domain, 0)
    if n <= 0:
        return set()
    tot = TOTALS[domain]
    out = set()
    for s in range(n):
        if (domain, s) in launched:
            out |= set(range(s, tot, n))
    return out


def gpu_busy():
    """返回当前有计算进程的 GPU index 集合。"""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        capture_output=True, text=True).stdout
    busy_uuids = {l.strip() for l in out.splitlines() if l.strip()}
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True).stdout
    busy = set()
    for line in q.splitlines():
        if ", " not in line:
            continue
        idx, uuid = [s.strip() for s in line.split(",", 1)]
        if uuid in busy_uuids:
            busy.add(int(idx))
    return busy


def stop_old_single_doc():
    """停掉 m_single-doc_g2（旧 nshards=1 worker），它在 GPU2 且只跑了 1/175。"""
    sess = "m_single-doc_g2"
    r = subprocess.run(["tmux", "has-session", "-t", sess],
                       stderr=subprocess.DEVNULL)
    if r.returncode == 0:
        subprocess.run(["tmux", "kill-session", "-t", sess], check=False)
        print(f"[mz] 已停掉旧 worker 会话 {sess}", flush=True)
        time.sleep(5)


def _inner(domain, env_extra, gpu, tag, log):
    return (f"source ~/miniconda/etc/profile.d/conda.sh && conda activate freechunker && "
            f"export PYTHONPATH='{ROOT}' && "
            f"export CUDA_VISIBLE_DEVICES='{gpu}' && "
            "export TOKENIZERS_PARALLELISM=false && "
            "export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1 && "
            f"export FC_CM_DOMAINS='{domain}' && "
            f"{env_extra} "
            f"cd '{ROOT}' && {PY} '{SCRIPT}' "
            f"> {log} 2>&1; echo EXIT=$? >> {log}; sleep 99999")


def launch(domain, shard, nshards, gpu):
    log = os.path.join(LOGS, f"margin_{domain}_sh{shard}of{nshards}_gpu{gpu}.log")
    sess = f"mz_{domain}_s{shard}_g{gpu}"
    env = (f"export FC_CM_SHARD='{shard}' && export FC_CM_NSHARDS='{nshards}' && ")
    subprocess.run(["tmux", "kill-session", "-t", sess], stderr=subprocess.DEVNULL)
    subprocess.run(["tmux", "new-session", "-d", "-s", sess,
                    _inner(domain, env, gpu, f"s{shard}", log)], check=False)
    launched[(domain, shard)] = gpu
    print(f"[mz] 启动 {domain} shard{shard}/{nshards} -> GPU{gpu}  ({sess})", flush=True)


def launch_gap(domain, idxs, claim_skip, gpu, tag):
    """续切：显式指定下标跑，不碰正在跑的 worker 认领的下标。"""
    if not idxs:
        return
    log = os.path.join(LOGS, f"margin_{domain}_gap_{tag}_gpu{gpu}.log")
    sess = f"mzg_{domain}_{tag}_g{gpu}"
    env = (f"export FC_CM_INDEXES='{','.join(str(i) for i in idxs)}' && "
           f"export FC_CM_SKIP='{','.join(str(i) for i in sorted(claim_skip))}' && ")
    subprocess.run(["tmux", "kill-session", "-t", sess], stderr=subprocess.DEVNULL)
    subprocess.run(["tmux", "new-session", "-d", "-s", sess,
                    _inner(domain, env, gpu, tag, log)], check=False)
    gap_launched[sess] = (domain, idxs)
    print(f"[mz] 续切 {domain} {len(idxs)}条 -> GPU{gpu}  ({sess})  "
          f"idx={idxs[:4]}{'...' if len(idxs) > 4 else ''}", flush=True)


def pending_shards():
    """还没启动过、且域未完成的分片。"""
    out = []
    for dom, n in PLAN.items():
        if len(done_ids(dom)) >= TOTALS[dom]:
            continue
        for s in range(n):
            if (dom, s) not in launched:
                out.append((dom, s, n))
    return out


def main():
    stop_old_single_doc()
    print("[mz] single-doc / long-icl 各 4 片；空卡出现时自动续切剩余缺口",
          flush=True)

    while True:
        busy = gpu_busy()
        free = [g for g in USABLE_GPUS if g not in busy]
        pend = pending_shards()

        # 第一优先：还没启动的分片铺上去
        for gpu in free:
            if not pend:
                break
            dom, shard, n = pend.pop(0)
            launch(dom, shard, n, gpu)
            time.sleep(3)
            free = [g for g in USABLE_GPUS if g not in gpu_busy()]
        if pend:
            pass  # 卡不够，下一轮再说

        # 第二优先：还有空卡 + 缺口够大 -> 续切
        if GAP_RESHARD:
            busy = gpu_busy()
            free = [g for g in USABLE_GPUS if g not in busy]
            if len(free) >= GAP_MIN_FREE_GPUS and not pending_shards():
                tasks = []
                for dom in PLAN:
                    done = done_ids(dom)
                    if len(done) >= TOTALS[dom]:
                        continue
                    skip = live_claimed_indexes(dom)
                    gap = [i for i in gap_indexes(dom, done) if i not in skip]
                    if len(gap) >= GAP_MIN_REMAIN:
                        tasks.append((dom, gap, skip))
                if tasks:
                    k = len(free)
                    for t_i, (dom, gap, skip) in enumerate(tasks):
                        chunks = [gap[j::k] for j in range(k)]
                        for j, ch in enumerate(chunks):
                            if ch and j < len(free):
                                launch_gap(dom, ch, skip, free[j],
                                           f"{dom[:4]}{time.strftime('%H%M')}{j}")
                            time.sleep(3)

        busy = gpu_busy()
        free = [g for g in USABLE_GPUS if g not in busy]
        st = {d: f"{len(done_ids(d))}/{TOTALS[d]}" for d in PLAN}
        left = [d for d in PLAN if len(done_ids(d)) < TOTALS[d]]
        print(f"[mz] {time.strftime('%H:%M:%S')} 进度 {st}  空卡{free}", flush=True)
        if not left and not pending_shards():
            print("[mz] MARGIN_LONGTAIL_DONE", flush=True)
            return 0
        time.sleep(120)


if __name__ == "__main__":
    sys.exit(main())
