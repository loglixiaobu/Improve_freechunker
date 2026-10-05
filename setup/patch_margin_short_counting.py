"""修 run_margin_short.py 的 done_ids：要同时认 {d}.jsonl 和 {d}.shard*.jsonl。

原实现只读 {domain}.shard0.jsonl，但当 worker 以 FC_CM_NSHARDS=1 启动时，
产物写的是 {domain}.jsonl，导致计数永远读到旧值。幂等。
"""
import sys

P = "/home1/lh/FreeChunker/setup/run_margin_short.py"
OLD = '''def done_ids(domain):
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
'''

NEW = '''def done_ids(domain):
    """同时认 {domain}.jsonl 与 {domain}.shard*.jsonl 两种形态，按 _id 去重。

    ⚠️ 踩坑记录：早先只读 {domain}.shard0.jsonl，但 worker 以 FC_CM_NSHARDS=1
    启动时产物写的是 {domain}.jsonl —— 于是计数一直读到冒烟残留的旧值，
    明明跑到 36/39 却一直报 2/39。
    """
    ck = os.path.join(ROOT, "LongBench-v2_chunked", "Margin", "Qwen2.5-1.5B-Instruct",
                      "checkpoints")
    if not os.path.isdir(ck):
        return 0
    ids = set()
    for fn in os.listdir(ck):
        if not fn.endswith(".jsonl"):
            continue
        stem = fn[:-len(".jsonl")]
        if stem != domain and not stem.startswith(domain + ".shard"):
            continue
        with open(os.path.join(ck, fn)) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line)["_id"])
                    except Exception:
                        pass
    return len(ids)
'''

src = open(P, encoding="utf-8").read()
if "两种形态" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)
if OLD not in src:
    print("ANCHOR_NOT_FOUND")
    print([l for l in src.splitlines() if "shard0.jsonl" in l])
    sys.exit(1)
src = src.replace(OLD, NEW, 1)
open(P, "w", encoding="utf-8").write(src)
print("PATCH_OK")
