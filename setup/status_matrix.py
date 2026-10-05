"""全量下游 chunking 状态盘点：6 方法 x 6 域。"""
import os
from datasets import load_from_disk

ROOT = "/home1/lh/FreeChunker/LongBench-v2_chunked"
DOMAINS = ["single-doc", "multi-doc", "code-repo",
           "long-icl", "long-dialogue", "long-structured"]
EXP = {"single-doc": 175, "multi-doc": 125, "code-repo": 50,
       "long-icl": 81, "long-dialogue": 39, "long-structured": 33}

# 方法 -> 输出基目录
METHODS = {
    "FreeChunker": None,  # 不走 chunked 目录，靠 encoder 现场分块
    "Traditional/256": os.path.join(ROOT, "Traditional", "256"),
    "Traditional/512": os.path.join(ROOT, "Traditional", "512"),
    "Semantic": os.path.join(ROOT, "Semantic", "bge-m3"),
    "PPL": os.path.join(ROOT, "PPL", "Qwen2.5-1.5B-Instruct"),
    "Margin": os.path.join(ROOT, "Margin", "Qwen2.5-1.5B-Instruct"),
    "Lumber": os.path.join(ROOT, "Lumber", "qwen3-8b"),
}


def probe(base, d):
    """返回 (状态字符串, 是否完成)。"""
    p = os.path.join(base, d)
    if not os.path.isdir(p):
        # 可能有分片目录
        try:
            shards = [e for e in os.listdir(base) if e.startswith(d + "")]
        except Exception:
            return "(无目录)", False
        if shards:
            return "分片未合并: " + ",".join(sorted(shards)[:4]), False
        return "(缺失)", False
    try:
        ds = load_from_disk(p)
    except Exception as e:
        return "加载失败: {}".format(str(e)[:40]), False
    n = len(ds)
    try:
        ids = set(ds["_id"])
    except Exception:
        ids = set()
    dup = n - len(ids) if ids else 0
    exp = EXP[d]
    ok = (n == exp and dup == 0)
    return "{:4d}/{:<4d}{}".format(n, exp, "  DONE" if ok else "  重复{}".format(dup)), ok


print("{:18s}".format("方法") + "".join("{:>22s}".format(d) for d in DOMAINS))
print("-" * (18 + 22 * len(DOMAINS)))
for m, base in METHODS.items():
    if base is None:
        print("{:18s}".format(m) + "{:>22s}".format("(另容器，见评测)") * len(DOMAINS))
        continue
    row = "{:18s}".format(m)
    for d in DOMAINS:
        s, _ = probe(base, d)
        row += "{:>22s}".format(s)
    print(row)
