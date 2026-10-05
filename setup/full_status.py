"""全量盘点：6 方法 x 6 域，扫描 chunked 目录 + checkpoint 目录。

关键：Margin 的 checkpoint 文件名有两种形态
  - `{domain}.jsonl`        —— FC_CM_NSHARDS=1 时
  - `{domain}.shard{N}.jsonl` —— nshards>1 时
必须都扫，且合并去重后再判断是否完成。
"""
import json
import os

from datasets import load_from_disk

ROOT = "/home1/lh/FreeChunker/LongBench-v2_chunked"
DOMAINS = ["single-doc", "multi-doc", "code-repo",
           "long-icl", "long-dialogue", "long-structured"]
EXP = {"single-doc": 175, "multi-doc": 125, "code-repo": 50,
       "long-icl": 81, "long-dialogue": 39, "long-structured": 33}

METHODS = [
    ("FreeChunker", None),
    ("Traditional/256", os.path.join(ROOT, "Traditional", "256")),
    ("Traditional/512", os.path.join(ROOT, "Traditional", "512")),
    ("Semantic", os.path.join(ROOT, "Semantic", "bge-m3")),
    ("PPL", os.path.join(ROOT, "PPL", "Qwen2.5-1.5B-Instruct")),
    ("Margin", os.path.join(ROOT, "Margin", "Qwen2.5-1.5B-Instruct")),
    ("Lumber", os.path.join(ROOT, "Lumber", "qwen3-8b")),
]


def scan_final(base, d):
    """最终数据集目录（已 save_to_disk）。"""
    p = os.path.join(base, d)
    if not os.path.isdir(p):
        return None
    try:
        ds = load_from_disk(p)
    except Exception:
        return None
    return len(ds), len(set(ds["_id"]))


def scan_ckpt(base, d):
    """checkpoint jsonl（未落盘前的过程产物），含 .shardN。"""
    ck = os.path.join(base, "checkpoints")
    if not os.path.isdir(ck):
        return 0
    ids = set()
    for f in os.listdir(ck):
        if not f.endswith(".jsonl"):
            continue
        stem = f[:-len(".jsonl")]
        if stem != d and not stem.startswith(d + ".shard"):
            continue
        with open(os.path.join(ck, f)) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line)["_id"])
                    except Exception:
                        pass
    return len(ids)


print("=" * 130)
print("{:17s}".format("方法") + "".join("{:>18s}".format(d) for d in DOMAINS) + "{:>12s}".format("小计"))
print("=" * 130)
grand = 0
for m, base in METHODS:
    if base is None:
        print("{:17s}".format(m) + "{:>18s}".format("—") * len(DOMAINS) + "{:>12s}".format("(另行)"))
        continue
    row = "{:17s}".format(m)
    sub = 0
    for d in DOMAINS:
        fin = scan_final(base, d)
        ck = scan_ckpt(base, d)
        exp = EXP[d]
        if fin and fin[0] >= exp and fin[1] == fin[0]:
            row += "{:>18s}".format("{} 完成".format(fin[0]))
            sub += exp
        elif ck >= exp:
            row += "{:>18s}".format("{} 待合并".format(ck))
            sub += exp
        elif ck > 0:
            row += "{:>18s}".format("{}/{} 进行".format(ck, exp))
            sub += ck
        else:
            row += "{:>18s}".format("-")
        grand += 0
    print(row + "{:>12s}".format("{}/503".format(sub)))
print("=" * 130)
