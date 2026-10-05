import collections
import glob
import json
import os

os.chdir("/home1/lh/FreeChunker")
TOT = {"single-doc": 175, "multi-doc": 125, "long-icl": 81,
       "code-repo": 50, "long-dialogue": 39, "long-structured": 33}


def uniq_ids(base, dom):
    ids = set()
    for f in glob.glob(os.path.join(base, dom + "*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line)["_id"])
                    except Exception:
                        pass
    return len(ids)


def line_for(d, n):
    mark = "DONE" if n >= TOT[d] else ""
    return "  {:16s} {:4d}/{:<4d} {}".format(d, n, TOT[d], mark)


print("=== PPL ===")
b = "LongBench-v2_chunked/PPL/Qwen2.5-1.5B-Instruct/checkpoints"
tot = 0
for d in TOT:
    n = uniq_ids(b, d)
    tot += n
    print(line_for(d, n))
print("  小计 {}/503".format(tot))

print("\n=== Margin ===")
b = "LongBench-v2_chunked/Margin/Qwen2.5-1.5B-Instruct/checkpoints"
mt = 0
for d in TOT:
    n = uniq_ids(b, d)
    mt += n
    if n:
        print(line_for(d, n))
print("  小计 {}/503".format(mt))

print("\n=== Lumber (全局池) ===")
gp = "LongBench-v2_chunked/Lumber/qwen3-8b/checkpoints/_done_global.jsonl"
c = collections.Counter()
with open(gp) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                c[json.loads(line).get("domain", "?")] += 1
            except Exception:
                pass
for k, v in sorted(c.items()):
    print("  {}: {}".format(k, v))
print("  合计 {}/503".format(sum(c.values())))
