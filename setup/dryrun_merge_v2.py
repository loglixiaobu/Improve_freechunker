import importlib.util
import os

spec = importlib.util.spec_from_file_location("m", "setup/merge_cm_shards.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print("导入 OK")

for sub, model in [("Margin", "Qwen2.5-1.5B-Instruct"), ("PPL", "Qwen2.5-1.5B-Instruct")]:
    root = f"LongBench-v2_chunked/{sub}/{model}"
    print("====", sub)
    for dn in m.DOMAINS:
        plain, shards = m._dir_candidates(root, dn)
        cks = m._ckpt_files(root, dn)
        shard_cks = [p for p in cks if os.path.basename(p)[:-len(".jsonl")] != dn]
        do, why = m._decide(plain, shards, shard_cks, cks)
        n_dir = len(shards) + (1 if plain else 0)
        print(f"  {dn:16s} 目录来源={n_dir} 分片ckpt={len(shard_cks)} "
              f"本体ckpt={'Y' if len(cks) > len(shard_cks) else 'N'} -> {why}")
