"""给 7_chunk_semantic.py 加样本级分片支持（FC_SEM_SHARD / FC_SEM_NSHARDS），
并在收尾时支持"分片写独立目录 + 事后合并"。

改动最小化：
  1. 样本循环里跳过 index % nshards != shard 的样本
  2. checkpoint 文件名带 .shardN 后缀，避免多进程写同一文件
  3. save_to_disk 输出目录带 .shardN 后缀，避免互相覆盖（由 merge 脚本合并）

幂等：看 MARK。
"""
P = "/home1/lh/FreeChunker/train and preprocess/7_chunk_semantic.py"
s = open(P, encoding="utf-8").read()
MARK = "# fc-patch: sample sharding"

if MARK in s:
    print("ALREADY_PATCHED")
    raise SystemExit(0)

# 1. 在 import 后插入分片常量
ANCHOR = "# fc-patch: domain sharding"
if ANCHOR not in s:
    raise SystemExit("需要先打 domain sharding 补丁")

INS = (
    MARK + "\n"
    "_SEM_SHARD = int(_os.environ.get('FC_SEM_SHARD', '0'))\n"
    "_SEM_NSHARDS = int(_os.environ.get('FC_SEM_NSHARDS', '1'))\n"
    "_SEM_SUFFIX = '' if _SEM_NSHARDS == 1 else f'.shard{_SEM_SHARD}'\n"
)
s = s.replace(ANCHOR, ANCHOR + "\n" + INS.rstrip("\n"), 1)

# 2. checkpoint 路径加后缀
old_ck = 'checkpoint_path = os.path.join(checkpoints_root, f"{dn}.jsonl")'
new_ck = 'checkpoint_path = os.path.join(checkpoints_root, f"{dn}{_SEM_SUFFIX}.jsonl")'
if old_ck not in s:
    raise SystemExit("checkpoint 行没找到")
s = s.replace(old_ck, new_ck, 1)

# 3. 样本循环里做 index 分片
old_loop = "            for i, sample in enumerate(task_pbar):\n                if sample['_id'] in processed_samples:"
new_loop = (
    "            for i, sample in enumerate(task_pbar):\n"
    "                if _SEM_NSHARDS > 1 and (i % _SEM_NSHARDS) != _SEM_SHARD:\n"
    "                    continue\n"
    "                if sample['_id'] in processed_samples:"
)
if old_loop not in s:
    raise SystemExit("样本循环没找到")
s = s.replace(old_loop, new_loop, 1)

# 4. 输出目录加后缀
old_out = 'out_dir = os.path.join(new_dataset_root, dn)'
new_out = 'out_dir = os.path.join(new_dataset_root, dn + _SEM_SUFFIX)'
if old_out not in s:
    raise SystemExit("输出目录行没找到")
s = s.replace(old_out, new_out, 1)

# 5. dataset_dict.json 也加后缀，避免互相覆盖
old_idx = 'index_path = os.path.join(new_dataset_root, "dataset_dict.json")'
new_idx = 'index_path = os.path.join(new_dataset_root, f"dataset_dict{_SEM_SUFFIX}.json")'
if old_idx not in s:
    raise SystemExit("index 行没找到")
s = s.replace(old_idx, new_idx, 1)

open(P, "w", encoding="utf-8").write(s)
print("SEM_SAMPLE_SHARD_PATCHED")
