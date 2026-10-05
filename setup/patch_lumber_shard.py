"""给 5_chunk_lumber.py 加样本级分片 + 端口选择。

为什么要改：Lumber 每篇文档要发 N 次 vLLM 请求（实测单篇 116 次），
但它是**串行**的 —— vLLM 侧 `Running: 0~1 reqs`，GPU 空转。
按样本分片跑多个 worker（各占一个 vLLM 端口），wall-clock 直接除以 worker 数。

改动：
  1. FC_LMB_SHARD / FC_LMB_NSHARDS  -> 样本级分片，checkpoint 与输出带 .shardN 后缀
  2. FC_VLLM_PORT                   -> base_url 端口

幂等：看 MARK。
"""
P = "/home1/lh/FreeChunker/train and preprocess/5_chunk_lumber.py"
s = open(P, encoding="utf-8").read()
MARK = "# fc-patch: lumber sharding"

if MARK in s:
    print("ALREADY_PATCHED")
    raise SystemExit(0)

# 1) 常量插到 fc-patch: lumber local tokenizer 之后
ANCHOR = "model_served_name = 'Qwen/Qwen3-8B'  # vLLM --served-model-name，保持不变"
if ANCHOR not in s:
    raise SystemExit("找不到 anchor")
INS = (
    ANCHOR + "\n"
    + MARK + "\n"
    "_LMB_SHARD = int(_os.environ.get('FC_LMB_SHARD', '0'))\n"
    "_LMB_NSHARDS = int(_os.environ.get('FC_LMB_NSHARDS', '1'))\n"
    "_LMB_SUFFIX = '' if _LMB_NSHARDS == 1 else f'.shard{_LMB_SHARD}'\n"
    "_VLLM_PORT = _os.environ.get('FC_VLLM_PORT', '8888')\n"
)
s = s.replace(ANCHOR, INS, 1)

# 2) base_url 用端口变量
old_url = 'base_url="http://localhost:8888/v1"'
new_url = 'base_url=f"http://localhost:{_VLLM_PORT}/v1"'
if old_url not in s:
    raise SystemExit("base_url 行没找到")
s = s.replace(old_url, new_url, 1)

# 3) checkpoint 路径加后缀
old_ck = 'checkpoint_path = os.path.join(checkpoints_root, f"{dn}.jsonl")'
new_ck = 'checkpoint_path = os.path.join(checkpoints_root, f"{dn}{_LMB_SUFFIX}.jsonl")'
if old_ck not in s:
    raise SystemExit("checkpoint 行没找到")
s = s.replace(old_ck, new_ck, 1)

# 4) 样本循环分片
old_loop = "        for i, sample in enumerate(task_pbar):\n            if sample['_id'] in processed_samples:"
new_loop = (
    "        for i, sample in enumerate(task_pbar):\n"
    "            if _LMB_NSHARDS > 1 and (i % _LMB_NSHARDS) != _LMB_SHARD:\n"
    "                continue\n"
    "            if sample['_id'] in processed_samples:"
)
if old_loop not in s:
    raise SystemExit("样本循环没找到")
s = s.replace(old_loop, new_loop, 1)

# 5) 输出目录加后缀
old_out = 'out_dir = os.path.join(new_dataset_root, dn)'
new_out = 'out_dir = os.path.join(new_dataset_root, dn + _LMB_SUFFIX)'
if old_out not in s:
    raise SystemExit("输出目录行没找到")
s = s.replace(old_out, new_out, 1)

# 6) dataset_dict.json 加后缀
old_idx = 'index_path = os.path.join(new_dataset_root, "dataset_dict.json")'
new_idx = 'index_path = os.path.join(new_dataset_root, f"dataset_dict{_LMB_SUFFIX}.json")'
if old_idx not in s:
    raise SystemExit("index 行没找到")
s = s.replace(old_idx, new_idx, 1)

open(P, "w", encoding="utf-8").write(s)
print("LUMBER_SHARD_PATCHED")
