"""修 5_chunk_lumber.py：tokenizer 用本地 Qwen3-8B，但 vLLM 的 model 名保持
'Qwen/Qwen3-8B'（服务端就是这个 served-model-name）。幂等。"""
P = "/home1/lh/FreeChunker/train and preprocess/5_chunk_lumber.py"
s = open(P, encoding="utf-8").read()
MARK = "# fc-patch: lumber local tokenizer"

if MARK in s:
    print("ALREADY_PATCHED")
    raise SystemExit(0)

OLD = 'model_name_or_path = "Qwen/Qwen3-8B"'
NEW = (
    MARK + "\n"
    "_QWEN3_LOCAL = _os.path.join(_MODELS, 'Qwen3-8B')\n"
    "model_name_or_path = _QWEN3_LOCAL if _os.path.isdir(_QWEN3_LOCAL) else 'Qwen/Qwen3-8B'\n"
    "model_served_name = 'Qwen/Qwen3-8B'  # vLLM --served-model-name，保持不变"
)
if OLD not in s:
    raise SystemExit("model_name_or_path 行没找到")
s = s.replace(OLD, NEW, 1)

# self.model_path 用于请求 vLLM，必须用 served name 而不是本地路径
old_mp = "        self.model_path = model_name_or_path"
new_mp = "        self.model_path = model_served_name"
if old_mp not in s:
    raise SystemExit("self.model_path 行没找到")
s = s.replace(old_mp, new_mp, 1)

open(P, "w", encoding="utf-8").write(s)
print("LUMBER_PATCHED")
