"""让 7_chunk_semantic.py 使用本地 bge-m3 目录。幂等。"""
import os

P = "/home1/lh/FreeChunker/train and preprocess/7_chunk_semantic.py"
s = open(P, encoding="utf-8").read()

OLD = "embed_models = [\n    'BAAI/bge-m3',\n]"
NEW = (
    "# fc-patch: 本地 bge-m3\n"
    "embed_models = [\n"
    "    _os.path.join(_MODELS, 'bge-m3') if _os.path.isdir(_os.path.join(_MODELS, 'bge-m3')) else 'BAAI/bge-m3',\n"
    "]"
)

if OLD in s:
    s = s.replace(OLD, NEW)
    open(P, "w", encoding="utf-8").write(s)
    print("SEMANTIC_PATCHED")
elif "_os.path.join(_MODELS, 'bge-m3')" in s:
    print("ALREADY_PATCHED")
else:
    i = s.find("embed_models")
    print("PATTERN_NOT_FOUND", repr(s[i:i + 90]))
