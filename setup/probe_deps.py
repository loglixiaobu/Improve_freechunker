"""探测 chunker 依赖是否齐备。"""
import importlib

mods = [
    "lmchunker",
    "langchain_experimental",
    "langchain_core",
    "nltk",
    "jieba",
    "sentence_transformers",
    "openai",
    "transformers",
    "datasets",
    "torch",
]
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        print(f"{m:28s} OK   {ver}")
    except Exception as e:
        print(f"{m:28s} FAIL {type(e).__name__}: {e}")

import nltk  # noqa: E402

for res in ["tokenizers/punkt", "tokenizers/punkt_tab"]:
    try:
        nltk.data.find(res)
        print(f"nltk {res:26s} OK")
    except Exception:
        print(f"nltk {res:26s} MISSING")

import torch  # noqa: E402

print("torch cuda:", torch.cuda.is_available(), torch.cuda.device_count())
print("PROBE_DONE")
