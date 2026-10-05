#!/usr/bin/env bash
# 安装 chunker 依赖：lmchunker(PPL/Margin) + langchain-experimental(Semantic) + jieba(PPL)
set -euo pipefail

source ~/miniconda/etc/profile.d/conda.sh
conda activate freechunker

export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

echo "=== pip 版本 ==="
python -m pip --version

echo "=== 安装 jieba langchain-experimental langchain-core ==="
python -m pip install -q jieba langchain-experimental langchain-core langchain-text-splitters

echo "=== 安装 lmchunker ==="
python -m pip install -q lmchunker || echo "lmchunker 安装失败，稍后手动处理"

echo "=== 校验 ==="
python - <<'PYEOF'
import importlib
for m in ["jieba", "langchain_experimental", "langchain_core", "lmchunker"]:
    try:
        importlib.import_module(m)
        print(m, "OK")
    except Exception as e:
        print(m, "FAIL", type(e).__name__, e)
PYEOF

echo "INSTALL_DONE"
