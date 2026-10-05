#!/bin/bash
# 为 FreeChunker 复现新建独立 conda 环境 freechunker（不复用任何已有环境）
# 版本对齐本机已验证可用的 dcs 环境
set -x
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_DISABLE_PIP_VERSION_CHECK=1
ROOT=/home1/lh/FreeChunker
LOGS=$ROOT/logs

source /home1/lh/miniconda/etc/profile.d/conda.sh

echo "===== [1/7] create env ====="
conda create -y -n freechunker python=3.10 || { echo "FATAL: conda create failed"; exit 1; }
conda activate freechunker || { echo "FATAL: activate failed"; exit 1; }
PY=$(which python)
echo "python: $PY  $($PY -V)"

echo "===== [2/7] upgrade pip ====="
$PY -m pip install -U pip setuptools wheel

echo "===== [3/7] bootstrap huggingface_hub，然后立刻启动资产下载（与后面的安装并行） ====="
$PY -m pip install "huggingface_hub==0.36.0" || { echo "FATAL: hf_hub install failed"; exit 2; }

cd $ROOT
HF_ENDPOINT=https://hf-mirror.com \
  setsid nohup $PY "$LOGS/download_assets.py" \
  > "$LOGS/download_assets.log" 2>&1 < /dev/null &
sleep 5
echo "下载进程已启动，pid=$(pgrep -f 'download_assets\.py' | head -1)"
tail -3 "$LOGS/download_assets.log" 2>/dev/null

echo "===== [4/7] torch stack ====="
$PY -m pip install \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  || { echo "FATAL: torch install failed"; exit 3; }

echo "===== [5/7] core libs ====="
$PY -m pip install \
  transformers==4.57.1 tokenizers==0.22.1 \
  sentence-transformers==3.2.0 datasets==4.4.1 \
  accelerate==1.11.0 numpy==2.2.6 pandas==2.3.3 \
  scikit-learn==1.7.2 matplotlib==3.10.7 tqdm==4.66.5 \
  einops==0.8.1 nltk==3.9.1 pyarrow==22.0.0 sentencepiece==0.2.1 \
  safetensors==0.6.2 openai==2.24.0 \
  json_repair \
  || { echo "FATAL: core libs install failed"; exit 4; }

echo "===== [6/7] vllm ====="
$PY -m pip install vllm==0.16.0 || { echo "FATAL: vllm install failed"; exit 5; }

echo "===== [7/7] verify ====="
$PY - <<'PY'
import importlib.metadata as md, sys
mods = ["torch","torchvision","transformers","tokenizers","sentence-transformers",
        "datasets","accelerate","numpy","pandas","scikit-learn","matplotlib","tqdm",
        "einops","nltk","pyarrow","sentencepiece","safetensors","huggingface-hub",
        "openai","json_repair","vllm"]
bad = []
for m in mods:
    try: print(f"  OK  {m}=={md.version(m)}")
    except Exception as e:
        print(f"  !!  {m}: {e}"); bad.append(m)
import torch
print("torch.cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("capability:", torch.cuda.get_device_capability(0), "| name:", torch.cuda.get_device_name(0))
import json_repair, einops, sentence_transformers, datasets, vllm
print("imports ok; python", sys.version.split()[0])
print("MISSING:", bad if bad else "none")
PY
echo "===== ENV_SETUP_DONE ====="
