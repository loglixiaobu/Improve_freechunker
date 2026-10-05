#!/usr/bin/env bash
# 启动 vLLM OpenAI 兼容服务（Qwen3-8B，单卡，端口 8888）
# 下游评测的所有 QA 生成都走这个服务。
set -euo pipefail

source ~/miniconda/etc/profile.d/conda.sh
conda activate freechunker

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export VLLM_DISABLE_COMPILATION=1
export TORCH_COMPILE_DISABLE=1

ROOT="${FC_ROOT:-$HOME/FreeChunker}"
GPU="${FC_VLLM_GPU:-0}"
PORT="${FC_VLLM_PORT:-8888}"

cd "$ROOT"

# 用 repo 里下载好的本地权重，避免联网
MODEL_PATH="${FC_QWEN3_LOCAL:-$ROOT/models/Qwen3-8B}"

echo "[vLLM] root=$ROOT gpu=$GPU port=$PORT model=$MODEL_PATH"

exec env CUDA_VISIBLE_DEVICES="$GPU" python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name Qwen/Qwen3-8B \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype auto \
    --enforce-eager \
    --gpu-memory-utilization "${FC_VLLM_GMU:-0.92}" \
    --max-model-len "${FC_VLLM_MAXLEN:-16384}"
